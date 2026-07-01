//! Simplified 2D cloud movement on an adaptive-resolution grid.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::io::Write;
use stormcell::{
    allocator::Address,
    grid::{CellData, Grid, Grid2d, GridConfig},
    pipeline::Transformer,
    quantizer::{CellContext, Quantizer, kernel::KernelQuantizer},
    topology::Topology2d,
};

fn main() {
    print!("\x1b[2J");

    let mut grid = Grid2d::<Air>::new(
        GridConfig {
            chunks_num: CHUNKS,
            chunk_max_depth: CHUNK_DEPTH,
            sampler_cache_limit: Some(512),
            ..Default::default()
        },
        Air::CLEAR,
    );

    // Build the fbm moisture source once (it owns permutation tables); the
    // per-step quantizer just borrows it.
    let moisture = Fbm::<Perlin>::new(MOIST_SEED).set_octaves(3);

    let mut tick: u64 = 0;
    loop {
        render(&grid);
        std::thread::sleep(std::time::Duration::from_millis(DELAY_MS));

        // One simulation step: advect, moisten + condense over the whole grid.
        // The wind and moisture fields are time-varying, so the quantizer is
        // rebuilt each step with the current tick.
        let kernel = KernelQuantizer {
            quantizer: Clouds {
                time: tick as f32,
                moisture: &moisture,
            },
        };
        grid = Transformer::new(&kernel, &grid).execute();
        tick = tick.wrapping_add(1);
    }
}

// 6x3 chunks ...
const CHUNKS: [usize; 2] = [6, 3];
// ... of 8x8 cells each => 48x24 grid
const CHUNK_DEPTH: usize = 3;
const DELAY_MS: u64 = 50;

// Prevailing breeze, in cells per step. Kept small so clouds drift slowly.
const WIND_DRIFT: [f32; 2] = [0.12, 0.05];
// Amplitude of the gentle undulation added on top of the breeze (cells/step), so
// the flow meanders instead of sliding in perfectly straight lines.
const WIND_WAVE: f32 = 0.2;

// Humidity at or above this condenses into cloud; below it, cloud evaporates.
const SATURATION: f32 = 1.0;
// Fraction of the super-saturated excess that condenses each step.
const CONDENSE_RATE: f32 = 0.25;
// Fraction of cloud that evaporates each step while sub-saturated.
const EVAP_RATE: f32 = 0.06;
// Fraction of cloud that survives each step (the rest rains out).
const CLOUD_RETENTION: f32 = 0.97;
// Fraction of humidity retained each step (slow drying keeps it bounded).
const HUMIDITY_RETENTION: f32 = 0.99;

// Patchy moisture source (Perlin fbm sampled per cell):
// seed for the noise permutation tables.
const MOIST_SEED: u32 = 0;
// spatial scale of the damp patches - larger makes them smaller and more numerous.
const MOIST_FREQ: f32 = 0.19;
// how fast the patches well up and fade - small so puffs persist and drift.
const MOIST_TIME: f32 = 0.012;
// only fbm (remapped to 0..1) above this injects moisture, so patches stay
// scattered. The fbm output clusters tightly around 0.5, so this sits in its
// upper tail.
const MOIST_THRESHOLD: f32 = 0.6;
// band of fbm above the threshold over which moisture ramps to full strength.
const MOIST_SPAN: f32 = 0.15;
// humidity added per step where the noise is at full strength.
const MOIST_RATE: f32 = 0.25;

// Cells merge into one coarse cell when both fields are within tolerance. Cloud
// is what we see, so its tolerance is tight to keep edges crisp; humidity is
// invisible, so a loose tolerance lets clear-but-humid sky still collapse
// (otherwise smooth humidity gradients would shatter the whole grid).
const CLOUD_EPS: f32 = 0.02;
const HUMIDITY_EPS: f32 = 0.35;

/// Per-cell air state: invisible water vapour plus condensed cloud droplets.
#[derive(Clone, Copy)]
struct Air {
    humidity: f32,
    cloud: f32,
}

impl Air {
    const CLEAR: Air = Air {
        humidity: 0.0,
        cloud: 0.0,
    };

    /// Linear blend between two air states, used for bilinear advection.
    fn lerp(a: Air, b: Air, t: f32) -> Air {
        Air {
            humidity: a.humidity + (b.humidity - a.humidity) * t,
            cloud: a.cloud + (b.cloud - a.cloud) * t,
        }
    }
}

impl CellData for Air {
    fn are_homogeneous(&self, other: &Self) -> bool {
        (self.cloud - other.cloud).abs() < CLOUD_EPS
            && (self.humidity - other.humidity).abs() < HUMIDITY_EPS
    }

    fn scale(&self, multiplier: usize) -> Self {
        // Both fields are per-cell intensities; scaling by a leaf's area gives
        // the total the leaf represents, matching the heat-diffusion example.
        let m = multiplier as f32;
        Air {
            humidity: self.humidity * m,
            cloud: self.cloud * m,
        }
    }
}

/// Advection + moisture + condensation rule. `time` drives the time-varying wind
/// and moisture fields; `moisture` is the shared fbm noise source.
struct Clouds<'a> {
    time: f32,
    moisture: &'a Fbm<Perlin>,
}

impl Clouds<'_> {
    /// Analytic wind at a fractional grid position, in cells per step. A slow
    /// prevailing breeze plus a gentle sinusoidal meander.
    fn wind(&self, x: f32, y: f32) -> [f32; 2] {
        let t = self.time;
        let wx = WIND_DRIFT[0] + WIND_WAVE * (y * 0.13 + t * 0.02).sin();
        let wy = WIND_DRIFT[1] + WIND_WAVE * (x * 0.11 - t * 0.017).sin();
        [wx, wy]
    }

    /// Humidity injected at a cell this step: a patchy, animated fbm field (the
    /// third axis is time), thresholded so only the damper regions well up moisture.
    fn moisture_at(&self, x: f32, y: f32) -> f32 {
        // fbm returns roughly -1..1; remap to 0..1 before thresholding.
        let raw = self.moisture.get([
            (x * MOIST_FREQ) as f64,
            (y * MOIST_FREQ) as f64,
            (self.time * MOIST_TIME) as f64,
        ]);
        let wet = raw as f32 * 0.5 + 0.5;
        let m = ((wet - MOIST_THRESHOLD) / MOIST_SPAN).clamp(0.0, 1.0);
        m * MOIST_RATE
    }
}

impl Quantizer for Clouds<'_> {
    type CellData = Air;
    type Topology = Topology2d;

    fn quantize(&self, ctx: CellContext<Air, Topology2d>) -> Address {
        let [cx, cy] = ctx.region.start;
        let [w, h] = ctx.sampler.grid_size();
        let px = cx as f32 + 0.5;
        let py = cy as f32 + 0.5;

        // Semi-Lagrangian back-trace: find where this parcel of air came from.
        let [wx, wy] = self.wind(px, py);
        let sx = (px - wx).clamp(0.0, w as f32 - 1.0);
        let sy = (py - wy).clamp(0.0, h as f32 - 1.0);

        // Bilinearly sample both fields at the back-traced point.
        let x0 = sx.floor() as usize;
        let y0 = sy.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = sx - x0 as f32;
        let ty = sy - y0 as f32;
        let c00 = *ctx.sampler.sample([x0, y0]).unwrap().data;
        let c10 = *ctx.sampler.sample([x1, y0]).unwrap().data;
        let c01 = *ctx.sampler.sample([x0, y1]).unwrap().data;
        let c11 = *ctx.sampler.sample([x1, y1]).unwrap().data;
        let top = Air::lerp(c00, c10, tx);
        let bottom = Air::lerp(c01, c11, tx);
        let mut air = Air::lerp(top, bottom, ty);

        // Moisture source: add humidity from the patchy noise field, leaving the
        // advected cloud untouched, then dry the air a touch so it stays bounded.
        air.humidity += self.moisture_at(px, py);
        air.humidity *= HUMIDITY_RETENTION;

        // Condensation/evaporation around the saturation threshold.
        let excess = air.humidity - SATURATION;
        if excess > 0.0 {
            let condensed = excess * CONDENSE_RATE;
            air.humidity -= condensed;
            air.cloud += condensed;
        } else {
            let evaporated = air.cloud * EVAP_RATE;
            air.cloud -= evaporated;
            air.humidity += evaporated;
        }

        // Cloud slowly rains out so it cannot pile up without bound.
        air.cloud *= CLOUD_RETENTION;

        ctx.emitter.emit_leaf(air)
    }
}

fn render(grid: &Grid<Air, Topology2d>) {
    let flat = grid.flatten(Air::CLEAR);
    let [w, h] = flat.size();
    let fields = flat.fields();
    let total = w * h;
    let cells = grid.iter().count();

    let mut out = String::with_capacity(total * 24);
    // cursor home (overwrite previous frame)
    out.push_str("\x1b[H");
    out.push_str(&format!(
        "Cells: {cells:>4} / {total} ({:>2}%)\n",
        cells * 100 / total,
    ));
    for y in 0..h {
        for x in 0..w {
            let cell = fields[x + y * w];
            let (r, g, b) = sky_color(cell.cloud);
            out.push_str(&format!("\x1b[48;2;{r};{g};{b}m  "));
        }
        out.push_str("\x1b[0m\n");
    }

    print!("{out}");
    let _ = std::io::stdout().flush();
}

/// Maps cloud density to a sky ramp: deep blue clear sky brightening to white as
/// clouds thicken. Values above 1 just saturate to white.
fn sky_color(cloud: f32) -> (u8, u8, u8) {
    let t = cloud.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| (a + (b - a) * t) as u8;
    (lerp(28.0, 244.0), lerp(58.0, 246.0), lerp(110.0, 252.0))
}
