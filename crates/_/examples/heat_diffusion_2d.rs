//! 2D heat diffusion on an adaptive-resolution grid.

use std::io::Write;
use stormcell::{
    allocator::Address,
    changes::Changes,
    grid::{CellData, Grid, Grid2d, GridConfig},
    pipeline::Transformer,
    quantizer::{CellContext, Quantizer, kernel::KernelQuantizer},
    topology::Topology2d,
};

fn main() {
    // clear the screen once; every frame redraws from the top afterwards
    print!("\x1b[2J");

    let mut grid = Grid2d::<Temperature>::new(
        GridConfig {
            chunks_num: CHUNKS,
            chunk_max_depth: CHUNK_DEPTH,
            sampler_cache_limit: Some(512),
            ..Default::default()
        },
        Temperature(0.0),
    );

    let mut tick: u64 = 0;
    let mut source_index = 0;
    loop {
        // Every PULSE_PERIOD steps, stamp a fresh hot spot from outside the
        // simulation, alternating between the two sources.
        if tick.is_multiple_of(PULSE_PERIOD) {
            let source = SOURCES[source_index];
            source_index = (source_index + 1) % SOURCES.len();
            let mut pulse = Changes::new(8);
            pulse.set(source, Temperature(PULSE_TEMP));
            grid = pulse.apply(&grid);
        }

        render(&grid);
        std::thread::sleep(std::time::Duration::from_millis(DELAY_MS));

        // One simulation step: the Transformer runs our diffusion quantizer
        // over the whole grid to build the next simulation step grid.
        let kernel = KernelQuantizer {
            quantizer: Diffuse {
                rate: RATE,
                retention: RETENTION,
            },
        };
        grid = Transformer::new(&kernel, &grid).execute();
        tick = tick.wrapping_add(1);
    }
}

// 3x3 chunks ...
const CHUNKS: [usize; 2] = [3, 3];
// ... of 8x8 cells each => 24x24 grid
const CHUNK_DEPTH: usize = 3;
// two heat sources at opposite ends, pulsed alternately
const SOURCES: [[usize; 2]; 2] = [[8, 12], [16, 12]];
// temperature stamped at a source when it fires
const PULSE_TEMP: f32 = 1024.0;
// fire a pulse (alternating source) every this many steps
const PULSE_PERIOD: u64 = 15;
// diffusion rate per step
const RATE: f32 = 0.2;
// fraction of its heat each cell keeps per step (cooling, so pulses fade)
const RETENTION: f32 = 0.99;
// cells within this temperature merge into one cell
const MERGE_EPS: f32 = 6.0;
const DELAY_MS: u64 = 50;

/// Temperature stored per cell, in the 0..=255 range used for colouring.
#[derive(Clone, Copy)]
struct Temperature(f32);

impl CellData for Temperature {
    fn are_homogeneous(&self, other: &Self) -> bool {
        (self.0 - other.0).abs() < MERGE_EPS
    }

    fn scale(&self, multiplier: usize) -> Self {
        Temperature(self.0 * multiplier as f32)
    }
}

// A `Quantizer` is how stormcell advances state, telling given this source cell,
// what should the matching cell in the next simulation step be.
// We are effectively reconstructing entire grid snapshots each step, but the
// adaptive tree means we only store and compute the detail that is actually present.
// This rule is pure physics: each cell relaxes towards its neighbours' average
// (`rate`) and keeps only `retention` of its heat each step, so pulses fade out.
// The heat sources are applied from outside in `main`, not by this quantizer.
struct Diffuse {
    rate: f32,
    retention: f32,
}

impl Quantizer for Diffuse {
    type CellData = Temperature;
    type Topology = Topology2d;

    fn quantize(&self, ctx: CellContext<Temperature, Topology2d>) -> Address {
        // KernelQuantizer drives this down to unit cells, so region.start is the
        // cell's coordinate.
        let [x, y] = ctx.region.start;
        let [w, h] = ctx.sampler.grid_size();
        let left = ctx.sampler.sample([x.saturating_sub(1), y]).unwrap().data.0;
        let right = ctx.sampler.sample([(x + 1).min(w - 1), y]).unwrap().data.0;
        let up = ctx.sampler.sample([x, y.saturating_sub(1)]).unwrap().data.0;
        let down = ctx.sampler.sample([x, (y + 1).min(h - 1)]).unwrap().data.0;
        let v = ctx.cell_data.0;
        // Diffuse towards the neighbour average, then cool a little.
        let next = (v + self.rate * (left + right + up + down - 4.0 * v)) * self.retention;
        ctx.emitter.emit_leaf(Temperature(next))
    }
}

fn render(grid: &Grid<Temperature, Topology2d>) {
    let flat = grid.flatten(Temperature(0.0));
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
            let (r, g, b) = heat_color(fields[x + y * w].0);
            out.push_str(&format!("\x1b[48;2;{r};{g};{b}m  "));
        }
        out.push_str("\x1b[0m\n");
    }

    print!("{out}");
    let _ = std::io::stdout().flush();
}

/// Maps a temperature in 0..=255 to a black -> red -> yellow -> white ramp.
/// `as u8` float casts saturate, so out-of-range channels clamp automatically.
fn heat_color(t: f32) -> (u8, u8, u8) {
    let v = t.clamp(0.0, 255.0);
    (
        (v * 3.0) as u8,
        (v * 3.0 - 255.0) as u8,
        (v * 3.0 - 510.0) as u8,
    )
}
