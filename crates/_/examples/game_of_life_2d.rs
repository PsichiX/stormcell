//! Conway's Game of Life on an adaptive-resolution grid.

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

    let mut grid = Grid2d::<Cell>::new(
        GridConfig {
            chunks_num: CHUNKS,
            chunk_max_depth: CHUNK_DEPTH,
            sampler_cache_limit: Some(512),
            ..Default::default()
        },
        Cell(false),
    );

    // Stamp the initial pattern from outside the simulation.
    let mut seed = Changes::new(8);
    for &[x, y] in GOSPER_GLIDER_GUN {
        seed.set([x + GUN_OFFSET[0], y + GUN_OFFSET[1]], Cell(true));
    }
    grid = seed.apply(&grid);

    let mut tick: u64 = 0;
    loop {
        render(&grid);
        std::thread::sleep(std::time::Duration::from_millis(DELAY_MS));

        // One generation: the Transformer runs our Conway rule over the whole
        // grid to build the next generation's grid.
        let kernel = KernelQuantizer { quantizer: Conway };
        grid = Transformer::new(&kernel, &grid).execute();
        tick = tick.wrapping_add(1);
    }
}

// 6x3 chunks ...
const CHUNKS: [usize; 2] = [6, 3];
// ... of 8x8 cells each => 48x24 grid
const CHUNK_DEPTH: usize = 3;
// where to place the gun's top-left corner inside the grid
const GUN_OFFSET: [usize; 2] = [1, 1];
const DELAY_MS: u64 = 50;

/// One Game of Life cell: `true` when alive.
#[derive(Clone, Copy)]
struct Cell(bool);

impl CellData for Cell {
    fn are_homogeneous(&self, other: &Self) -> bool {
        // Uniform regions (all alive or all dead) collapse into one coarse cell.
        self.0 == other.0
    }

    fn scale(&self, _multiplier: usize) -> Self {
        // A coarse cell's state is shared by every unit cell it covers, so
        // area-scaling is a no-op for this boolean field.
        *self
    }
}

// Conway's rule: a live cell survives with 2 or 3 live neighbours; a dead cell
// is born with exactly 3. Out-of-bounds neighbours count as dead.
struct Conway;

impl Quantizer for Conway {
    type CellData = Cell;
    type Topology = Topology2d;

    fn quantize(&self, ctx: CellContext<Cell, Topology2d>) -> Address {
        let [x, y] = ctx.region.start;
        let [w, h] = ctx.sampler.grid_size();

        let mut live = 0;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                // Treat the borders as permanently dead (no wrap-around).
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                if ctx
                    .sampler
                    .sample([nx as usize, ny as usize])
                    .unwrap()
                    .data
                    .0
                {
                    live += 1;
                }
            }
        }

        let alive = ctx.cell_data.0;
        let next = matches!((alive, live), (true, 2) | (true, 3) | (false, 3));
        ctx.emitter.emit_leaf(Cell(next))
    }
}

fn render(grid: &Grid<Cell, Topology2d>) {
    let [w, h] = grid.size();
    let total = w * h;

    // Walk the tree once, recording each unit cell's alive state and the edge
    // length of the leaf it belongs to. Coarse leaves (large `size`) mark the
    // collapsed, uniform regions the adaptive grid saved us from storing.
    let mut alive = vec![false; total];
    let mut leaf_size = vec![1usize; total];
    let mut cells = 0;
    for (range, data) in grid.iter() {
        cells += 1;
        let size = range.end[0] - range.start[0];
        for cy in range.start[1]..range.end[1] {
            for cx in range.start[0]..range.end[0] {
                let i = cx + cy * w;
                alive[i] = data.0;
                leaf_size[i] = size;
            }
        }
    }

    let mut out = String::with_capacity(total * 24);
    // cursor home (overwrite previous frame)
    out.push_str("\x1b[H");
    out.push_str(&format!(
        "Cells: {cells:>4} / {total} ({:>2}%)\n",
        cells * 100 / total,
    ));
    for y in 0..h {
        for x in 0..w {
            let i = x + y * w;
            let (r, g, b) = cell_color(alive[i], leaf_size[i]);
            out.push_str(&format!("\x1b[48;2;{r};{g};{b}m  "));
        }
        out.push_str("\x1b[0m\n");
    }

    print!("{out}");
    let _ = std::io::stdout().flush();
}

/// Live cells are bright green. Dead cells are tinted by the size of the leaf
/// they collapsed into: fine (size 1) stays near-black, coarser leaves get
/// progressively bluer, so the adaptive grid's coarse regions stay visible.
fn cell_color(alive: bool, leaf_size: usize) -> (u8, u8, u8) {
    if alive {
        return (80, 240, 120);
    }
    let shade = (leaf_size.trailing_zeros() * 14).min(70) as u8;
    (10, 14, 20 + shade)
}

/// Gosper glider gun, as `(x, y)` offsets from the pattern's top-left corner.
/// It periodically emits gliders that stream off towards the bottom-right.
const GOSPER_GLIDER_GUN: &[[usize; 2]] = &[
    // left square block
    [0, 4],
    [0, 5],
    [1, 4],
    [1, 5],
    // left ship
    [10, 4],
    [10, 5],
    [10, 6],
    [11, 3],
    [11, 7],
    [12, 2],
    [12, 8],
    [13, 2],
    [13, 8],
    [14, 5],
    [15, 3],
    [15, 7],
    [16, 4],
    [16, 5],
    [16, 6],
    [17, 5],
    // right ship
    [20, 2],
    [20, 3],
    [20, 4],
    [21, 2],
    [21, 3],
    [21, 4],
    [22, 1],
    [22, 5],
    [24, 0],
    [24, 1],
    [24, 5],
    [24, 6],
    // right square block
    [34, 2],
    [34, 3],
    [35, 2],
    [35, 3],
];
