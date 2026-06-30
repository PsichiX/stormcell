use crate::{
    allocator::{Address, Allocator},
    grid::{Cell, CellData, Grid, GridSampler},
    quantizer::{CellContext, CellEmitter, Quantizer},
    topology::Topology,
};
use std::ops::Range;

/// Runs a [`Quantizer`] over an entire grid to produce a new grid.
///
/// The transformer walks every chunk of the source `grid`, invoking the
/// quantizer per cell and assembling the results into a freshly-allocated
/// destination grid (same config and dimensions). The source grid is read-only;
/// nothing is mutated in place.
pub struct Transformer<'a, T: CellData, Topo: Topology> {
    /// Depth interval at which the source sampler's leaf cache is trimmed during
    /// the walk, bounding peak cache size on deep trees.
    pub cached_samples_evicting_depth: usize,
    quantizer: &'a dyn Quantizer<CellData = T, Topology = Topo>,
    grid: &'a Grid<T, Topo>,
}

impl<'a, T: CellData, Topo: Topology> Transformer<'a, T, Topo> {
    /// Creates a transformer pairing `quantizer` with the source `grid`.
    pub fn new(
        quantizer: &'a dyn Quantizer<CellData = T, Topology = Topo>,
        grid: &'a Grid<T, Topo>,
    ) -> Self {
        Self {
            cached_samples_evicting_depth: 4,
            quantizer,
            grid,
        }
    }

    /// Overrides [`cached_samples_evicting_depth`](Transformer::cached_samples_evicting_depth).
    pub fn with_cached_samples_evicting_depth(mut self, value: usize) -> Self {
        self.cached_samples_evicting_depth = value;
        self
    }

    /// Runs the quantizer over the whole grid and returns the new grid.
    pub fn execute(self) -> Grid<T, Topo> {
        let Self {
            cached_samples_evicting_depth,
            quantizer,
            grid,
        } = self;
        let mut allocator = Allocator::new(grid.config.page_capacity, grid.config.pages_capacity);
        let mut sampler = grid.sampler();
        let chunks = grid
            .chunks
            .iter()
            .copied()
            .enumerate()
            .map(|(index, address)| {
                let cell = grid
                    .allocator
                    .read(address)
                    .unwrap_or_else(|| panic!("Could not read cell at address: {}", address));
                let chunk_coords = Topo::from_linear_index(index, grid.config.chunks_num);
                let coords_from = Topo::map(chunk_coords, |coord| coord * grid.chunk_size);
                let coords_to = Topo::map(coords_from, |coord| coord + grid.chunk_size);
                Self::process_cell(
                    coords_from..coords_to,
                    grid.chunk_size,
                    0,
                    cached_samples_evicting_depth,
                    cell,
                    quantizer,
                    &mut sampler,
                    &mut allocator,
                )
            })
            .collect::<Vec<_>>();
        Grid {
            config: grid.config.clone(),
            size: grid.size,
            chunk_size: grid.chunk_size,
            chunks,
            allocator,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_cell(
        region: Range<Topo::Coord>,
        cell_size: usize,
        depth: usize,
        cached_samples_evicting_depth: usize,
        cell: &Cell<T, Topo>,
        quantizer: &dyn Quantizer<CellData = T, Topology = Topo>,
        sampler: &mut GridSampler<T, Topo>,
        allocator: &mut Allocator<Cell<T, Topo>>,
    ) -> Address {
        match cell {
            Cell::Leaf { data } => {
                let mut emitter = CellEmitter { allocator };
                quantizer.quantize(CellContext {
                    region,
                    cell_size,
                    depth,
                    cell_data: data,
                    sampler: unsafe { std::mem::transmute::<&mut _, &mut _>(sampler) },
                    emitter: &mut emitter,
                })
            }
            Cell::Branch { children } => {
                let region_half_size = Topo::from_axes(|axis| {
                    (Topo::axis(region.end, axis) - Topo::axis(region.start, axis)) / 2
                });
                let children = Topo::children_from_fn(|index| {
                    let address = Topo::children_as_slice(children)[index];
                    let child_coords = Topo::child_offset(index);
                    let coords_from = Topo::from_axes(|axis| {
                        Topo::axis(region.start, axis)
                            + Topo::axis(region_half_size, axis) * Topo::axis(child_coords, axis)
                    });
                    let coords_to = Topo::from_axes(|axis| {
                        Topo::axis(region.start, axis)
                            + Topo::axis(region_half_size, axis)
                                * (Topo::axis(child_coords, axis) + 1)
                    });
                    Self::process_cell(
                        coords_from..coords_to,
                        cell_size / 2,
                        depth + 1,
                        cached_samples_evicting_depth,
                        sampler.grid.allocator.read(address).unwrap_or_else(|| {
                            panic!("Could not read cell at address: {}", address)
                        }),
                        quantizer,
                        sampler,
                        allocator,
                    )
                });
                let cells = Topo::children_as_slice(&children)
                    .iter()
                    .map(|&address| {
                        *allocator.read(address).unwrap_or_else(|| {
                            panic!("Could not read cell at address: {}", address)
                        })
                    })
                    .collect::<Vec<_>>();
                if cells.iter().all(|cell| cell.is_leaf()) {
                    let cells_data = cells
                        .iter()
                        .map(|cell| cell.data().expect("Could not read cell data"))
                        .collect::<Vec<_>>();
                    if cells_data
                        .iter()
                        .skip(1)
                        .all(|data| cells_data[0].are_homogeneous(data))
                    {
                        let merged_data = quantizer.merge(&cells_data);
                        for _ in 0..Topo::CHILDREN {
                            allocator.pop();
                        }
                        return allocator.push(Cell::Leaf { data: merged_data });
                    }
                }
                if depth.is_multiple_of(cached_samples_evicting_depth) {
                    sampler.evict_least_accessed_samples();
                }
                allocator.push(Cell::Branch { children })
            }
        }
    }
}
