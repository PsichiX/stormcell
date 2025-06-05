use crate::{
    allocator::{Address, Allocator},
    grid::{Cell, CellData, Grid, GridSampler},
    quantizer::{CELL_CHILDREN_INDICES, COORD_INDICES, CellContext, CellEmitter, Quantizer},
};
use std::ops::Range;

pub struct Transformer<'a, T: CellData> {
    pub cached_samples_evicting_depth: usize,
    quantizer: &'a dyn Quantizer<CellData = T>,
    grid: &'a Grid<T>,
}

impl<'a, T: CellData> Transformer<'a, T> {
    pub fn new(quantizer: &'a dyn Quantizer<CellData = T>, grid: &'a Grid<T>) -> Self {
        Self {
            cached_samples_evicting_depth: 4,
            quantizer,
            grid,
        }
    }

    pub fn with_cached_samples_evicting_depth(mut self, value: usize) -> Self {
        self.cached_samples_evicting_depth = value;
        self
    }

    pub fn execute(self) -> Grid<T> {
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
                let chunk_coords = [
                    index % grid.config.chunks_num[0],
                    (index / grid.config.chunks_num[0]) % grid.config.chunks_num[1],
                    index / (grid.config.chunks_num[0] * grid.config.chunks_num[1]),
                ];
                let coords_from = chunk_coords.map(|coord| coord * grid.chunk_size);
                let coords_to = coords_from.map(|coord| coord + grid.chunk_size);
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
        region: Range<[usize; 3]>,
        cell_size: usize,
        depth: usize,
        cached_samples_evicting_depth: usize,
        cell: &Cell<T>,
        quantizer: &dyn Quantizer<CellData = T>,
        sampler: &mut GridSampler<T>,
        allocator: &mut Allocator<Cell<T>>,
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
                let region_half_size =
                    COORD_INDICES.map(|index| (region.end[index] - region.start[index]) / 2);
                let children = CELL_CHILDREN_INDICES.map(|index| {
                    let address = children[index];
                    let child_coords = [index & 1, (index >> 1) & 1, (index >> 2) & 1];
                    let coords_from = COORD_INDICES.map(|index| {
                        region.start[index] + region_half_size[index] * child_coords[index]
                    });
                    let coords_to = COORD_INDICES.map(|index| {
                        region.start[index] + region_half_size[index] * (child_coords[index] + 1)
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
                let cells = children.map(|address| {
                    allocator
                        .read(address)
                        .unwrap_or_else(|| panic!("Could not read cell at address: {}", address))
                });
                if cells.iter().all(|cell| cell.is_leaf()) {
                    let cells_data =
                        cells.map(|cell| cell.data().expect("Could not read cell data"));
                    if cells_data
                        .iter()
                        .skip(1)
                        .all(|data| cells_data[0].are_homogeneous(data))
                    {
                        let merged_data = quantizer.merge(cells_data);
                        for _ in 0..8 {
                            allocator.pop();
                        }
                        return allocator.push(Cell::Leaf { data: merged_data });
                    }
                }
                if depth % cached_samples_evicting_depth == 0 {
                    sampler.evict_least_accessed_samples();
                }
                allocator.push(Cell::Branch { children })
            }
        }
    }
}
