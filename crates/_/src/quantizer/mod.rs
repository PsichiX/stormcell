pub mod kernel;
pub mod neighbors;

use crate::{
    allocator::{Address, Allocator},
    grid::{Cell, CellData, GridSampler},
};
use std::ops::Range;

pub const COORD_INDICES: [usize; 3] = [0, 1, 2];
pub const CELL_CHILDREN_INDICES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
pub const CELL_CHILDREN_COORDS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

pub struct CellEmitter<'a, T: CellData> {
    pub(crate) allocator: &'a mut Allocator<Cell<T>>,
}

impl<'a, T: CellData> CellEmitter<'a, T> {
    pub fn emit_leaf(&mut self, data: T) -> Address {
        self.allocator.push(Cell::Leaf { data })
    }

    pub fn emit_branch(&mut self, children: [Address; 8]) -> Address {
        self.allocator.push(Cell::Branch { children })
    }

    pub fn emit_branch_possibly_merged(
        &mut self,
        children: [Address; 8],
        merger: impl Fn([&T; 8]) -> T,
    ) -> Address {
        let cells = children.map(|address| {
            self.allocator
                .read(address)
                .unwrap_or_else(|| panic!("Could not read cell at address: {}", address))
        });
        if cells.iter().all(|cell| cell.is_leaf()) {
            let cells_data = cells.map(|cell| cell.data().expect("Could not read cell data"));
            if cells_data
                .iter()
                .skip(1)
                .all(|data| cells_data[0].are_homogeneous(data))
            {
                let merged_data = merger(cells_data);
                for _ in 0..8 {
                    self.allocator.pop();
                }
                return self.allocator.push(Cell::Leaf { data: merged_data });
            }
        }
        self.allocator.push(Cell::Branch { children })
    }

    pub fn emit_group(&mut self, f: impl FnOnce(&mut Self) -> [Address; 8]) -> Address {
        let children = f(self);
        self.emit_branch(children)
    }

    pub fn emit_group_possibly_merged(
        &mut self,
        f: impl FnOnce(&mut Self) -> [Address; 8],
        merger: impl Fn([&T; 8]) -> T,
    ) -> Address {
        let children = f(self);
        self.emit_branch_possibly_merged(children, merger)
    }
}

pub struct CellContext<'a, T: CellData> {
    pub region: Range<[usize; 3]>,
    pub cell_size: usize,
    pub depth: usize,
    pub cell_data: &'a T,
    pub sampler: &'a mut GridSampler<'a, T>,
    pub emitter: &'a mut CellEmitter<'a, T>,
}

impl<'a, T: CellData> CellContext<'a, T> {
    pub fn depth_size(&self) -> usize {
        1 >> self.depth
    }

    pub fn area(&self) -> usize {
        self.cell_size * self.cell_size * self.cell_size
    }

    pub fn subdivide_region(&self) -> [ContextSubRegion; 8] {
        let region_half_size =
            COORD_INDICES.map(|index| (self.region.end[index] - self.region.start[index]) / 2);
        let cell_size = self.cell_size / 2;
        let depth = self.depth + 1;
        CELL_CHILDREN_INDICES.map(move |index| {
            let child_coords = [index & 1, (index >> 1) & 1, (index >> 2) & 1];
            let coords_from = COORD_INDICES.map(|index| {
                self.region.start[index] + region_half_size[index] * child_coords[index]
            });
            let coords_to = COORD_INDICES.map(|index| {
                self.region.start[index] + region_half_size[index] * (child_coords[index] + 1)
            });
            ContextSubRegion {
                index,
                region_coord: child_coords,
                region: coords_from..coords_to,
                cell_size,
                depth,
            }
        })
    }

    /// # Safety
    /// This function is unsafe because it creates a new context with a mutable
    /// reference to the same emitter and sampler. This can lead to undefined
    /// behavior if not cerefully used, as the emitter and sampler may be
    /// modified while it is being used in another context.
    /// Please ensure proper emitter and sampler usage.
    pub unsafe fn subregion(&mut self, subregion: &ContextSubRegion) -> Self {
        CellContext {
            region: subregion.region.clone(),
            cell_size: subregion.cell_size,
            depth: subregion.depth,
            cell_data: self.cell_data,
            sampler: unsafe { &mut *(self.sampler as *mut _) },
            emitter: unsafe { &mut *(self.emitter as *mut _) },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextSubRegion {
    pub index: usize,
    pub region_coord: [usize; 3],
    pub region: Range<[usize; 3]>,
    pub cell_size: usize,
    pub depth: usize,
}

pub trait Quantizer {
    type CellData: CellData;

    fn quantize(&self, context: CellContext<Self::CellData>) -> Address;

    fn merge(&self, cells: [&Self::CellData; 8]) -> Self::CellData {
        *cells[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        changes::Changes,
        grid::{Grid, GridConfig},
        pipeline::Transformer,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Stability(pub usize);

    impl CellData for Stability {
        fn are_homogeneous(&self, other: &Self) -> bool {
            self == other
        }

        fn scale(&self, multiplier: usize) -> Self {
            Stability(self.0 * multiplier)
        }
    }

    struct Decay {
        pub rate: usize,
    }

    impl Quantizer for Decay {
        type CellData = Stability;

        fn quantize(&self, context: CellContext<Self::CellData>) -> Address {
            let value = context.cell_data.0.saturating_sub(self.rate);

            context.emitter.emit_leaf(Stability(value))
        }
    }

    fn random(coord: [usize; 3], limit: usize) -> usize {
        let mut x = coord[0] as u64;
        let mut y = coord[1] as u64;
        let mut z = coord[2] as u64;
        x = x.wrapping_mul(73856093);
        y = y.wrapping_mul(19349663);
        z = z.wrapping_mul(83492791);
        let hash = x ^ y ^ z;
        (hash ^ (hash >> 16)) as usize % limit
    }

    #[test]
    fn test_quantizer() {
        let quantizer = Decay { rate: 1 };

        let mut grid = Grid::new(
            GridConfig {
                chunk_max_depth: 2,
                sampler_cache_limit: Some(64),
                ..Default::default()
            },
            Stability(0),
        );
        let mut changes = Changes::default();
        for x in 0..grid.size()[0] {
            for y in 0..grid.size()[1] {
                for z in 0..grid.size()[2] {
                    changes.set([x, y, z], Stability(random([x, y, z], 5)));
                }
            }
        }
        grid = changes.apply(&grid);
        let flat = grid.flatten(Stability(0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<usize>(), 139);

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Stability(0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<usize>(), 83);

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Stability(0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<usize>(), 44);

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Stability(0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<usize>(), 16);

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Stability(0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<usize>(), 0);
    }
}
