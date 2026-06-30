/// Forces a quantizer to run at unit resolution.
pub mod kernel;
/// Matches a cell's resolution to its neighbourhood.
pub mod neighbors;

use crate::{
    allocator::{Address, Allocator},
    grid::{Cell, CellData, GridSampler},
    topology::Topology,
};
use std::ops::Range;

/// Writes new cells into the destination arena while a [`Quantizer`] runs.
///
/// A quantizer is handed an emitter and must return exactly one [`Address`] - the
/// root of the (sub)tree it produced for its cell - built via these methods.
pub struct CellEmitter<'a, T: CellData, Topo: Topology> {
    pub(crate) allocator: &'a mut Allocator<Cell<T, Topo>>,
}

impl<'a, T: CellData, Topo: Topology> CellEmitter<'a, T, Topo> {
    /// Emits a leaf holding `data` and returns its address.
    pub fn emit_leaf(&mut self, data: T) -> Address {
        self.allocator.push(Cell::Leaf { data })
    }

    /// Emits a branch with the given child addresses and returns its address.
    pub fn emit_branch(&mut self, children: Topo::Children<Address>) -> Address {
        self.allocator.push(Cell::Branch { children })
    }

    /// Emits a branch, but collapses it back into a single leaf when all
    /// children are [homogeneous](CellData::are_homogeneous) leaves.
    ///
    /// `merger` combines the children's data into the merged leaf's value and is
    /// only called when the merge actually happens. Maintains the
    /// "no branch that could be a leaf" invariant.
    pub fn emit_branch_possibly_merged(
        &mut self,
        children: Topo::Children<Address>,
        merger: impl Fn(&[&T]) -> T,
    ) -> Address {
        let cells = Topo::children_as_slice(&children)
            .iter()
            .map(|&address| {
                *self
                    .allocator
                    .read(address)
                    .unwrap_or_else(|| panic!("Could not read cell at address: {}", address))
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
                let merged_data = merger(&cells_data);
                for _ in 0..Topo::CHILDREN {
                    self.allocator.pop();
                }
                return self.allocator.push(Cell::Leaf { data: merged_data });
            }
        }
        self.allocator.push(Cell::Branch { children })
    }

    /// Convenience wrapper around [`emit_branch`](CellEmitter::emit_branch) that
    /// builds the children via `f` first.
    pub fn emit_group(&mut self, f: impl FnOnce(&mut Self) -> Topo::Children<Address>) -> Address {
        let children = f(self);
        self.emit_branch(children)
    }

    /// Convenience wrapper around
    /// [`emit_branch_possibly_merged`](CellEmitter::emit_branch_possibly_merged)
    /// that builds the children via `f` first.
    pub fn emit_group_possibly_merged(
        &mut self,
        f: impl FnOnce(&mut Self) -> Topo::Children<Address>,
        merger: impl Fn(&[&T]) -> T,
    ) -> Address {
        let children = f(self);
        self.emit_branch_possibly_merged(children, merger)
    }
}

/// The cell currently being processed by a [`Quantizer`], plus everything
/// needed to read neighbours and emit the replacement cell.
pub struct CellContext<'a, T: CellData, Topo: Topology> {
    /// Half-open region of grid coordinates this cell covers.
    pub region: Range<Topo::Coord>,
    /// Edge length of this cell in cells (`chunk_size >> depth`).
    pub cell_size: usize,
    /// Depth of this cell below its chunk root (`0` = whole chunk).
    pub depth: usize,
    /// This cell's current data value.
    pub cell_data: &'a T,
    /// Sampler over the *source* grid, for reading neighbours.
    pub sampler: &'a mut GridSampler<'a, T, Topo>,
    /// Emitter into the *destination* grid, for writing the replacement cell.
    pub emitter: &'a mut CellEmitter<'a, T, Topo>,
}

impl<'a, T: CellData, Topo: Topology> CellContext<'a, T, Topo> {
    /// The edge length of a cell at this depth (`chunk_size >> depth`),
    /// equivalently [`cell_size`](CellContext::cell_size).
    pub fn depth_size(&self) -> usize {
        self.sampler.grid_chunk_size() >> self.depth
    }

    /// Number of unit cells this cell covers (`cell_size.pow(DIMENSIONS)`).
    pub fn area(&self) -> usize {
        Topo::cell_volume(self.cell_size)
    }

    /// Compute the descriptor of the `index`-th child subregion.
    pub fn subregion_at(&self, index: usize) -> ContextSubRegion<Topo> {
        let region_coord = Topo::child_offset(index);
        let coords_from = Topo::from_axes(|axis| {
            let start = Topo::axis(self.region.start, axis);
            let half = (Topo::axis(self.region.end, axis) - start) / 2;
            start + half * Topo::axis(region_coord, axis)
        });
        let coords_to = Topo::from_axes(|axis| {
            let start = Topo::axis(self.region.start, axis);
            let half = (Topo::axis(self.region.end, axis) - start) / 2;
            start + half * (Topo::axis(region_coord, axis) + 1)
        });
        ContextSubRegion {
            index,
            region_coord,
            region: coords_from..coords_to,
            cell_size: self.cell_size / 2,
            depth: self.depth + 1,
        }
    }

    /// All child subregions of this cell, in child-index order.
    pub fn subdivide_region(&self) -> Vec<ContextSubRegion<Topo>> {
        (0..Topo::CHILDREN)
            .map(|index| self.subregion_at(index))
            .collect()
    }

    /// Subdivide this cell, invoking `f` with a child context per slot and
    /// collecting the emitted addresses into a topology-shaped children array.
    pub fn subdivide(
        &mut self,
        mut f: impl FnMut(CellContext<T, Topo>) -> Address,
    ) -> Topo::Children<Address> {
        Topo::children_from_fn(|index| {
            let subregion = self.subregion_at(index);
            let context = unsafe { self.subregion(&subregion) };
            f(context)
        })
    }

    /// # Safety
    /// This function is unsafe because it creates a new context with a mutable
    /// reference to the same emitter and sampler. This can lead to undefined
    /// behavior if not cerefully used, as the emitter and sampler may be
    /// modified while it is being used in another context.
    /// Please ensure proper emitter and sampler usage.
    pub unsafe fn subregion(&mut self, subregion: &ContextSubRegion<Topo>) -> Self {
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

/// Describes one child slot when subdividing a [`CellContext`].
#[derive(Debug, Clone)]
pub struct ContextSubRegion<Topo: Topology> {
    /// Child index within the parent (`0..CHILDREN`).
    pub index: usize,
    /// The child's `0`/`1`-per-axis offset within the parent.
    pub region_coord: Topo::Coord,
    /// Half-open region of grid coordinates the child covers.
    pub region: Range<Topo::Coord>,
    /// Edge length of the child cell in cells (half the parent's).
    pub cell_size: usize,
    /// Depth of the child (one deeper than the parent).
    pub depth: usize,
}

/// A transformation rule applied to a grid, cell by cell, by a
/// [`Transformer`](crate::pipeline::Transformer).
///
/// For each cell of the source grid, [`quantize`](Quantizer::quantize) decides
/// what to emit into the destination grid: a leaf, or a branch built by
/// [subdividing](CellContext::subdivide) and recursing. Implementors may read
/// neighbouring cells through [`CellContext::sampler`].
pub trait Quantizer {
    /// The cell value type this quantizer operates on.
    type CellData: CellData;
    /// The grid topology this quantizer operates on.
    type Topology: Topology;

    /// Produces the replacement (sub)tree for one source cell, returning the
    /// address of its root in the destination arena.
    fn quantize(&self, context: CellContext<Self::CellData, Self::Topology>) -> Address;

    /// Combines the data of `CHILDREN` homogeneous leaves into the single value
    /// they collapse to. `cells` is non-empty; the default keeps the first.
    fn merge(&self, cells: &[&Self::CellData]) -> Self::CellData {
        *cells[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        changes::Changes,
        grid::{Grid3d, GridConfig},
        pipeline::Transformer,
        topology::Topology3d,
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
        type Topology = Topology3d;

        fn quantize(&self, context: CellContext<Self::CellData, Self::Topology>) -> Address {
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

        let mut grid = Grid3d::new(
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
