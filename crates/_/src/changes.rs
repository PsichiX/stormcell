use crate::{
    allocator::Address,
    grid::{CellData, Grid},
    pipeline::Transformer,
    quantizer::{CellContext, Quantizer},
    topology::{Topology, Topology3d},
};
use std::{collections::BTreeMap, ops::Range};

/// A sparse staging buffer of pending cell edits.
///
/// Edits are accumulated by coordinate and only materialised when
/// [`apply`](Changes::apply)ed to a grid. `Changes` is itself a
/// [`Quantizer`]: applying it rebuilds the grid, subdividing only where edits
/// introduce heterogeneity and leaving untouched regions coarse. Edits are
/// bucketed into cubic chunks of `chunk_size` (independent of the target grid's
/// own chunking) to keep the backing map compact.
pub struct Changes<T: CellData, Topo: Topology = Topology3d> {
    chunk_size: usize,
    chunks: BTreeMap<Topo::Coord, Vec<Option<T>>>,
}

impl<T: CellData, Topo: Topology> Default for Changes<T, Topo> {
    fn default() -> Self {
        Self::new(32)
    }
}

impl<T: CellData, Topo: Topology> Changes<T, Topo> {
    /// Creates an empty change set bucketing edits into cubes of `chunk_size`.
    #[inline]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            chunks: Default::default(),
        }
    }

    /// The edit-bucket edge length this change set uses.
    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Discards all pending edits.
    #[inline]
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    fn local_index(&self, local_coord: Topo::Coord) -> usize {
        Topo::linear_index(local_coord, Topo::splat(self.chunk_size))
    }

    /// Stages `value` at a single `coord`, overwriting any prior edit there.
    pub fn set(&mut self, coord: Topo::Coord, value: T) {
        let chunk_coord = Topo::map(coord, |c| c / self.chunk_size);
        let local_coord = Topo::map(coord, |c| c % self.chunk_size);
        let index = self.local_index(local_coord);
        let capacity = Topo::cell_volume(self.chunk_size);
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| vec![None; capacity])
            .as_mut_slice();
        chunk[index] = Some(value);
    }

    /// Stages `value` at every coordinate in the half-open `region`.
    pub fn set_region(&mut self, region: Range<Topo::Coord>, value: T) {
        let mut coords = Vec::new();
        Topo::for_each_coord(&region, |coord| coords.push(coord));
        for coord in coords {
            self.set(coord, value);
        }
    }

    /// Stages many `(coord, value)` edits at once.
    pub fn extend(&mut self, values: impl IntoIterator<Item = (Topo::Coord, T)>) {
        for (coord, value) in values {
            self.set(coord, value);
        }
    }

    /// Returns the staged value at `coord`, or `None` if nothing is staged there.
    pub fn sample(&self, coord: Topo::Coord) -> Option<&T> {
        let chunk_coord = Topo::map(coord, |c| c / self.chunk_size);
        let local_coord = Topo::map(coord, |c| c % self.chunk_size);
        let index = self.local_index(local_coord);
        self.chunks.get(&chunk_coord)?.get(index)?.as_ref()
    }

    /// Iterates over every coordinate in `region` paired with its staged value
    /// (`None` where nothing is staged).
    pub fn sample_region(
        &self,
        region: Range<Topo::Coord>,
    ) -> impl Iterator<Item = (Topo::Coord, Option<&T>)> {
        let mut coords = Vec::new();
        Topo::for_each_coord(&region, |coord| coords.push(coord));
        coords
            .into_iter()
            .map(move |coord| (coord, self.sample(coord)))
    }

    /// If every coordinate in `region` resolves to
    /// [homogeneous](CellData::are_homogeneous) data - using `default_value`
    /// wherever no edit is staged - returns that common value, else `None`.
    pub fn is_region_homogeneous(
        &self,
        region: Range<Topo::Coord>,
        default_value: &T,
    ) -> Option<T> {
        let mut value: Option<&T> = None;
        let mut homogeneous = true;
        Topo::for_each_coord(&region, |coord| {
            if !homogeneous {
                return;
            }
            let sample = self.sample(coord).unwrap_or(default_value);
            if let Some(value) = value
                && !sample.are_homogeneous(value)
            {
                homogeneous = false;
                return;
            }
            value = Some(sample);
        });
        if homogeneous { value.copied() } else { None }
    }

    /// Applies all staged edits to `grid`, returning the resulting new grid.
    /// The input grid is left unchanged.
    pub fn apply(self, grid: &Grid<T, Topo>) -> Grid<T, Topo> {
        Transformer::new(&self, grid).execute()
    }
}

impl<T: CellData, Topo: Topology> Quantizer for Changes<T, Topo> {
    type CellData = T;
    type Topology = Topo;

    fn quantize(&self, mut context: CellContext<Self::CellData, Self::Topology>) -> Address {
        if let Some(value) = self.is_region_homogeneous(context.region.clone(), context.cell_data) {
            context.emitter.emit_leaf(value)
        } else {
            let children = context.subdivide(|ctx| self.quantize(ctx));
            context.emitter.emit_branch(children)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid3d, GridConfig, GridPreviewCell};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Data(pub i32);

    impl CellData for Data {
        fn are_homogeneous(&self, other: &Self) -> bool {
            self == other
        }

        fn scale(&self, multiplier: usize) -> Self {
            Data(self.0 * multiplier as i32)
        }
    }

    #[test]
    fn test_changes() {
        let grid = Grid3d::new(
            GridConfig {
                chunks_num: [1, 1, 1],
                chunk_max_depth: 2,
                ..Default::default()
            },
            Data(0),
        );
        assert_eq!(grid.size(), [4, 4, 4]);
        assert_eq!(
            grid.iter().collect::<Vec<_>>(),
            [([0, 0, 0]..[4, 4, 4], &Data(0))]
        );
        assert_eq!(grid.preview(), [GridPreviewCell::Leaf(Data(0))]);
        let sampler = grid.sampler();
        assert_eq!(sampler.region_granularity_depth([0, 0, 0]..grid.size()), 0);

        let mut changes = Changes::new(2);
        changes.set([0, 0, 0], Data(42));
        assert_eq!(
            changes
                .chunks
                .iter()
                .map(|(range, items)| (*range, items.clone()))
                .collect::<Vec<_>>(),
            [(
                [0, 0, 0],
                vec![Some(Data(42)), None, None, None, None, None, None, None]
            )]
        );

        let grid = changes.apply(&grid);
        let sampler = grid.sampler();
        assert_eq!(sampler.region_granularity_depth([0, 0, 0]..grid.size()), 2);
        assert_eq!(
            grid.iter().collect::<Vec<_>>(),
            [
                ([2, 2, 2]..[4, 4, 4], &Data(0)),
                ([0, 2, 2]..[2, 4, 4], &Data(0)),
                ([2, 0, 2]..[4, 2, 4], &Data(0)),
                ([0, 0, 2]..[2, 2, 4], &Data(0)),
                ([2, 2, 0]..[4, 4, 2], &Data(0)),
                ([0, 2, 0]..[2, 4, 2], &Data(0)),
                ([2, 0, 0]..[4, 2, 2], &Data(0)),
                ([1, 1, 1]..[2, 2, 2], &Data(0)),
                ([0, 1, 1]..[1, 2, 2], &Data(0)),
                ([1, 0, 1]..[2, 1, 2], &Data(0)),
                ([0, 0, 1]..[1, 1, 2], &Data(0)),
                ([1, 1, 0]..[2, 2, 1], &Data(0)),
                ([0, 1, 0]..[1, 2, 1], &Data(0)),
                ([1, 0, 0]..[2, 1, 1], &Data(0)),
                ([0, 0, 0]..[1, 1, 1], &Data(42))
            ]
        );
        assert_eq!(
            grid.preview(),
            [GridPreviewCell::Branch(vec![
                GridPreviewCell::Branch(vec![
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0))
                ]),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0))
            ])]
        );

        let mut changes = Changes::new(2);
        changes.set_region([0, 0, 0]..[2, 2, 2], Data(42));

        let grid = changes.apply(&grid);
        let sampler = grid.sampler();
        assert_eq!(sampler.region_granularity_depth([0, 0, 0]..grid.size()), 1);
        assert_eq!(
            grid.iter().collect::<Vec<_>>(),
            [
                ([2, 2, 2]..[4, 4, 4], &Data(0)),
                ([0, 2, 2]..[2, 4, 4], &Data(0)),
                ([2, 0, 2]..[4, 2, 4], &Data(0)),
                ([0, 0, 2]..[2, 2, 4], &Data(0)),
                ([2, 2, 0]..[4, 4, 2], &Data(0)),
                ([0, 2, 0]..[2, 4, 2], &Data(0)),
                ([2, 0, 0]..[4, 2, 2], &Data(0)),
                ([0, 0, 0]..[2, 2, 2], &Data(42))
            ]
        );
        assert_eq!(
            grid.preview(),
            [GridPreviewCell::Branch(vec![
                GridPreviewCell::Leaf(Data(42)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0))
            ])]
        );

        let mut changes = Changes::new(2);
        changes.set_region([0, 0, 0]..[4, 4, 2], Data(42));

        let grid = changes.apply(&grid);
        let sampler = grid.sampler();
        assert_eq!(sampler.region_granularity_depth([0, 0, 0]..grid.size()), 1);
        assert_eq!(
            grid.iter().collect::<Vec<_>>(),
            [
                ([2, 2, 2]..[4, 4, 4], &Data(0)),
                ([0, 2, 2]..[2, 4, 4], &Data(0)),
                ([2, 0, 2]..[4, 2, 4], &Data(0)),
                ([0, 0, 2]..[2, 2, 4], &Data(0)),
                ([2, 2, 0]..[4, 4, 2], &Data(42)),
                ([0, 2, 0]..[2, 4, 2], &Data(42)),
                ([2, 0, 0]..[4, 2, 2], &Data(42)),
                ([0, 0, 0]..[2, 2, 2], &Data(42))
            ]
        );
        assert_eq!(
            grid.preview(),
            [GridPreviewCell::Branch(vec![
                GridPreviewCell::Leaf(Data(42)),
                GridPreviewCell::Leaf(Data(42)),
                GridPreviewCell::Leaf(Data(42)),
                GridPreviewCell::Leaf(Data(42)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0)),
                GridPreviewCell::Leaf(Data(0))
            ])]
        );

        let mut changes = Changes::new(2);
        changes.set_region([0, 0, 2]..[4, 4, 4], Data(42));

        let grid = changes.apply(&grid);
        let sampler = grid.sampler();
        assert_eq!(sampler.region_granularity_depth([0, 0, 0]..grid.size()), 0);
        assert_eq!(
            grid.iter().collect::<Vec<_>>(),
            [([0, 0, 0]..[4, 4, 4], &Data(42))]
        );
        assert_eq!(grid.preview(), [GridPreviewCell::Leaf(Data(42))]);
    }
}
