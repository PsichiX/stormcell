use crate::{
    allocator::Address,
    grid::{CellData, Grid},
    pipeline::Transformer,
    quantizer::{CellContext, Quantizer},
};
use std::{collections::BTreeMap, ops::Range};

pub struct Changes<T: CellData> {
    chunk_size: usize,
    chunks: BTreeMap<[usize; 3], Vec<Option<T>>>,
}

impl<T: CellData> Default for Changes<T> {
    fn default() -> Self {
        Self::new(32)
    }
}

impl<T: CellData> Changes<T> {
    #[inline]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            chunks: Default::default(),
        }
    }

    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    #[inline]
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    pub fn set(&mut self, coord: [usize; 3], value: T) {
        let chunk_coord = coord.map(|c| c / self.chunk_size);
        let local_coord = coord.map(|c| c % self.chunk_size);
        let index = local_coord[0] * self.chunk_size * self.chunk_size
            + local_coord[1] * self.chunk_size
            + local_coord[2];
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| vec![None; self.chunk_size * self.chunk_size * self.chunk_size])
            .as_mut_slice();
        chunk[index] = Some(value);
    }

    pub fn set_region(&mut self, region: Range<[usize; 3]>, value: T) {
        for coord in (region.start[0]..region.end[0])
            .flat_map(move |x| (region.start[1]..region.end[1]).map(move |y| [x, y]))
            .flat_map(move |[x, y]| (region.start[2]..region.end[2]).map(move |z| ([x, y, z])))
        {
            self.set(coord, value);
        }
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = ([usize; 3], T)>) {
        for (coord, value) in values {
            self.set(coord, value);
        }
    }

    pub fn sample(&self, coord: [usize; 3]) -> Option<&T> {
        let chunk_coord = coord.map(|c| c / self.chunk_size);
        let local_coord = coord.map(|c| c % self.chunk_size);
        let index = local_coord[0] * self.chunk_size * self.chunk_size
            + local_coord[1] * self.chunk_size
            + local_coord[2];
        self.chunks.get(&chunk_coord)?.get(index)?.as_ref()
    }

    pub fn sample_region(
        &self,
        region: Range<[usize; 3]>,
    ) -> impl Iterator<Item = ([usize; 3], Option<&T>)> {
        (region.start[0]..region.end[0])
            .flat_map(move |x| (region.start[1]..region.end[1]).map(move |y| [x, y]))
            .flat_map(move |[x, y]| (region.start[2]..region.end[2]).map(move |z| ([x, y, z])))
            .map(|coord| (coord, self.sample(coord)))
    }

    pub fn is_region_homogeneous(&self, region: Range<[usize; 3]>, default_value: &T) -> Option<T> {
        let mut value = None;
        for coord in (region.start[0]..region.end[0])
            .flat_map(move |x| (region.start[1]..region.end[1]).map(move |y| [x, y]))
            .flat_map(move |[x, y]| (region.start[2]..region.end[2]).map(move |z| ([x, y, z])))
        {
            let sample = self.sample(coord).unwrap_or(default_value);
            if let Some(value) = value {
                if !sample.are_homogeneous(value) {
                    return None;
                }
            }
            value = Some(sample);
        }
        value.copied()
    }

    pub fn apply(self, grid: &Grid<T>) -> Grid<T> {
        Transformer::new(&self, grid).execute()
    }
}

impl<T: CellData> Quantizer for Changes<T> {
    type CellData = T;

    fn quantize(&self, mut context: CellContext<Self::CellData>) -> Address {
        if let Some(value) = self.is_region_homogeneous(context.region.clone(), context.cell_data) {
            context.emitter.emit_leaf(value)
        } else {
            let children = context.subdivide_region().map(|subregion| {
                let context = unsafe { context.subregion(&subregion) };
                self.quantize(context)
            });
            context.emitter.emit_branch(children)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid, GridConfig, GridPreviewCell};

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
        let grid = Grid::new(
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
            [GridPreviewCell::Branch(
                [
                    GridPreviewCell::Branch(
                        [
                            GridPreviewCell::Leaf(Data(42)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0)),
                            GridPreviewCell::Leaf(Data(0))
                        ]
                        .into()
                    ),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0))
                ]
                .into()
            )]
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
            [GridPreviewCell::Branch(
                [
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0))
                ]
                .into()
            )]
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
            [GridPreviewCell::Branch(
                [
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(42)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0)),
                    GridPreviewCell::Leaf(Data(0))
                ]
                .into()
            )]
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
