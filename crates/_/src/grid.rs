use crate::{
    allocator::{Address, Allocator},
    quantizer::COORD_INDICES,
};
use std::ops::Range;

pub trait CellData: Copy {
    fn are_homogeneous(&self, other: &Self) -> bool;

    fn scale(&self, multiplier: usize) -> Self;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Cell<T: CellData> {
    Leaf { data: T },
    Branch { children: [Address; 8] },
}

impl<T: CellData> Cell<T> {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Cell::Leaf { .. })
    }

    #[inline]
    pub fn data(&self) -> Option<&T> {
        match self {
            Cell::Leaf { data } => Some(data),
            Cell::Branch { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub chunks_num: [usize; 3],
    pub chunk_max_depth: usize,
    pub page_capacity: u32,
    pub pages_capacity: u32,
    pub sampler_cache_limit: Option<usize>,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            chunks_num: [1, 1, 1],
            chunk_max_depth: 0,
            page_capacity: 1024,
            pages_capacity: 32,
            sampler_cache_limit: None,
        }
    }
}

#[derive(Clone)]
pub struct Grid<T: CellData> {
    pub(crate) config: GridConfig,
    pub(crate) size: [usize; 3],
    pub(crate) chunk_size: usize,
    pub(crate) chunks: Vec<Address>,
    pub(crate) allocator: Allocator<Cell<T>>,
}

impl<T: CellData> Grid<T> {
    pub fn new(mut config: GridConfig, fill_data: T) -> Self {
        config.chunks_num = config.chunks_num.map(|num| num.max(1));
        config.page_capacity = config.page_capacity.max(1);
        config.pages_capacity = config.pages_capacity.max(1);
        let mut allocator = Allocator::new(config.page_capacity, config.pages_capacity);
        let chunks = (0..config.chunks_num.iter().product::<usize>())
            .map(|_| allocator.push(Cell::Leaf { data: fill_data }))
            .collect();
        let chunk_size = 1 << config.chunk_max_depth;
        let size = config.chunks_num.map(|num| num * chunk_size);
        Self {
            config,
            size,
            chunk_size,
            chunks,
            allocator,
        }
    }

    pub fn sampler(&self) -> GridSampler<T> {
        GridSampler {
            grid: self,
            cache: Vec::with_capacity(self.config.sampler_cache_limit.unwrap_or_default()),
            limit: self.config.sampler_cache_limit.unwrap_or_default(),
        }
    }

    #[inline]
    pub fn config(&self) -> &GridConfig {
        &self.config
    }

    #[inline]
    pub fn size(&self) -> [usize; 3] {
        self.size
    }

    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    #[inline]
    pub fn reset(&mut self, data: T) {
        self.allocator = Allocator::new(self.config.page_capacity, self.config.pages_capacity);
        self.chunks = (0..self.config.chunks_num.iter().product::<usize>())
            .map(|_| self.allocator.push(Cell::Leaf { data }))
            .collect();
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Range<[usize; 3]>, &'a T)> + 'a {
        GridIter {
            allocator: &self.allocator,
            stack: self
                .chunks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, address)| {
                    let chunk_coords = [
                        index % self.config.chunks_num[0],
                        (index / self.config.chunks_num[0]) % self.config.chunks_num[1],
                        index / (self.config.chunks_num[0] * self.config.chunks_num[1]),
                    ];
                    let coords_from = chunk_coords.map(|coord| coord * self.chunk_size);
                    let coords_to = coords_from.map(|coord| coord + self.chunk_size);
                    (coords_from..coords_to, address)
                })
                .collect(),
        }
    }

    pub fn preview(&self) -> Vec<GridPreviewCell<T>> {
        self.chunks
            .iter()
            .copied()
            .map(|address| {
                let cell = self
                    .allocator
                    .read(address)
                    .unwrap_or_else(|| panic!("Could not read cell at address: {}", address));
                GridPreviewCell::new(cell, &self.allocator)
            })
            .collect()
    }

    pub fn flatten(&self, default_value: T) -> GridFlat<T> {
        let mut result = GridFlat {
            size: self.size,
            fields: vec![default_value; self.size[0] * self.size[1] * self.size[2]],
        };
        for (range, data) in self.iter() {
            let start_x = range.start[0];
            let start_y = range.start[1];
            let start_z = range.start[2];
            let end_x = range.end[0];
            let end_y = range.end[1];
            let end_z = range.end[2];
            for z in start_z..end_z {
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let index = z * self.size[0] * self.size[1] + y * self.size[0] + x;
                        result.fields[index] = *data;
                    }
                }
            }
        }
        result
    }
}

pub struct GridIter<'a, T: CellData> {
    allocator: &'a Allocator<Cell<T>>,
    stack: Vec<(Range<[usize; 3]>, Address)>,
}

impl<'a, T: CellData> Iterator for GridIter<'a, T> {
    type Item = (Range<[usize; 3]>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((range, address)) = self.stack.pop() {
            if let Some(cell) = self.allocator.read(address) {
                match cell {
                    Cell::Leaf { data } => return Some((range, data)),
                    Cell::Branch { children } => {
                        self.stack.extend(children.iter().copied().enumerate().map(
                            |(index, address)| {
                                let child_coords = [index & 1, (index >> 1) & 1, (index >> 2) & 1];
                                let region_half_size = [
                                    (range.end[0] - range.start[0]) / 2,
                                    (range.end[1] - range.start[1]) / 2,
                                    (range.end[2] - range.start[2]) / 2,
                                ];
                                let coords_from = COORD_INDICES.map(|index| {
                                    range.start[index]
                                        + region_half_size[index] * child_coords[index]
                                });
                                let coords_to = COORD_INDICES.map(|index| {
                                    range.start[index]
                                        + region_half_size[index] * (child_coords[index] + 1)
                                });
                                (coords_from..coords_to, address)
                            },
                        ));
                    }
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GridPreviewCell<T: CellData> {
    Leaf(T),
    Branch(Box<[GridPreviewCell<T>; 8]>),
}

impl<T: CellData> GridPreviewCell<T> {
    fn new(cell: &Cell<T>, allocator: &Allocator<Cell<T>>) -> Self {
        match cell {
            Cell::Leaf { data } => GridPreviewCell::Leaf(*data),
            Cell::Branch { children } => {
                let children = children.map(|address| {
                    let cell = allocator
                        .read(address)
                        .unwrap_or_else(|| panic!("Could not read cell at address: {}", address));
                    GridPreviewCell::new(cell, allocator)
                });
                GridPreviewCell::Branch(Box::new(children))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridFlat<T: CellData> {
    size: [usize; 3],
    fields: Vec<T>,
}

impl<T: CellData> GridFlat<T> {
    #[inline]
    pub fn size(&self) -> [usize; 3] {
        self.size
    }

    #[inline]
    pub fn fields(&self) -> &[T] {
        &self.fields
    }

    #[inline]
    pub fn into_3d(self) -> Vec<Vec<Vec<T>>> {
        let mut result = Vec::with_capacity(self.size[2]);
        for z in 0..self.size[2] {
            let mut layer = Vec::with_capacity(self.size[1]);
            for y in 0..self.size[1] {
                let start = z * self.size[0] * self.size[1] + y * self.size[0];
                let end = start + self.size[0];
                layer.push(self.fields[start..end].to_vec());
            }
            result.push(layer);
        }
        result
    }

    #[inline]
    pub fn map_into_3d<U, F: Fn(&T) -> U>(self, f: &F) -> Vec<Vec<Vec<U>>> {
        let mut result = Vec::with_capacity(self.size[2]);
        for z in 0..self.size[2] {
            let mut layer = Vec::with_capacity(self.size[1]);
            for y in 0..self.size[1] {
                let start = z * self.size[0] * self.size[1] + y * self.size[0];
                let end = start + self.size[0];
                layer.push(self.fields[start..end].iter().map(f).collect());
            }
            result.push(layer);
        }
        result
    }
}

impl<T> std::fmt::Display for GridFlat<T>
where
    T: CellData + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GridFlat({:?})", self.size)?;
        for z in 0..self.size[2] {
            writeln!(f, " {{0..{} x 0..{} x {}}}", self.size[0], self.size[1], z)?;
            for y in 0..self.size[1] {
                let start = z * self.size[0] * self.size[1] + y * self.size[0];
                let end = start + self.size[0];
                writeln!(f, " {:?}", &self.fields[start..end])?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSample<'a, T: CellData> {
    pub offset: [usize; 3],
    pub depth: usize,
    pub size: usize,
    pub data: &'a T,
}

impl<'a, T: CellData> GridSample<'a, T> {
    pub fn range(&self) -> Range<[usize; 3]> {
        self.offset..self.offset.map(|coord| coord + self.size)
    }

    pub fn area(&self) -> usize {
        self.size * self.size * self.size
    }

    pub fn data_scaled(&self) -> T {
        self.data.scale(self.area())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSampleView<'a, T: CellData> {
    pub sample: GridSample<'a, T>,
    pub overlap_area: usize,
}

impl<'a, T: CellData> GridSampleView<'a, T> {
    pub fn data_scaled(&self) -> T {
        self.sample.data.scale(self.overlap_area)
    }
}

pub struct GridSampler<'a, T: CellData> {
    pub(crate) grid: &'a Grid<T>,
    cache: Vec<GridSample<'a, T>>,
    limit: usize,
}

impl<'a, T: CellData> GridSampler<'a, T> {
    pub fn evict_least_accessed_samples(&mut self) {
        if self.cache.len() > self.limit {
            self.cache.drain(0..(self.cache.len() - self.limit));
        }
    }

    #[inline]
    pub fn grid_config(&self) -> &GridConfig {
        self.grid.config()
    }

    #[inline]
    pub fn grid_size(&self) -> [usize; 3] {
        self.grid.size()
    }

    #[inline]
    pub fn grid_chunk_size(&self) -> usize {
        self.grid.chunk_size()
    }

    pub fn sample(&mut self, coords: [usize; 3]) -> Option<GridSample<T>> {
        if coords[0] >= self.grid.size[0]
            || coords[1] >= self.grid.size[1]
            || coords[2] >= self.grid.size[2]
        {
            return None;
        }
        if self.limit > 0 {
            for (index, sample) in self.cache.iter().enumerate().rev() {
                let range = sample.range();
                if coords[0] >= range.start[0]
                    && coords[0] < range.end[0]
                    && coords[1] >= range.start[1]
                    && coords[1] < range.end[1]
                    && coords[2] >= range.start[2]
                    && coords[2] < range.end[2]
                {
                    let result = *sample;
                    if index + 1 < self.cache.len() {
                        self.cache.swap(index, index + 1);
                    }
                    return Some(result);
                }
            }
        }
        let chunk_coords = coords.map(|coord| coord / self.grid.chunk_size);
        let chunk_index =
            chunk_coords[2] * self.grid.config.chunks_num[1] * self.grid.config.chunks_num[0]
                + chunk_coords[1] * self.grid.config.chunks_num[0]
                + chunk_coords[0];
        let mut address = self.grid.chunks.get(chunk_index).copied()?;
        let mut local_coords = coords.map(|coord| coord % self.grid.chunk_size);
        let mut offset = chunk_coords.map(|coord| coord * self.grid.chunk_size);
        let mut depth = 0;
        let mut size = self.grid.chunk_size;
        while let Some(cell) = self.grid.allocator.read(address) {
            match cell {
                Cell::Leaf { data } => {
                    let sample = GridSample {
                        offset,
                        depth,
                        size,
                        data,
                    };
                    if self.limit > 0 {
                        self.cache.push(sample);
                    }
                    return Some(sample);
                }
                Cell::Branch { children } => {
                    depth += 1;
                    size >>= 1;
                    let child_coords = local_coords.map(|coord| coord / size);
                    let child_index = child_coords[2] * 4 + child_coords[1] * 2 + child_coords[0];
                    address = children.get(child_index).copied()?;
                    local_coords = local_coords.map(|coord| coord % size);
                    offset = COORD_INDICES.map(|index| offset[index] + child_coords[index] * size);
                }
            }
        }
        None
    }

    pub fn sample_region(
        &self,
        region: Range<[usize; 3]>,
    ) -> impl Iterator<Item = GridSampleView<'a, T>> {
        GridSampleIter {
            allocator: &self.grid.allocator,
            region: region.clone(),
            stack: self
                .grid
                .chunks
                .iter()
                .copied()
                .enumerate()
                .filter_map(move |(index, address)| {
                    let chunk_coords = [
                        index % self.grid.config.chunks_num[0],
                        (index / self.grid.config.chunks_num[0]) % self.grid.config.chunks_num[1],
                        index / (self.grid.config.chunks_num[0] * self.grid.config.chunks_num[1]),
                    ];
                    let offset = chunk_coords.map(|coord| coord * self.grid.chunk_size);
                    if region.start[0] >= offset[0] + self.grid.chunk_size
                        || region.start[1] >= offset[1] + self.grid.chunk_size
                        || region.start[2] >= offset[2] + self.grid.chunk_size
                        || region.end[0] <= offset[0]
                        || region.end[1] <= offset[1]
                        || region.end[2] <= offset[2]
                    {
                        return None;
                    }
                    Some((address, offset, self.grid.chunk_size, 0))
                })
                .collect(),
        }
    }

    pub fn region_granularity_depth(&self, region: Range<[usize; 3]>) -> usize {
        fn walk_cell<T: CellData>(
            address: Address,
            allocator: &Allocator<Cell<T>>,
            region: &Range<[usize; 3]>,
            offset: [usize; 3],
            mut depth: usize,
            mut size: usize,
        ) -> usize {
            if region.start[0] >= offset[0] + size
                || region.start[1] >= offset[1] + size
                || region.start[2] >= offset[2] + size
                || region.end[0] <= offset[0]
                || region.end[1] <= offset[1]
                || region.end[2] <= offset[2]
            {
                return 0;
            }
            match allocator
                .read(address)
                .unwrap_or_else(|| panic!("Could not read cell at address: {}", address))
            {
                Cell::Leaf { .. } => depth,
                Cell::Branch { children } => {
                    size >>= 1;
                    depth += 1;
                    let mut max_depth = depth;
                    for (child_index, address) in children.iter().copied().enumerate() {
                        let offset = COORD_INDICES
                            .map(|index| offset[index] + (child_index >> index & 1) * size);
                        max_depth = max_depth
                            .max(walk_cell(address, allocator, region, offset, depth, size));
                    }
                    max_depth
                }
            }
        }
        self.grid
            .chunks
            .iter()
            .copied()
            .enumerate()
            .map(|(index, address)| {
                let chunk_coords = [
                    index % self.grid.config.chunks_num[0],
                    (index / self.grid.config.chunks_num[0]) % self.grid.config.chunks_num[1],
                    index / (self.grid.config.chunks_num[0] * self.grid.config.chunks_num[1]),
                ];
                let offset = chunk_coords.map(|coord| coord * self.grid.chunk_size);
                walk_cell(
                    address,
                    &self.grid.allocator,
                    &region,
                    offset,
                    0,
                    self.grid.chunk_size,
                )
            })
            .max()
            .unwrap_or(0)
    }
}

// TODO: add caching?
pub struct GridSampleIter<'a, T: CellData> {
    allocator: &'a Allocator<Cell<T>>,
    region: Range<[usize; 3]>,
    stack: Vec<(Address, [usize; 3], usize, usize)>,
}

impl<'a, T: CellData> Iterator for GridSampleIter<'a, T> {
    type Item = GridSampleView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((address, offset, mut size, mut depth)) = self.stack.pop() {
            if let Some(cell) = self.allocator.read(address) {
                match cell {
                    Cell::Leaf { data } => {
                        let overlap_area = COORD_INDICES
                            .map(|index| {
                                let start = offset[index].max(self.region.start[index]);
                                let end = (offset[index] + size).min(self.region.end[index]);
                                end.saturating_sub(start)
                            })
                            .iter()
                            .product();
                        return Some(GridSampleView {
                            sample: GridSample {
                                offset,
                                depth,
                                size,
                                data,
                            },
                            overlap_area,
                        });
                    }
                    Cell::Branch { children } => {
                        size >>= 1;
                        depth += 1;
                        for (child_index, address) in children.iter().copied().enumerate() {
                            let offset = COORD_INDICES
                                .map(|index| offset[index] + (child_index >> index & 1) * size);
                            if self.region.start[0] >= offset[0] + size
                                || self.region.start[1] >= offset[1] + size
                                || self.region.start[2] >= offset[2] + size
                                || self.region.end[0] <= offset[0]
                                || self.region.end[1] <= offset[1]
                                || self.region.end[2] <= offset[2]
                            {
                                continue;
                            }
                            self.stack.push((address, offset, size, depth));
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_grid() {
        let grid = Grid::new(
            GridConfig {
                chunks_num: [1, 1, 1],
                chunk_max_depth: 0,
                ..Default::default()
            },
            Data(0),
        );
        let mut sampler = grid.sampler();
        assert_eq!(grid.size(), [1, 1, 1]);
        assert_eq!(grid.chunk_size(), 1);
        assert_eq!(sampler.sample([0, 0, 0]).unwrap().data, &Data(0));
        assert!(sampler.sample([1, 1, 1]).is_none());
        assert_eq!(
            sampler
                .sample_region([0, 0, 0]..[2, 2, 2])
                .collect::<Vec<_>>(),
            [GridSampleView {
                sample: GridSample {
                    offset: [0, 0, 0],
                    depth: 0,
                    size: 1,
                    data: &Data(0)
                },
                overlap_area: 1
            }]
        );

        let grid = Grid::new(
            GridConfig {
                chunks_num: [1, 1, 1],
                chunk_max_depth: 1,
                ..Default::default()
            },
            Data(0),
        );
        let mut sampler = grid.sampler();
        assert_eq!(grid.size(), [2, 2, 2]);
        assert_eq!(grid.chunk_size(), 2);
        assert_eq!(sampler.sample([0, 0, 0]).unwrap().data, &Data(0));
        assert_eq!(sampler.sample([1, 1, 1]).unwrap().data, &Data(0));
        assert_eq!(
            sampler
                .sample_region([0, 0, 0]..[2, 2, 2])
                .collect::<Vec<_>>(),
            [GridSampleView {
                sample: GridSample {
                    offset: [0, 0, 0],
                    depth: 0,
                    size: 2,
                    data: &Data(0)
                },
                overlap_area: 8
            }]
        );

        let grid = Grid::new(
            GridConfig {
                chunks_num: [2, 2, 2],
                chunk_max_depth: 0,
                ..Default::default()
            },
            Data(0),
        );
        let mut sampler = grid.sampler();
        assert_eq!(grid.size(), [2, 2, 2]);
        assert_eq!(grid.chunk_size(), 1);
        assert_eq!(sampler.sample([0, 0, 0]).unwrap().data, &Data(0));
        assert_eq!(sampler.sample([1, 1, 1]).unwrap().data, &Data(0));

        let grid = Grid::new(
            GridConfig {
                chunks_num: [4, 4, 1],
                chunk_max_depth: 3,
                ..Default::default()
            },
            Data(0),
        );
        let mut sampler = grid.sampler();
        assert_eq!(grid.size(), [32, 32, 8]);
        assert_eq!(grid.chunk_size(), 8);
        for index in 0..10 {
            assert_eq!(
                sampler.sample([index, index, index % 8]).unwrap().data,
                &Data(0)
            );
            if index >= 8 {
                assert!(sampler.sample([index, index, index]).is_none());
            }
        }
        assert_eq!(
            sampler
                .sample_region([1, 1, 0]..[9, 9, 1])
                .collect::<Vec<_>>(),
            [
                GridSampleView {
                    sample: GridSample {
                        offset: [8, 8, 0],
                        depth: 0,
                        size: 8,
                        data: &Data(0)
                    },
                    overlap_area: 1
                },
                GridSampleView {
                    sample: GridSample {
                        offset: [0, 8, 0],
                        depth: 0,
                        size: 8,
                        data: &Data(0)
                    },
                    overlap_area: 7
                },
                GridSampleView {
                    sample: GridSample {
                        offset: [8, 0, 0],
                        depth: 0,
                        size: 8,
                        data: &Data(0)
                    },
                    overlap_area: 7
                },
                GridSampleView {
                    sample: GridSample {
                        offset: [0, 0, 0],
                        depth: 0,
                        size: 8,
                        data: &Data(0)
                    },
                    overlap_area: 49
                }
            ]
        );
    }
}
