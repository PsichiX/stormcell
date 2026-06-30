use crate::{
    allocator::{Address, Allocator},
    topology::{Topology, Topology2d, Topology3d},
};
use std::ops::Range;

/// Value stored in a grid cell.
///
/// Must be [`Copy`] because cells live inside the [`Allocator`] arena. Two
/// adjacent cells holding "equal enough" data are merged into a coarser cell,
/// so the semantics of [`are_homogeneous`](CellData::are_homogeneous) directly
/// control how aggressively the grid coarsens.
pub trait CellData: Copy {
    /// Whether `self` and `other` are interchangeable for the purpose of
    /// merging neighbouring cells. Returning `true` more often yields a coarser,
    /// cheaper grid; returning `false` preserves detail.
    fn are_homogeneous(&self, other: &Self) -> bool;

    /// Scales this value as if it covered `multiplier` unit cells.
    ///
    /// Used to weight a coarse leaf by its area/volume when accumulating it
    /// against finer cells, as [`GridSample::data_scaled`] does.
    fn scale(&self, multiplier: usize) -> Self;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Cell<T: CellData, Topo: Topology> {
    Leaf { data: T },
    Branch { children: Topo::Children<Address> },
}

impl<T: CellData, Topo: Topology> Cell<T, Topo> {
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

/// Construction parameters for a [`Grid`].
///
/// `GridConfig` is intentionally `Clone`-but-not-`Copy`: it can be large and is
/// only cloned at well-defined points (e.g. when a [`Transformer`] rebuilds a
/// grid), not implicitly on every access.
///
/// [`Transformer`]: crate::pipeline::Transformer
#[derive(Debug, Clone)]
pub struct GridConfig<Topo: Topology = Topology3d> {
    /// Number of chunks along each axis. Each component is clamped to at least
    /// `1` by [`Grid::new`].
    pub chunks_num: Topo::Coord,
    /// Maximum subdivision depth within a chunk; sets `chunk_size = 1 << this`.
    pub chunk_max_depth: usize,
    /// Number of cells stored per arena page.
    pub page_capacity: u32,
    /// Number of pages pre-reserved in the arena.
    pub pages_capacity: u32,
    /// Upper bound on the sampler's leaf cache; `None` disables caching.
    pub sampler_cache_limit: Option<usize>,
}

impl<Topo: Topology> Default for GridConfig<Topo> {
    fn default() -> Self {
        Self {
            chunks_num: Topo::splat(1),
            chunk_max_depth: 0,
            page_capacity: 1024,
            pages_capacity: 32,
            sampler_cache_limit: None,
        }
    }
}

/// Octree-backed 3D grid (the default topology).
pub type Grid3d<T> = Grid<T, Topology3d>;

/// Quadtree-backed 2D grid.
pub type Grid2d<T> = Grid<T, Topology2d>;

/// A sparse, depth-adaptive grid of [`CellData`] values.
///
/// The grid is a fixed array of cubic **chunks** (`config.chunks_num` of them
/// along each axis); each chunk is the root of an adaptive subtree stored in the
/// shared [`Allocator`]. Uniform regions stay coarse leaves, detailed regions
/// subdivide down to `chunk_size = 1 << chunk_max_depth` cells per axis.
///
/// Grids are immutable in practice: edits and simulation steps go through a
/// [`Quantizer`](crate::quantizer::Quantizer) /
/// [`Transformer`](crate::pipeline::Transformer), which produce a new grid.
/// Cloning a grid deep-copies its arena.
///
/// # Invariants
/// - `chunk_size == 1 << chunk_max_depth` and
///   `size[axis] == chunks_num[axis] * chunk_size`, so every chunk is a cube
///   whose edge is a power of two.
/// - A well-formed grid never contains a branch whose children are all
///   [homogeneous](CellData::are_homogeneous) leaves; such branches are always
///   collapsed back into a single leaf by
///   [`CellEmitter::emit_branch_possibly_merged`](crate::quantizer::CellEmitter::emit_branch_possibly_merged).
#[derive(Clone)]
pub struct Grid<T: CellData, Topo: Topology = Topology3d> {
    pub(crate) config: GridConfig<Topo>,
    pub(crate) size: Topo::Coord,
    pub(crate) chunk_size: usize,
    pub(crate) chunks: Vec<Address>,
    pub(crate) allocator: Allocator<Cell<T, Topo>>,
}

impl<T: CellData, Topo: Topology> Grid<T, Topo> {
    /// Creates a grid with every cell initialised to `fill_data`.
    ///
    /// `config.chunks_num` is clamped to at least `1` per axis and the page
    /// capacities to at least `1`. The resulting [`size`](Grid::size) is
    /// `chunks_num * chunk_size` per axis.
    pub fn new(mut config: GridConfig<Topo>, fill_data: T) -> Self {
        config.chunks_num = Topo::map(config.chunks_num, |num| num.max(1));
        config.page_capacity = config.page_capacity.max(1);
        config.pages_capacity = config.pages_capacity.max(1);
        let mut allocator = Allocator::new(config.page_capacity, config.pages_capacity);
        let chunks = (0..Topo::volume(config.chunks_num))
            .map(|_| allocator.push(Cell::Leaf { data: fill_data }))
            .collect();
        let chunk_size = 1 << config.chunk_max_depth;
        let size = Topo::map(config.chunks_num, |num| num * chunk_size);
        Self {
            config,
            size,
            chunk_size,
            chunks,
            allocator,
        }
    }

    /// Returns a [`GridSampler`] for point and region queries against this grid.
    pub fn sampler(&self) -> GridSampler<'_, T, Topo> {
        GridSampler {
            grid: self,
            cache: Vec::with_capacity(self.config.sampler_cache_limit.unwrap_or_default()),
            limit: self.config.sampler_cache_limit.unwrap_or_default(),
        }
    }

    /// The configuration this grid was built with.
    #[inline]
    pub fn config(&self) -> &GridConfig<Topo> {
        &self.config
    }

    /// The total extent of the grid in cells per axis (`chunks_num * chunk_size`).
    #[inline]
    pub fn size(&self) -> Topo::Coord {
        self.size
    }

    /// The edge length, in cells, of one fully-subdivided chunk
    /// (`1 << chunk_max_depth`).
    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Resets every chunk to a single leaf holding `data`, discarding all
    /// existing subdivision.
    #[inline]
    pub fn reset(&mut self, data: T) {
        self.allocator = Allocator::new(self.config.page_capacity, self.config.pages_capacity);
        self.chunks = (0..Topo::volume(self.config.chunks_num))
            .map(|_| self.allocator.push(Cell::Leaf { data }))
            .collect();
    }

    /// Iterates over every leaf as a `(region, data)` pair, where `region` is
    /// the half-open box of grid coordinates the leaf covers. Order is
    /// unspecified (depth-first over chunks).
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Range<Topo::Coord>, &'a T)> + 'a {
        GridIter {
            allocator: &self.allocator,
            stack: self
                .chunks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, address)| {
                    let chunk_coords = Topo::from_linear_index(index, self.config.chunks_num);
                    let coords_from = Topo::map(chunk_coords, |coord| coord * self.chunk_size);
                    let coords_to = Topo::map(coords_from, |coord| coord + self.chunk_size);
                    (coords_from..coords_to, address)
                })
                .collect(),
        }
    }

    /// Builds an owned, fully-expanded tree view of every chunk, mainly for
    /// debugging and tests. Unlike the arena this materialises each node, so it
    /// is not suitable for large grids.
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

    /// Expands the sparse grid into a dense [`GridFlat`] buffer, writing each
    /// leaf's value into every cell it covers. `default_value` fills any cell
    /// not touched by a leaf (there should be none in a well-formed grid).
    pub fn flatten(&self, default_value: T) -> GridFlat<T, Topo> {
        let mut result = GridFlat {
            size: self.size,
            fields: vec![default_value; Topo::volume(self.size)],
        };
        let size = self.size;
        for (range, data) in self.iter() {
            Topo::for_each_coord(&range, |coord| {
                let index = Topo::linear_index(coord, size);
                result.fields[index] = *data;
            });
        }
        result
    }
}

/// Depth-first iterator over a grid's leaves, yielded by [`Grid::iter`].
pub struct GridIter<'a, T: CellData, Topo: Topology> {
    allocator: &'a Allocator<Cell<T, Topo>>,
    stack: Vec<(Range<Topo::Coord>, Address)>,
}

impl<'a, T: CellData, Topo: Topology> Iterator for GridIter<'a, T, Topo> {
    type Item = (Range<Topo::Coord>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((range, address)) = self.stack.pop() {
            if let Some(cell) = self.allocator.read(address) {
                match cell {
                    Cell::Leaf { data } => return Some((range, data)),
                    Cell::Branch { children } => {
                        let region_half_size = Topo::from_axes(|axis| {
                            (Topo::axis(range.end, axis) - Topo::axis(range.start, axis)) / 2
                        });
                        self.stack.extend(
                            Topo::children_as_slice(children)
                                .iter()
                                .copied()
                                .enumerate()
                                .map(|(index, address)| {
                                    let child_coords = Topo::child_offset(index);
                                    let coords_from = Topo::from_axes(|axis| {
                                        Topo::axis(range.start, axis)
                                            + Topo::axis(region_half_size, axis)
                                                * Topo::axis(child_coords, axis)
                                    });
                                    let coords_to = Topo::from_axes(|axis| {
                                        Topo::axis(range.start, axis)
                                            + Topo::axis(region_half_size, axis)
                                                * (Topo::axis(child_coords, axis) + 1)
                                    });
                                    (coords_from..coords_to, address)
                                }),
                        );
                    }
                }
            }
        }
        None
    }
}

/// An owned, expanded view of one cell subtree, produced by [`Grid::preview`].
///
/// The branch fan-out is erased into a `Vec` (rather than a topology-shaped
/// array), so this type is independent of [`Topology`].
#[derive(Debug, Clone, PartialEq)]
pub enum GridPreviewCell<T: CellData> {
    /// A leaf carrying its data value.
    Leaf(T),
    /// A branch with one child per sub-octant/-quadrant.
    Branch(Vec<GridPreviewCell<T>>),
}

impl<T: CellData> GridPreviewCell<T> {
    fn new<Topo: Topology>(cell: &Cell<T, Topo>, allocator: &Allocator<Cell<T, Topo>>) -> Self {
        match cell {
            Cell::Leaf { data } => GridPreviewCell::Leaf(*data),
            Cell::Branch { children } => {
                let children = Topo::children_as_slice(children)
                    .iter()
                    .copied()
                    .map(|address| {
                        let cell = allocator.read(address).unwrap_or_else(|| {
                            panic!("Could not read cell at address: {}", address)
                        });
                        GridPreviewCell::new(cell, allocator)
                    })
                    .collect();
                GridPreviewCell::Branch(children)
            }
        }
    }
}

/// A dense, fully-expanded copy of a grid, produced by [`Grid::flatten`].
///
/// Values are stored row-major with axis `0` contiguous; index a coordinate via
/// [`Topology::linear_index`].
#[derive(Debug, Clone)]
pub struct GridFlat<T: CellData, Topo: Topology = Topology3d> {
    size: Topo::Coord,
    fields: Vec<T>,
}

impl<T: CellData, Topo: Topology> GridFlat<T, Topo> {
    /// The extent of the buffer in cells per axis.
    #[inline]
    pub fn size(&self) -> Topo::Coord {
        self.size
    }

    /// The flat backing buffer (`size` product entries, axis `0` contiguous).
    #[inline]
    pub fn fields(&self) -> &[T] {
        &self.fields
    }
}

impl<T: CellData> GridFlat<T, Topology3d> {
    /// Reshapes the flat buffer into nested `[z][y][x]` vectors (3D only).
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

    /// Like [`into_3d`](GridFlat::into_3d) but maps each value through `f` while
    /// reshaping (3D only).
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

impl<T> std::fmt::Display for GridFlat<T, Topology3d>
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

/// The result of sampling a single point: the leaf covering it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSample<'a, T: CellData, Topo: Topology> {
    /// Lower corner of the leaf's region in grid coordinates.
    pub offset: Topo::Coord,
    /// Depth of the leaf below its chunk root (`0` = whole chunk).
    pub depth: usize,
    /// Edge length of the leaf in cells (`chunk_size >> depth`).
    pub size: usize,
    /// The leaf's data value.
    pub data: &'a T,
}

impl<'a, T: CellData, Topo: Topology> GridSample<'a, T, Topo> {
    /// The half-open region `offset..offset + size` this leaf covers.
    pub fn range(&self) -> Range<Topo::Coord> {
        self.offset..Topo::map(self.offset, |coord| coord + self.size)
    }

    /// Number of unit cells the leaf covers (`size.pow(DIMENSIONS)`).
    pub fn area(&self) -> usize {
        Topo::cell_volume(self.size)
    }

    /// The leaf's value [scaled](CellData::scale) by its [`area`](GridSample::area).
    pub fn data_scaled(&self) -> T {
        self.data.scale(self.area())
    }
}

/// A leaf intersected by a region query, with the size of the overlap.
///
/// Yielded by [`GridSampler::sample_region`]; lets callers area-weight coarse
/// leaves that only partially fall inside the queried region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSampleView<'a, T: CellData, Topo: Topology> {
    /// The intersected leaf.
    pub sample: GridSample<'a, T, Topo>,
    /// Number of unit cells of the leaf that lie inside the queried region.
    pub overlap_area: usize,
}

impl<'a, T: CellData, Topo: Topology> GridSampleView<'a, T, Topo> {
    /// The leaf's value [scaled](CellData::scale) by its
    /// [`overlap_area`](GridSampleView::overlap_area).
    pub fn data_scaled(&self) -> T {
        self.sample.data.scale(self.overlap_area)
    }
}

/// Point and region query interface over a [`Grid`].
///
/// A sampler keeps an optional bounded cache of recently returned leaves
/// (sized by [`GridConfig::sampler_cache_limit`]) so repeated nearby
/// [`sample`](GridSampler::sample) queries - common in neighbourhood-based
/// quantizers - avoid re-walking the tree.
pub struct GridSampler<'a, T: CellData, Topo: Topology> {
    pub(crate) grid: &'a Grid<T, Topo>,
    cache: Vec<GridSample<'a, T, Topo>>,
    limit: usize,
}

impl<'a, T: CellData, Topo: Topology> GridSampler<'a, T, Topo> {
    /// Trims the leaf cache back down to its configured limit.
    pub fn evict_least_accessed_samples(&mut self) {
        if self.cache.len() > self.limit {
            self.cache.drain(0..(self.cache.len() - self.limit));
        }
    }

    /// The configuration of the underlying grid.
    #[inline]
    pub fn grid_config(&self) -> &GridConfig<Topo> {
        self.grid.config()
    }

    /// The extent of the underlying grid in cells per axis.
    #[inline]
    pub fn grid_size(&self) -> Topo::Coord {
        self.grid.size()
    }

    /// The chunk edge length of the underlying grid.
    #[inline]
    pub fn grid_chunk_size(&self) -> usize {
        self.grid.chunk_size()
    }

    /// Returns the leaf covering `coords`, or `None` if `coords` is out of
    /// bounds. May populate the leaf cache.
    pub fn sample(&mut self, coords: Topo::Coord) -> Option<GridSample<'_, T, Topo>> {
        if (0..Topo::DIMENSIONS)
            .any(|axis| Topo::axis(coords, axis) >= Topo::axis(self.grid.size, axis))
        {
            return None;
        }
        if self.limit > 0 {
            for (index, sample) in self.cache.iter().enumerate().rev() {
                let range = sample.range();
                let inside = (0..Topo::DIMENSIONS).all(|axis| {
                    let coord = Topo::axis(coords, axis);
                    coord >= Topo::axis(range.start, axis) && coord < Topo::axis(range.end, axis)
                });
                if inside {
                    let result = *sample;
                    if index + 1 < self.cache.len() {
                        self.cache.swap(index, index + 1);
                    }
                    return Some(result);
                }
            }
        }
        let chunk_coords = Topo::map(coords, |coord| coord / self.grid.chunk_size);
        let chunk_index = Topo::linear_index(chunk_coords, self.grid.config.chunks_num);
        let mut address = self.grid.chunks.get(chunk_index).copied()?;
        let mut local_coords = Topo::map(coords, |coord| coord % self.grid.chunk_size);
        let mut offset = Topo::map(chunk_coords, |coord| coord * self.grid.chunk_size);
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
                    let child_coords = Topo::map(local_coords, |coord| coord / size);
                    let child_index = Topo::child_index(local_coords, size);
                    address = Topo::children_as_slice(children)
                        .get(child_index)
                        .copied()?;
                    local_coords = Topo::map(local_coords, |coord| coord % size);
                    offset = Topo::from_axes(|axis| {
                        Topo::axis(offset, axis) + Topo::axis(child_coords, axis) * size
                    });
                }
            }
        }
        None
    }

    /// Iterates over every leaf overlapping `region`, each paired with its
    /// overlap area. Does not use or populate the cache.
    pub fn sample_region(
        &self,
        region: Range<Topo::Coord>,
    ) -> impl Iterator<Item = GridSampleView<'a, T, Topo>> {
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
                    let chunk_coords = Topo::from_linear_index(index, self.grid.config.chunks_num);
                    let offset = Topo::map(chunk_coords, |coord| coord * self.grid.chunk_size);
                    if !Topo::overlaps(&region, offset, self.grid.chunk_size) {
                        return None;
                    }
                    Some((address, offset, self.grid.chunk_size, 0))
                })
                .collect(),
        }
    }

    /// The maximum leaf depth found among leaves overlapping `region` - i.e.
    /// how finely subdivided that region currently is. Used by quantizers to
    /// match a neighbourhood's resolution before processing it.
    pub fn region_granularity_depth(&self, region: Range<Topo::Coord>) -> usize {
        fn walk_cell<T: CellData, Topo: Topology>(
            address: Address,
            allocator: &Allocator<Cell<T, Topo>>,
            region: &Range<Topo::Coord>,
            offset: Topo::Coord,
            mut depth: usize,
            mut size: usize,
        ) -> usize {
            if !Topo::overlaps(region, offset, size) {
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
                    for (child_index, address) in Topo::children_as_slice(children)
                        .iter()
                        .copied()
                        .enumerate()
                    {
                        let child_coords = Topo::child_offset(child_index);
                        let offset = Topo::from_axes(|axis| {
                            Topo::axis(offset, axis) + Topo::axis(child_coords, axis) * size
                        });
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
                let chunk_coords = Topo::from_linear_index(index, self.grid.config.chunks_num);
                let offset = Topo::map(chunk_coords, |coord| coord * self.grid.chunk_size);
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

/// Iterator over leaves overlapping a region, yielded by
/// [`GridSampler::sample_region`].
// TODO: add caching?
pub struct GridSampleIter<'a, T: CellData, Topo: Topology> {
    allocator: &'a Allocator<Cell<T, Topo>>,
    region: Range<Topo::Coord>,
    stack: Vec<(Address, Topo::Coord, usize, usize)>,
}

impl<'a, T: CellData, Topo: Topology> Iterator for GridSampleIter<'a, T, Topo> {
    type Item = GridSampleView<'a, T, Topo>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((address, offset, mut size, mut depth)) = self.stack.pop() {
            if let Some(cell) = self.allocator.read(address) {
                match cell {
                    Cell::Leaf { data } => {
                        let overlap_area = Topo::overlap_volume(&self.region, offset, size);
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
                        for (child_index, address) in Topo::children_as_slice(children)
                            .iter()
                            .copied()
                            .enumerate()
                        {
                            let child_coords = Topo::child_offset(child_index);
                            let offset = Topo::from_axes(|axis| {
                                Topo::axis(offset, axis) + Topo::axis(child_coords, axis) * size
                            });
                            if !Topo::overlaps(&self.region, offset, size) {
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
    use crate::topology::Topology2d;

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
        let grid = Grid3d::new(
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

        let grid = Grid3d::new(
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

        let grid = Grid3d::new(
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

        let grid = Grid3d::new(
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

    #[test]
    fn test_grid_2d() {
        // A quadtree grid: 4 children per branch, [usize; 2] coordinates.
        let grid: Grid<Data, Topology2d> = Grid::new(
            GridConfig {
                chunks_num: [2, 3],
                chunk_max_depth: 2,
                ..Default::default()
            },
            Data(0),
        );
        assert_eq!(grid.size(), [8, 12]);
        assert_eq!(grid.chunk_size(), 4);
        assert_eq!(Topology2d::CHILDREN, 4);

        let mut sampler = grid.sampler();
        assert_eq!(sampler.sample([0, 0]).unwrap().data, &Data(0));
        assert_eq!(sampler.sample([7, 11]).unwrap().data, &Data(0));
        assert!(sampler.sample([8, 0]).is_none());
        assert!(sampler.sample([0, 12]).is_none());

        // A uniform grid collapses to one leaf per chunk (2 * 3 = 6 chunks).
        assert_eq!(grid.iter().count(), 6);

        // Edit a single cell and confirm it subdivides down to a 1x1 leaf.
        use crate::changes::Changes;
        let mut changes: Changes<Data, Topology2d> = Changes::new(4);
        changes.set([1, 1], Data(42));
        let grid = changes.apply(&grid);
        let mut sampler = grid.sampler();
        let sample = sampler.sample([1, 1]).unwrap();
        assert_eq!(sample.data, &Data(42));
        assert_eq!(sample.size, 1);
        assert_eq!(sample.area(), 1);
        // Neighboring cell stays the coarse background value.
        assert_eq!(sampler.sample([6, 10]).unwrap().data, &Data(0));

        let flat = grid.flatten(Data(0));
        assert_eq!(flat.size(), [8, 12]);
        assert_eq!(flat.fields().len(), 96);
        assert_eq!(flat.fields().iter().filter(|d| d.0 == 42).count(), 1);
    }
}
