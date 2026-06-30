use std::ops::Range;

/// Describes the dimensionality of a [`crate::grid::Grid`] and all the
/// coordinate / subdivision arithmetic that depends on it.
///
/// A topology decides:
/// - the coordinate type ([`Topology::Coord`], e.g. `[usize; 2]` or `[usize; 3]`),
/// - the branch fan-out ([`Topology::Children`], e.g. `[A; 4]` for a quadtree
///   or `[A; 8]` for an octree),
/// - and the operations that map between coordinates, linear indices and child
///   slots.
///
/// Most methods have a default implementation expressed purely in terms of
/// [`Topology::from_axes`] / [`Topology::axis`] / [`Topology::DIMENSIONS`], so a
/// concrete topology only needs to provide a handful of array-shaped methods.
pub trait Topology: Copy + std::fmt::Debug + 'static {
    /// Integer coordinate, e.g. `[usize; 2]` or `[usize; 3]`.
    type Coord: Copy + Ord + std::hash::Hash + std::fmt::Debug;

    /// Fixed-size container holding one entry per child of a branch cell.
    ///
    /// For an octree this is `[A; 8]`, for a quadtree `[A; 4]`. It must be
    /// `Copy` so that branch cells can live in the `Copy`-only arena allocator.
    type Children<A>: Copy + std::fmt::Debug
    where
        A: Copy + std::fmt::Debug;

    /// Number of axes (2 for a quadtree, 3 for an octree).
    const DIMENSIONS: usize;

    /// Number of children per branch (`2.pow(DIMENSIONS)`).
    const CHILDREN: usize;

    /// Read a single axis of a coordinate (`0` is the fastest-varying axis).
    fn axis(coord: Self::Coord, axis: usize) -> usize;

    /// Build a coordinate from a per-axis function (`0..DIMENSIONS`).
    fn from_axes(f: impl FnMut(usize) -> usize) -> Self::Coord;

    /// Build a [`Topology::Children`] from a per-child function (`0..CHILDREN`).
    fn children_from_fn<A>(f: impl FnMut(usize) -> A) -> Self::Children<A>
    where
        A: Copy + std::fmt::Debug;

    /// Borrow the children as a slice for iteration.
    fn children_as_slice<A>(children: &Self::Children<A>) -> &[A]
    where
        A: Copy + std::fmt::Debug;

    /// Coordinate with the same `value` on every axis.
    #[inline]
    fn splat(value: usize) -> Self::Coord {
        Self::from_axes(|_| value)
    }

    /// Map every axis of a coordinate through `f`.
    #[inline]
    fn map(coord: Self::Coord, mut f: impl FnMut(usize) -> usize) -> Self::Coord {
        Self::from_axes(|axis| f(Self::axis(coord, axis)))
    }

    /// Product of all axes (number of cells contained in a box of this size).
    #[inline]
    fn volume(coord: Self::Coord) -> usize {
        (0..Self::DIMENSIONS).fold(1, |acc, axis| acc * Self::axis(coord, axis))
    }

    /// Volume of a cube cell with the given edge length (`size.pow(DIMENSIONS)`).
    #[inline]
    fn cell_volume(size: usize) -> usize {
        (0..Self::DIMENSIONS).fold(1, |acc, _| acc * size)
    }

    /// Flatten a coordinate into a linear index within a box of `size`
    /// (axis `0` is contiguous / fastest-varying).
    #[inline]
    fn linear_index(coord: Self::Coord, size: Self::Coord) -> usize {
        let mut index = 0;
        let mut stride = 1;
        for axis in 0..Self::DIMENSIONS {
            index += Self::axis(coord, axis) * stride;
            stride *= Self::axis(size, axis);
        }
        index
    }

    /// Inverse of [`Topology::linear_index`].
    #[inline]
    fn from_linear_index(index: usize, size: Self::Coord) -> Self::Coord {
        Self::from_axes(|axis| {
            let mut stride = 1;
            for lower in 0..axis {
                stride *= Self::axis(size, lower);
            }
            (index / stride) % Self::axis(size, axis)
        })
    }

    /// The `0`/`1`-per-axis offset of child `index` within its parent.
    #[inline]
    fn child_offset(index: usize) -> Self::Coord {
        Self::from_axes(|axis| (index >> axis) & 1)
    }

    /// The child slot containing `local` coordinates, where `half` is the
    /// half-size (child edge length) of the parent cell.
    #[inline]
    fn child_index(local: Self::Coord, half: usize) -> usize {
        (0..Self::DIMENSIONS).fold(0, |acc, axis| {
            acc | ((Self::axis(local, axis) / half) << axis)
        })
    }

    /// Whether the axis-aligned box `offset..offset+size` overlaps `region`.
    #[inline]
    fn overlaps(region: &Range<Self::Coord>, offset: Self::Coord, size: usize) -> bool {
        for axis in 0..Self::DIMENSIONS {
            let off = Self::axis(offset, axis);
            if Self::axis(region.start, axis) >= off + size {
                return false;
            }
            if Self::axis(region.end, axis) <= off {
                return false;
            }
        }
        true
    }

    /// Volume of the intersection between `region` and `offset..offset+size`.
    #[inline]
    fn overlap_volume(region: &Range<Self::Coord>, offset: Self::Coord, size: usize) -> usize {
        let mut volume = 1;
        for axis in 0..Self::DIMENSIONS {
            let off = Self::axis(offset, axis);
            let start = off.max(Self::axis(region.start, axis));
            let end = (off + size).min(Self::axis(region.end, axis));
            volume *= end.saturating_sub(start);
        }
        volume
    }

    /// Invoke `f` for every coordinate in the half-open box `range`.
    #[inline]
    fn for_each_coord(range: &Range<Self::Coord>, mut f: impl FnMut(Self::Coord)) {
        for axis in 0..Self::DIMENSIONS {
            if Self::axis(range.start, axis) >= Self::axis(range.end, axis) {
                return;
            }
        }
        let mut counter: Vec<usize> = (0..Self::DIMENSIONS)
            .map(|axis| Self::axis(range.start, axis))
            .collect();
        loop {
            f(Self::from_axes(|axis| counter[axis]));
            let mut axis = 0;
            loop {
                if axis == Self::DIMENSIONS {
                    return;
                }
                counter[axis] += 1;
                if counter[axis] < Self::axis(range.end, axis) {
                    break;
                }
                counter[axis] = Self::axis(range.start, axis);
                axis += 1;
            }
        }
    }
}

/// Two-dimensional topology: `[usize; 2]` coordinates, quadtree subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Topology2d;

impl Topology for Topology2d {
    type Coord = [usize; 2];
    type Children<A>
        = [A; 4]
    where
        A: Copy + std::fmt::Debug;

    const DIMENSIONS: usize = 2;
    const CHILDREN: usize = 4;

    #[inline]
    fn axis(coord: Self::Coord, axis: usize) -> usize {
        coord[axis]
    }

    #[inline]
    fn from_axes(mut f: impl FnMut(usize) -> usize) -> Self::Coord {
        [f(0), f(1)]
    }

    #[inline]
    fn children_from_fn<A>(mut f: impl FnMut(usize) -> A) -> Self::Children<A>
    where
        A: Copy + std::fmt::Debug,
    {
        [f(0), f(1), f(2), f(3)]
    }

    #[inline]
    fn children_as_slice<A>(children: &Self::Children<A>) -> &[A]
    where
        A: Copy + std::fmt::Debug,
    {
        children
    }
}

/// Three-dimensional topology: `[usize; 3]` coordinates, octree subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Topology3d;

impl Topology for Topology3d {
    type Coord = [usize; 3];
    type Children<A>
        = [A; 8]
    where
        A: Copy + std::fmt::Debug;

    const DIMENSIONS: usize = 3;
    const CHILDREN: usize = 8;

    #[inline]
    fn axis(coord: Self::Coord, axis: usize) -> usize {
        coord[axis]
    }

    #[inline]
    fn from_axes(mut f: impl FnMut(usize) -> usize) -> Self::Coord {
        [f(0), f(1), f(2)]
    }

    #[inline]
    fn children_from_fn<A>(mut f: impl FnMut(usize) -> A) -> Self::Children<A>
    where
        A: Copy + std::fmt::Debug,
    {
        [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)]
    }

    #[inline]
    fn children_as_slice<A>(children: &Self::Children<A>) -> &[A]
    where
        A: Copy + std::fmt::Debug,
    {
        children
    }
}
