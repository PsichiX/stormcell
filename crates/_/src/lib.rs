//! # stormcell
//!
//! Adaptive Resolution Grid processing for simulations.
//!
//! `stormcell` stores spatial data in a **sparse, depth-adaptive tree** - an
//! octree in 3D, a quadtree in 2D - where uniform regions collapse into a
//! single coarse cell and only heterogeneous regions are subdivided down to
//! fine resolution. This keeps both memory and per-step compute proportional to
//! the *detail* in the field rather than to its bounding volume.
//!
//! ## Core concepts
//!
//! - [`grid::Grid`] - the container. It is split into a fixed array of
//!   equally-sized **chunks**; each chunk is the root of an adaptive subtree of
//!   cells. A cell is either a `Leaf` (holding one [`CellData`] value covering
//!   its whole region) or a `Branch` (holding one child per
//!   sub-octant/-quadrant).
//! - [`topology::Topology`] - abstracts dimensionality. It defines the
//!   coordinate type, the branch fan-out, and all the coordinate arithmetic, so
//!   the same `Grid` code drives both [`Topology2D`](topology::Topology2D)
//!   (quadtree) and [`Topology3D`](topology::Topology3D) (octree, the default).
//! - [`quantizer::Quantizer`] - a transformation rule. Given a cell's region
//!   and data it emits the next cell (leaf or branch), optionally sampling
//!   neighbours. [`pipeline::Transformer`] walks a whole grid through a
//!   quantizer to produce a new grid.
//! - [`changes::Changes`] - a sparse staging buffer of point/region edits that
//!   is itself a [`Quantizer`](quantizer::Quantizer), so applying edits reuses
//!   the same machinery.
//! - [`allocator::Allocator`] - a paged arena that owns every cell and hands
//!   out lightweight [`Address`](allocator::Address) handles.
//!
//! ## Coordinates
//!
//! All coordinates are non-negative integer grid indices
//! ([`Topology::Coord`](crate::topology::Topology::Coord), e.g. `[usize; 3]`).
//! Regions are half-open ranges `start..end`. Axis `0` is the fastest-varying
//! (contiguous) axis in every linear-index conversion.
//!
//! [`CellData`]: grid::CellData

pub mod allocator;
pub mod changes;
pub mod grid;
pub mod pipeline;
pub mod quantizer;
pub mod topology;
