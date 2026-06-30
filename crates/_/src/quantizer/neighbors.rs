use crate::{
    allocator::Address,
    quantizer::{CellContext, Quantizer},
    topology::Topology,
};

/// Adapter that matches each cell's resolution to that of its neighbourhood
/// before delegating to the wrapped quantizer.
///
/// Before processing a cell it inspects the surrounding region (the cell grown
/// by [`tile_margin`](NeighborsQuantizer::tile_margin) on every side) and
/// subdivides down to the finest depth found there, so rules that read
/// neighbours aren't starved of detail across a coarse/fine boundary.
pub struct NeighborsQuantizer<Q> {
    /// How far, in cells, to look past the cell's own bounds when measuring the
    /// neighbourhood's resolution (clamped to at least `1`).
    pub tile_margin: usize,
    /// The inner per-cell quantizer.
    pub quantizer: Q,
}

impl<Q: Quantizer> NeighborsQuantizer<Q> {
    fn subquantize(
        &self,
        mut context: CellContext<Q::CellData, Q::Topology>,
        desired_depth: usize,
    ) -> Address {
        if context.depth < desired_depth {
            let children = context.subdivide(|ctx| self.subquantize(ctx, desired_depth));
            context
                .emitter
                .emit_branch_possibly_merged(children, |cells| self.quantizer.merge(cells))
        } else {
            self.quantizer.quantize(context)
        }
    }
}

impl<Q: Quantizer> Quantizer for NeighborsQuantizer<Q> {
    type CellData = Q::CellData;
    type Topology = Q::Topology;

    fn quantize(&self, context: CellContext<Self::CellData, Self::Topology>) -> Address {
        if context.depth >= context.sampler.grid_config().chunk_max_depth {
            return self.quantizer.quantize(context);
        }
        let margin = self.tile_margin.max(1);
        let coords_from = <Q::Topology as Topology>::map(context.region.start, |coord| {
            coord.saturating_sub(margin)
        });
        let coords_to = <Q::Topology as Topology>::map(context.region.end, |coord| {
            coord.saturating_add(margin)
        });
        let desired_depth = context
            .sampler
            .region_granularity_depth(coords_from..coords_to)
            .max(context.depth);
        self.subquantize(context, desired_depth)
    }

    fn merge(&self, cells: &[&Self::CellData]) -> Self::CellData {
        self.quantizer.merge(cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        changes::Changes,
        grid::{CellData, Grid3d, GridConfig},
        pipeline::Transformer,
        topology::Topology3d,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Virus(pub bool);

    impl CellData for Virus {
        fn are_homogeneous(&self, other: &Self) -> bool {
            self == other
        }

        fn scale(&self, multiplier: usize) -> Self {
            if self.0 {
                if multiplier > 0 {
                    Virus(true)
                } else {
                    Virus(false)
                }
            } else {
                Virus(false)
            }
        }
    }

    struct Contamination;

    impl Quantizer for Contamination {
        type CellData = Virus;
        type Topology = Topology3d;

        fn quantize(&self, context: CellContext<Self::CellData, Self::Topology>) -> Address {
            let from = context.region.start.map(|coord| coord.saturating_sub(1));
            let to = context.region.end.map(|coord| coord.saturating_add(1));
            let value = context
                .sampler
                .sample_region(from..to)
                .any(|sample| sample.sample.data.0);

            context.emitter.emit_leaf(Virus(value))
        }
    }

    #[test]
    fn test_neighbors_quantizer() {
        let quantizer = NeighborsQuantizer {
            tile_margin: 1,
            quantizer: Contamination,
        };

        let mut grid = Grid3d::new(
            GridConfig {
                chunk_max_depth: 2,
                sampler_cache_limit: Some(64),
                ..Default::default()
            },
            Virus(false),
        );
        let mut changes = Changes::default();
        changes.set([3, 3, 3], Virus(true));
        grid = changes.apply(&grid);
        let flat = grid.flatten(Virus(false));
        assert_eq!(flat.fields().iter().filter(|v| v.0).count(), 1);
        assert_eq!(
            flat.map_into_3d(&|v| v.0),
            [
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, true]
                ]
            ]
        );

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Virus(false));
        assert_eq!(flat.fields().iter().filter(|v| v.0).count(), 8);
        assert_eq!(
            flat.map_into_3d(&|v| v.0),
            [
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, false, false]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, true, true],
                    [false, false, true, true]
                ],
                [
                    [false, false, false, false],
                    [false, false, false, false],
                    [false, false, true, true],
                    [false, false, true, true]
                ]
            ]
        );

        grid = Transformer::new(&quantizer, &grid).execute();
        let flat = grid.flatten(Virus(false));
        assert_eq!(flat.fields().iter().filter(|v| v.0).count(), 64);
        assert_eq!(
            flat.map_into_3d(&|v| v.0),
            [
                [
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true]
                ],
                [
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true]
                ],
                [
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true]
                ],
                [
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true],
                    [true, true, true, true]
                ]
            ]
        );
    }
}
