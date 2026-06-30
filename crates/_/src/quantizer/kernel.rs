use crate::{
    allocator::Address,
    quantizer::{CellContext, Quantizer},
};

/// Adapter that forces the wrapped quantizer to run at the *finest* resolution.
///
/// It recursively subdivides every cell down to size `1` before delegating to
/// the inner quantizer, re-merging homogeneous results on the way back up. Use
/// it for rules that must see individual unit cells (e.g. per-cell stencils)
/// regardless of how coarse the source grid is.
pub struct KernelQuantizer<Q> {
    /// The inner per-cell quantizer.
    pub quantizer: Q,
}

impl<Q: Quantizer> Quantizer for KernelQuantizer<Q> {
    type CellData = Q::CellData;
    type Topology = Q::Topology;

    fn quantize(&self, mut context: CellContext<Self::CellData, Self::Topology>) -> Address {
        if context.cell_size <= 1 {
            self.quantizer.quantize(context)
        } else {
            let children = context.subdivide(|ctx| self.quantize(ctx));
            context
                .emitter
                .emit_branch_possibly_merged(children, |cells| self.quantizer.merge(cells))
        }
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
    struct Temperature(pub f32);

    impl CellData for Temperature {
        fn are_homogeneous(&self, other: &Self) -> bool {
            (self.0 - other.0).abs() < 0.01
        }

        fn scale(&self, multiplier: usize) -> Self {
            Temperature(self.0 * multiplier as f32)
        }
    }

    struct Diffusion {
        pub rate: f32,
    }

    impl Quantizer for Diffusion {
        type CellData = Temperature;
        type Topology = Topology3d;

        fn quantize(&self, context: CellContext<Self::CellData, Self::Topology>) -> Address {
            let grid_size = context.sampler.grid_size();

            let cx = context.region.start[0];
            let cy = context.region.start[1];
            let cz = context.region.start[2];
            let nx = cx.saturating_sub(1);
            let ny = cy.saturating_sub(1);
            let nz = cz.saturating_sub(1);
            let px = (cx.saturating_add(1)).min(grid_size[0].saturating_sub(1));
            let py = (cy.saturating_add(1)).min(grid_size[1].saturating_sub(1));
            let pz = (cz.saturating_add(1)).min(grid_size[2].saturating_sub(1));

            let vnx = context.sampler.sample([nx, cy, cz]).unwrap().data.0;
            let vpx = context.sampler.sample([px, cy, cz]).unwrap().data.0;
            let vny = context.sampler.sample([cx, ny, cz]).unwrap().data.0;
            let vpy = context.sampler.sample([cx, py, cz]).unwrap().data.0;
            let vnz = context.sampler.sample([cx, cy, nz]).unwrap().data.0;
            let vpz = context.sampler.sample([cx, cy, pz]).unwrap().data.0;
            let v = context.cell_data.0;
            let neighbor_sum = vnx + vpx + vny + vpy + vnz + vpz;
            let value = v + (neighbor_sum - v * 6.0) * self.rate;

            context.emitter.emit_leaf(Temperature(value))
        }
    }

    #[test]
    fn test_kernel_quantizer() {
        let quantizer = KernelQuantizer {
            quantizer: Diffusion { rate: 0.1 },
        };

        let mut grid = Grid3d::new(
            GridConfig {
                chunk_max_depth: 2,
                sampler_cache_limit: Some(64),
                ..Default::default()
            },
            Temperature(0.0),
        );
        let mut changes = Changes::default();
        changes.set([1, 1, 1], Temperature(100.0));
        grid = changes.apply(&grid);
        let flat = grid.flatten(Temperature(2.0));
        assert_eq!(flat.fields().iter().map(|v| v.0).sum::<f32>(), 100.0);

        for _ in 0..50 {
            grid = Transformer::new(&quantizer, &grid).execute();
            let flat = grid.flatten(Temperature(2.0));
            let sum = flat.fields().iter().map(|v| v.0).sum::<f32>();
            assert!((sum - 100.0).abs() < 0.1, "sum: {sum}");
        }
    }
}
