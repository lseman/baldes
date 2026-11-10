use crate::graph::coupled::{CoupledPlugin, CoupledResource, PluginResource, PluginState};
use crate::{Graph, Result};
use std::time::Instant;

#[derive(Debug)]
struct CostPlugin {
    proof: Option<[f64; 2]>,
    inflate_work: bool,
}
impl CoupledPlugin for CostPlugin {
    fn objective_bounds(&self, _: usize, work: &mut usize, _: Option<Instant>) -> Option<[f64; 2]> {
        if self.inflate_work {
            *work += 1;
        }
        self.proof
    }
    fn has_costs(&self) -> bool {
        true
    }
    fn objective(&self, _: &PluginState, _: bool) -> Result<f64> {
        Ok(3.5)
    }
    fn validate(&self, _: usize) -> Result<()> {
        Ok(())
    }
    fn root(&self, _: bool) -> PluginState {
        PluginState::new(())
    }
    fn extend(&self, s: &PluginState, _: usize, _: bool) -> Result<Option<PluginState>> {
        Ok(Some(s.clone()))
    }
    fn join(&self, _: &PluginState, _: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        Ok(true)
    }
    fn certify(&self, _: &[usize], _: &mut usize, _: Option<Instant>) -> Option<bool> {
        Some(true)
    }
    fn certify_objective(
        &self,
        _: &[usize],
        work: &mut usize,
        _: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        if self.inflate_work {
            *work += 1;
        }
        self.proof.map(|cost| (true, cost))
    }
}
#[test]
fn global_plugin_bounds_preserve_signed_coefficients_and_reject_bad_proofs() {
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let g = graph(Some([3., 4.]), false);
    assert_eq!(
        g.coupled_kernel()
            .unwrap()
            .objective_walk_lower_bound(&g, 1, &mut 10_000, deadline),
        Some(-9.)
    );
    for (proof, inflate) in [
        (None, false),
        (Some([4., 3.]), false),
        (Some([f64::NAN, 4.]), false),
        (Some([0., f64::INFINITY]), false),
        (Some([3., 4.]), true),
    ] {
        let g = graph(proof, inflate);
        assert!(g
            .coupled_kernel()
            .unwrap()
            .objective_walk_lower_bound(&g, 1, &mut 10_000, deadline)
            .is_none());
    }
}
fn graph(proof: Option<[f64; 2]>, inflate_work: bool) -> Graph<2> {
    let mut g = Graph::<2>::new(2, 0).unwrap();
    g.add_arc(0, 1, 0.).unwrap();
    g.set_coupled_resources(&[
        CoupledResource::Plugin(PluginResource::new(CostPlugin {
            proof,
            inflate_work,
        })),
        CoupledResource::CustomResource {
            increments: vec![2.],
            weights: vec![3.],
            forward_bound: None,
            backward_bound: None,
        },
    ])
    .unwrap();
    g.set_coupled_objective_coefficients(&[(1, -2.)]).unwrap();
    g
}
#[test]
fn cost_certificate_is_added_to_other_objectives_and_phase_one_suppresses_it() {
    let g = graph(Some([3., 4.]), false);
    let kernel = g.coupled_kernel().unwrap();
    assert_eq!(kernel.certify(&[0]), Some((true, [-9., -8.])));
    let mut phase_one = kernel.clone();
    phase_one.suppress_costs();
    assert_eq!(phase_one.certify(&[0]), Some((true, [0.; 2])));
}
#[test]
fn missing_malformed_or_budget_inflating_cost_proofs_remain_unknown() {
    for proof in [
        None,
        Some([4., 3.]),
        Some([f64::NAN, 4.]),
        Some([0., f64::INFINITY]),
    ] {
        assert!(graph(proof, false)
            .coupled_kernel()
            .unwrap()
            .certify(&[0])
            .is_none());
    }
    assert!(graph(Some([3., 4.]), true)
        .coupled_kernel()
        .unwrap()
        .certify(&[0])
        .is_none());
}
