use super::*;
use crate::graph::coupled::{
    CoupledPlugin, EncodedFunction, EvaluationPolicy, PluginResource, PluginState,
    PreparedSlopeFunction,
};

fn slope(value: f64, lo: f64, m: f64, hi: f64) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(vec![(value, lo), (m, hi)]).unwrap()
}
fn knots(points: Vec<[f64; 2]>) -> EncodedFunction {
    EncodedFunction::Knots {
        function: PiecewiseLinearFunction::new(points).unwrap(),
    }
}
fn encoded(value: f64, m: f64, policy: EvaluationPolicy) -> EncodedFunction {
    EncodedFunction::SlopeSegments {
        function: slope(value, 0., m, 10.),
        policy,
    }
}

#[test]
fn encoded_numeric_dependencies_and_objective_keep_their_literal_encodings() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::EncodedPiecewiseLinearResource {
                initial: 0.,
                bounds: Window::new(0., 10.).unwrap(),
                functions: vec![
                    knots(vec![[0., 1.], [4., 5.]]),
                    encoded(2., 1., EvaluationPolicy::StrictDomain),
                ],
            },
            CoupledResource::EncodedPiecewiseLinearDependent {
                parent: 0,
                initial: 0.,
                bounds: Window::new(0., 100.).unwrap(),
                functions: vec![
                    encoded(2., 0., EvaluationPolicy::StrictDomain),
                    knots(vec![[0., 2.], [10., 2.]]),
                ],
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 1,
                functions: vec![encoded(5., 0., EvaluationPolicy::NativeTolerance); 2],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(10.));
    assert_eq!(kernel.certify(&[0, 1]), Some((true, [10., 10.])));
}

#[test]
fn encoded_native_endpoint_snapping_requires_proved_acceptance() {
    for (initial, expected) in [
        (-0.00005, Some((true, [0., 0.]))),
        (10.00005, Some((true, [0., 0.]))),
        (-0.001, None),
    ] {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::EncodedPiecewiseLinearResource {
                initial,
                bounds: Window::new(-100., 100.).unwrap(),
                functions: vec![encoded(7., 1., EvaluationPolicy::NativeTolerance)],
            }])
            .unwrap();
        assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), expected);
    }
}

#[test]
fn raw_interval_operator_certificate_preserves_one_shared_initial_value() {
    for policy in [
        descriptor::FrontOperationPolicy::StrictDomain,
        descriptor::FrontOperationPolicy::NativeFrontAlgebra,
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut definition = CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 4.).unwrap(),
            bounds: Window::new(0., 4.).unwrap(),
            increments: vec![1., -1.],
            functions: vec![slope(0., 0., 1., 5.), slope(5., 0., -1., 5.)],
            policy,
        };
        graph.set_coupled_resources(&[definition.clone()]).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        // At initial x in [0,3], charges are (x+1) and (5-x), not
        // independent per-arc minima. Both operator policies yield six.
        assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(6.));
        assert_eq!(kernel.certify(&[0, 1]), Some((true, [6., 6.])));
        let CoupledResource::SlopeIntervalCost { increments, .. } = &mut definition else {
            unreachable!()
        };
        increments[0] = 0.25;
        graph.set_coupled_resources(&[definition]).unwrap();
        assert_eq!(graph.coupled_kernel().unwrap().certify(&[0, 1]), None);
    }
}

#[derive(Debug)]
struct Load {
    capacity: u64,
    proof: bool,
    increase_work: bool,
}
impl CoupledPlugin for Load {
    fn validate(&self, _: usize) -> crate::Result<()> {
        Ok(())
    }
    fn root(&self, _: bool) -> PluginState {
        PluginState::new(0u64)
    }
    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        _: bool,
    ) -> crate::Result<Option<PluginState>> {
        let value = state.get::<u64>().unwrap() + [2, 3][arc];
        Ok((value <= self.capacity).then(|| PluginState::new(value)))
    }
    fn join(
        &self,
        state: &PluginState,
        suffix: &mut dyn Iterator<Item = usize>,
    ) -> crate::Result<bool> {
        Ok(*state.get::<u64>().unwrap() + suffix.map(|i| [2, 3][i]).sum::<u64>() <= self.capacity)
    }
    fn certify(&self, arcs: &[usize], work: &mut usize, deadline: Option<Instant>) -> Option<bool> {
        if !self.proof || deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        if self.increase_work {
            *work = work.checked_mul(2)?;
            return Some(true);
        }
        *work = work.checked_sub(arcs.len() + 1)?;
        let total = arcs
            .iter()
            .try_fold(0u64, |sum, &i| sum.checked_add(*[2, 3].get(i)?))?;
        Some(total <= self.capacity)
    }
}

#[test]
fn plugins_opt_into_directed_proof_without_promoting_numerical_transitions() {
    for (capacity, proof, increase_work, expected) in [
        (5, false, false, None),
        (5, true, false, Some((true, [0., 0.]))),
        (4, true, false, Some((false, [0., 0.]))),
        (5, true, true, None),
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(Load {
                capacity,
                proof,
                increase_work,
            }))])
            .unwrap();
        assert_eq!(graph.coupled_kernel().unwrap().certify(&[0, 1]), expected);
        let mut work = 1000;
        assert_eq!(
            graph
                .coupled_kernel()
                .unwrap()
                .certify_with_budget(&[0, 1], &mut work, None),
            expected
        );
        assert!(work <= 1000, "a rejected plugin cannot refund proof work");
    }
}
