use super::*;
use crate::graph::coupled::{CoupledResource, PairedResourceMode};
fn graph(edges: &[(usize, usize, f64)], vertices: usize, upper: f64) -> Graph<2> {
    let mut graph = Graph::new(vertices, 1).unwrap();
    for v in 0..vertices {
        graph.set_bounds(v, 0, 0., upper).unwrap();
    }
    for &(tail, head, increment) in edges {
        let id = graph.add_arc(tail, head, -1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = increment;
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    graph
}
fn inputs<'a>(
    bases: &'a [PreparedSlopeFunction],
    edges: &[(usize, usize, f64)],
    upper: f64,
) -> Vec<MachineArcFunctionInput<'a>> {
    let affine = |value| MachineRootInput::Affine(descriptor::RawAffine::new(value, 0.).unwrap());
    bases
        .iter()
        .enumerate()
        .map(|(physical_arc, base)| {
            let _ = edges[physical_arc];
            MachineArcFunctionInput {
                physical_arc,
                tail_bounds: Window::new(0., upper).unwrap(),
                waiting_bounds: Window::new(0., upper).unwrap(),
                waiting_rate: 0,
                base,
                dependent: affine(0.25),
                dependent_lower: affine(0.),
                dependent_upper: affine(8.),
            }
        })
        .collect()
}
#[test]
fn raw_parallel_chain_matches_recorded_domains_and_installs_atomically() {
    let edges = [(0, 1, 1.), (1, 2, 1.), (0, 1, 1.), (1, 2, 1.)];
    let mut graph = graph(&edges, 3, 4.);
    let bases: Vec<_> = edges
        .iter()
        .map(|e| PreparedSlopeFunction::from_prepared_pairs(vec![(e.2, 0.), (0., 4.)]).unwrap())
        .collect();
    let raw = inputs(&bases, &edges, 4.);
    assert!(matches!(
        graph.preprocess_paired_base_domains(0, &raw, FrontierBudget::default()),
        FrontierOutcome::Unknown(_)
    ));
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    let identity = graph.coupled_kernel().unwrap().cache_identity().clone();
    let domains = match graph.preprocess_paired_base_domains(0, &raw, FrontierBudget::default()) {
        FrontierOutcome::Complete(v) => v,
        other => panic!("{other:?}"),
    };
    assert!(std::sync::Arc::ptr_eq(
        &identity,
        graph.coupled_kernel().unwrap().cache_identity()
    ));
    assert_eq!(
        domains
            .vertex_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(0., 3.), (1., 3.), (0., 4.)]
    );
    assert_eq!(
        domains
            .arc_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(1., 3.), (2., 4.), (1., 3.), (2., 4.)]
    );
    let root = MachineRootInput::Affine(descriptor::RawAffine::new(0., 0.).unwrap());
    let before = graph.vertices[0].bounds[0];
    assert!(matches!(
        domains.install_front_program(
            &mut graph,
            &raw,
            root,
            FrontierBudget {
                max_operations: 0,
                ..FrontierBudget::default()
            }
        ),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
    assert_eq!(graph.vertices[0].bounds[0], before);
    assert!(matches!(
        domains.install_front_program(&mut graph, &raw, root, FrontierBudget::default()),
        FrontierOutcome::Complete(())
    ));
    let grid = crate::graph::buckets::BucketGraph::prepare(&graph, [1.], 10, true).unwrap();
    assert_eq!(grid.middle, 1.8);
    assert_eq!(grid.upper[grid.internal[graph.source]][0], 3.);
    assert_eq!(grid.lower[grid.internal[graph.sink]][0], 0.);
    assert_eq!(grid.arc_lower[1][0], 2.);
    assert_eq!(graph.coupled_pricing_path_cost(&[0, 1]).unwrap(), Some(0.5));
}
#[test]
fn raw_fork_merge_uses_incoming_minimum_and_original_outgoing_upper() {
    // Native base-domain-projection/fork trace: both alternatives participate;
    // this is not a source-control shortest path or a stronger inverse closure.
    let edges = [
        (0, 1, 0.5),
        (0, 2, 1.),
        (1, 3, 0.5),
        (2, 3, 0.5),
        (3, 4, 0.5),
    ];
    let mut graph = graph(&edges, 5, 2.5);
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    let bases: Vec<_> = edges
        .iter()
        .map(|e| PreparedSlopeFunction::from_prepared_pairs(vec![(e.2, 0.), (0., 2.5)]).unwrap())
        .collect();
    let raw = inputs(&bases, &edges, 2.5);
    let domains = match graph.preprocess_paired_base_domains(0, &raw, FrontierBudget::default()) {
        FrontierOutcome::Complete(v) => v,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        domains
            .vertex_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(0., 2.), (0.5, 2.), (1., 2.), (1., 2.), (0., 2.5)]
    );
    assert_eq!(
        domains
            .arc_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(0.5, 2.), (1., 2.), (1., 2.), (1.5, 2.), (1.5, 2.5)]
    );
    graph.set_bounds(1, 0, 0., 2.).unwrap();
    let identity = graph.coupled_kernel().unwrap().cache_identity().clone();
    let root = MachineRootInput::Affine(descriptor::RawAffine::new(0., 0.).unwrap());
    assert!(matches!(
        domains.install_front_program(&mut graph, &raw, root, FrontierBudget::default()),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
    ));
    assert!(std::sync::Arc::ptr_eq(
        &identity,
        graph.coupled_kernel().unwrap().cache_identity()
    ));
}

#[test]
fn zero_consumption_cycle_retains_the_recorded_weak_domain_relaxation() {
    // Native cycle trace: the zero cycle keeps the incoming minimum at0,
    // and its outgoing inverse keeps the customer upper at2.5.
    let edges = [(0, 1, 0.5), (1, 1, 0.), (1, 2, 0.5)];
    let mut graph = graph(&edges, 3, 2.5);
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    let bases: Vec<_> = edges
        .iter()
        .map(|e| PreparedSlopeFunction::from_prepared_pairs(vec![(e.2, 0.), (0., 2.5)]).unwrap())
        .collect();
    let raw = inputs(&bases, &edges, 2.5);
    let domains = match graph.preprocess_paired_base_domains(0, &raw, FrontierBudget::default()) {
        FrontierOutcome::Complete(v) => v,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        domains
            .vertex_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(0., 2.), (0., 2.5), (0., 2.5)]
    );
    assert_eq!(
        domains
            .arc_bounds()
            .iter()
            .map(|w| (w.lower, w.upper))
            .collect::<Vec<_>>(),
        [(0.5, 2.5), (0., 2.5), (0.5, 2.5)]
    );
    graph.arc_mut(0).unwrap().consumption[0] = 1.;
    assert!(matches!(
        graph.preprocess_paired_base_domains(0, &raw, FrontierBudget::default()),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
    ));
}
