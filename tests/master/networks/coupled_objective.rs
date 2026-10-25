use super::*;
use baldes::graph::coupled::CoupledResource;

fn problem() -> Problem<2> {
    let mut first = network(7, &[(10., &[0]), (0., &[0])], 1);
    first.multiplicity.upper = 1.;
    first
        .graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![-9., 0., 5., 0.],
            weights: vec![1., 0., 1., 0.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let mut second = network(42, &[(2., &[0]), (0., &[0])], 1);
    second.multiplicity.upper = 1.;
    second
        .graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0., 0., 4., 0.],
            weights: vec![1., 0., 1., 0.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![first, second],
    }
}

#[test]
fn paired_network_costs_reach_certified_enumeration_fixing_and_integer_solving() {
    use baldes::graph::coupled::{PairedResourceArc, PairedResourceMode, PiecewiseLinearFunction};
    let constant = |value| PiecewiseLinearFunction::new(vec![[0., value], [12., value]]).unwrap();
    let mut problem = problem();
    for (network, increments) in problem
        .networks
        .iter_mut()
        .zip([[-9., 0., 5., 0.], [0., 0., 4., 0.]])
    {
        network
            .graph
            .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
                initial: constant(0.),
                initial_base: Window::new(0., 0.).unwrap(),
                arcs: increments
                    .into_iter()
                    .map(|increment| PairedResourceArc {
                        arrival: PiecewiseLinearFunction::new(vec![[0., 1.], [10., 11.]]).unwrap(),
                        increment: constant(increment),
                        lower: constant(-100.),
                        upper: constant(100.),
                        waiting: 0,
                    })
                    .collect(),
                mode: PairedResourceMode::CostUnbounded,
            }])
            .unwrap();
    }
    let root = mp::solve(problem, options()).unwrap();
    optimal(&root, 1.);
    let partial = root
        .snapshot
        .enumerate(&mp::EnumerationOptions {
            max_work: 80,
            ..Default::default()
        })
        .unwrap();
    assert!(!partial.enumeration_report().complete);
    assert_eq!(partial.enumeration_report().transferred, 0);
    assert!(partial.enumeration_report().failure.is_some());
    let full = root
        .snapshot
        .enumerate(&mp::EnumerationOptions::default())
        .unwrap();
    assert!(
        full.enumeration_report().complete,
        "{:?}",
        full.enumeration_report()
    );
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    let solved = full.solve(certified.clone()).unwrap();
    optimal(&solved, 1.);
    let tag = solved
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    optimal(
        &solved
            .snapshot
            .fix_route(tag, 1.)
            .unwrap()
            .solve(certified)
            .unwrap(),
        1.,
    );
    #[cfg(feature = "highs")]
    assert_eq!(
        full.solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        1.
    );
}

#[test]
fn signed_coupled_costs_survive_full_transfer_certificates_fixing_and_siblings() {
    // Independent route objectives: network 7: 10-9=1, 0+5=5;
    // network 42: 2+0=2, 0+4=4. The unit cover has LP/IP optimum one.
    let root = mp::solve(problem(), options()).unwrap();
    optimal(&root, 1.);
    assert!(root.oracle_calls.iter().all(|&(_, calls)| calls > 0));
    let full = root
        .snapshot
        .enumerate(&mp::EnumerationOptions::default())
        .unwrap();
    assert!(full.enumeration_report().complete);
    assert_eq!(full.enumeration_report().routes, 4);
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    let solved = full.solve(certified.clone()).unwrap();
    optimal(&solved, 1.);
    let best = solved
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    assert_eq!(solved.routes[best].route.original_cost, 1.);
    let fixed = solved.snapshot.fix_route(best, 1.).unwrap();
    optimal(&fixed.solve(certified.clone()).unwrap(), 1.);
    optimal(&full.solve(certified).unwrap(), 1.);
    optimal(&root.snapshot.solve(options()).unwrap(), 1.);
}

#[test]
fn cutoff_keeps_an_expensive_scalar_path_with_a_negative_resource_charge() {
    let root = mp::solve(problem(), options()).unwrap();
    let view = root
        .snapshot
        .enumerate_below(1.5, &mp::EnumerationOptions::default())
        .unwrap();
    assert!(view.report().complete);
    assert_eq!(view.coverage_cutoff(), 1.5);
    // Scalar cost ten exceeds the cutoff. Omitting its -9 resource charge
    // from the proof would incorrectly discard the only improving path.
    assert!(view
        .routes()
        .any(|r| r.network == 7 && r.route.arcs == [0, 1] && r.route.original_cost == 1.));
    assert!(!root.snapshot.enumeration_report().complete);
    #[cfg(feature = "highs")]
    {
        let result = view.solve_mip(&Default::default()).unwrap();
        assert_eq!(
            result.mip.status,
            baldes::master::solvers::mip::MipStatus::Optimal
        );
        assert_eq!(result.mip.solution.unwrap().objective, 1.);
    }
    assert!(root
        .snapshot
        .enumerate_below(
            1.5,
            &mp::EnumerationOptions {
                max_work: 1,
                ..Default::default()
            },
        )
        .is_err());
    optimal(&root.snapshot.solve(options()).unwrap(), 1.);
}
