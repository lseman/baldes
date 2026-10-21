//! Original-domain implications from sums of registered nonlinear kernels.
use super::*;
use baldes::cuts::CutRegistry;
use baldes::master::preprocessing::Bounds;
use multi_pricing::{MasterVariable, RegisteredCutCoefficients};

fn problem() -> Problem<2> {
    let networks = [7, 42]
        .into_iter()
        .enumerate()
        .map(|(row, id)| {
            let mut graph = Graph::<2>::new(6, 1).unwrap();
            for (a, b, cost) in [
                (0, 1, 1.),
                (1, 2, 0.),
                (2, 5, 0.),
                (1, 3, 0.),
                (3, 5, 0.),
                (0, 4, 3.),
                (4, 5, 0.),
            ] {
                graph.add_arc(a, b, cost).unwrap();
            }
            Network {
                id,
                graph,
                multiplicity: Window::new(0., 3.).unwrap(),
                arc_entries: vec![vec![]; 7],
                route_entries: vec![(row, 1.)],
            }
        })
        .collect();
    Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks,
    }
}
fn coefficients(network: i32) -> Vec<RegisteredCutCoefficients> {
    let mut registry = CutRegistry::default();
    registry.add_strong_k_path_cut(6, &[1, 2], 1, 0.).unwrap();
    let mut first = registry.cuts.remove(0);
    first.visit_once = false;
    first.class = baldes::cuts::CutClass::Packing;
    let mut second = first.clone();
    second.weights[2] = 0;
    second.weights[3] = 1;
    vec![
        RegisteredCutCoefficients {
            network,
            cut: first,
        },
        RegisteredCutCoefficients {
            network,
            cut: second,
        },
    ]
}
fn variable(upper: f64) -> MasterVariable {
    MasterVariable {
        id: 99,
        cost: 0.,
        bounds: Bounds { lower: 0., upper },
        integer: false,
        entries: vec![],
    }
}
fn row(rhs: f64) -> Row {
    Row {
        sense: RowSense::LessEqual,
        rhs,
    }
}

#[test]
fn registered_composites_reach_a_cross_network_static_link_fixed_point_without_enumeration() {
    let root = multi_pricing::solve_with_master_variables(
        problem(),
        vec![variable(f64::INFINITY)],
        options(),
    )
    .unwrap();
    optimal(&root, 2.);
    let first = root
        .snapshot
        .with_composite_cut_row(row(0.), &coefficients(7))
        .unwrap();
    let linked = first
        .with_master_row(
            row(0.),
            &[RowCoefficients {
                network: 7,
                arcs: vec![-1., 0., 0., 0., 0., 0., 0.],
                per_route: 0.,
            }],
            &[(99, 1.)],
        )
        .unwrap();
    let chain = linked
        .with_composite_cut_master_row(row(0.), &coefficients(42), &[(99, -1.)])
        .unwrap();
    let propagated = chain.preprocess().unwrap();
    let report = propagated.preprocessing_report();
    assert!(!report.complete_domain);
    assert!(report.propagation.complete && !report.propagation.infeasible);
    assert_eq!(report.forbidden_arcs, 10);
    assert_eq!(
        report.master_bounds,
        vec![(
            99,
            Bounds {
                lower: 0.,
                upper: 0.
            }
        )]
    );
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    let solved = propagated.solve(certified.clone()).unwrap();
    optimal(&solved, 6.);
    assert!(!solved.snapshot.enumeration_report().complete);
    let safe = solved
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [5, 6])
        .unwrap();
    let fixed = solved.snapshot.fix_route(safe, 1.).unwrap();
    optimal(&fixed.solve(certified).unwrap(), 6.);
    #[cfg(feature = "highs")]
    assert_eq!(
        fixed
            .enumerate(&Default::default())
            .unwrap()
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        6.
    );
    for budget in [0, 64, 128, 256, 512, 1024, 2048, 4096, 16384] {
        let partial = chain
            .preprocess_with_options(&multi_pricing::PreprocessingOptions {
                work_limit: budget,
                ..Default::default()
            })
            .unwrap();
        let r = partial.preprocessing_report();
        assert!(r.propagation.work_used <= budget);
        assert!(!r.propagation.infeasible);
        if r.propagation.complete {
            assert_eq!(r.forbidden_arcs, 10);
        }
        let mut disabled = options();
        disabled.preprocessing.enabled = false;
        optimal(&partial.solve(disabled).unwrap(), 6.);
    }
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn signed_static_compensation_prevents_registered_zero_deductions_until_fixed() {
    for (upper, expected) in [(1., 4.), (f64::INFINITY, 2.)] {
        let root =
            multi_pricing::solve_with_master_variables(problem(), vec![variable(upper)], options())
                .unwrap();
        let kernels: Vec<_> = coefficients(7)
            .into_iter()
            .chain(coefficients(42))
            .collect();
        let parent = root
            .snapshot
            .with_composite_cut_master_row(row(0.), &kernels, &[(99, -1.)])
            .unwrap();
        let retained = parent.preprocess().unwrap();
        assert_eq!(retained.preprocessing_report().forbidden_arcs, 0);
        optimal(&retained.solve(options()).unwrap(), expected);
        let zero = parent
            .with_master_variable_bounds(
                99,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap()
            .preprocess()
            .unwrap();
        assert_eq!(zero.preprocessing_report().forbidden_arcs, 10);
        let mut certified = options();
        certified.cg.bound_mode = BoundMode::Certified;
        optimal(&zero.solve(certified).unwrap(), 6.);
        optimal(&parent.solve(options()).unwrap(), expected);
    }
}

#[test]
fn positive_fractional_registered_budget_preserves_fractional_activity() {
    let root = multi_pricing::solve(problem(), options()).unwrap();
    let kernels: Vec<_> = coefficients(7)
        .into_iter()
        .chain(coefficients(42))
        .collect();
    let child = root
        .snapshot
        .with_composite_cut_row(row(0.5), &kernels)
        .unwrap()
        .preprocess()
        .unwrap();
    assert_eq!(child.preprocessing_report().forbidden_arcs, 0);
    optimal(&child.solve(options()).unwrap(), 5.);
    #[cfg(feature = "highs")]
    assert_eq!(
        child
            .enumerate(&Default::default())
            .unwrap()
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        6.
    );
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}
