//! Fractional schedules are checked against the equivalent integer model.
use super::*;
use baldes::cuts::knapsack::{KnapsackKind, RouteLoadKnapsackCut};
use mp::{EnumerationOptions, ResourceKnapsackCoefficients};

fn fixture(divisor: f64, variant: &str) -> (Problem<2>, Vec<ResourceKnapsackCoefficients>) {
    let mut networks = vec![
        network(7, &[(1., &[0]), (3., &[0])], 2),
        network(42, &[(1., &[1]), (3., &[1])], 2),
    ];
    let mut coefficients = vec![];
    for (i, n) in networks.iter_mut().enumerate() {
        for vertex in &mut n.graph.vertices {
            vertex.bounds[0].lower /= divisor;
            vertex.bounds[0].upper /= divisor;
        }
        n.graph.arcs[0].consumption[0] = 2. / divisor;
        n.graph.arcs[2].consumption[0] = 1. / divisor;
        let mut threshold = if i == 0 { 3 } else { 2 };
        if variant != "plain" {
            n.graph.initial[0] = Some(if i == 0 { 1. / divisor } else { 2. / divisor });
            threshold = if i == 0 { 5 } else { 4 };
        }
        if variant == "vertex_wait" {
            n.graph.vertices[1].bounds[0] = Window::new(5. / divisor, 10. / divisor).unwrap();
            threshold = if i == 0 { 9 } else { 5 };
        } else if variant == "arc_wait" {
            n.graph.arcs[0].bounds[0] = Some(Window::new(5. / divisor, 10. / divisor).unwrap());
            threshold = if i == 0 { 9 } else { 5 };
        } else if variant == "equality" {
            n.graph.resources[0] = baldes::graph::ResourceKind::Equality;
        }
        coefficients.push(ResourceKnapsackCoefficients {
            network: n.id,
            resource: 0,
            scale: divisor * if i == 0 { 2. } else { 1. },
            cut: RouteLoadKnapsackCut {
                kind: KnapsackKind::Steps,
                numerator: 1,
                denominator: 1,
                rhs: 1,
                lhs: 0.,
                violation: 0.,
                thresholds: vec![threshold],
                coefficients: std::collections::BTreeMap::from([(threshold, 1)]),
            },
        });
    }
    (
        Problem {
            rows: vec![
                Row {
                    sense: RowSense::Equal,
                    rhs: 1.
                };
                2
            ],
            networks,
        },
        coefficients,
    )
}

#[test]
fn fractional_resource_steps_match_integer_schedules_through_fixing_enumeration_and_mip() {
    for divisor in [2., 4., 8., 262_144.] {
        for variant in ["plain", "initial", "vertex_wait", "arc_wait", "equality"] {
            let (problem, coefficients) = fixture(divisor, variant);
            let mut forward = options();
            forward
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".into(), 0.));
            let root = mp::solve(problem, forward).unwrap();
            optimal(&root, 2.);
            let child = root
                .snapshot
                .with_resource_knapsack_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 1.,
                    },
                    &coefficients,
                )
                .unwrap();
            let mut backward = options();
            backward
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".into(), 1.));
            optimal(&child.solve(backward).unwrap(), 4.);
            let route = root
                .routes
                .iter()
                .position(|r| r.network == 7 && r.route.arcs == [0, 1])
                .unwrap();
            let fixed = child.fix_route(route, 1.).unwrap();
            optimal(&fixed.solve(options()).unwrap(), 4.);
            let reordered = root
                .snapshot
                .fix_route(route, 1.)
                .unwrap()
                .with_resource_knapsack_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 1.,
                    },
                    &coefficients,
                )
                .unwrap();
            optimal(&reordered.solve(options()).unwrap(), 4.);
            let complete = fixed.enumerate(&EnumerationOptions::default()).unwrap();
            let mut certified = options();
            certified.cg.bound_mode = BoundMode::Certified;
            optimal(&complete.solve(certified).unwrap(), 4.);
            #[cfg(feature = "highs")]
            assert_eq!(
                complete
                    .solve_enumerated_mip(&Default::default())
                    .unwrap()
                    .unwrap()
                    .mip
                    .solution
                    .unwrap()
                    .objective,
                4.,
                "{divisor} {variant}"
            );
            optimal(&root.snapshot.solve(options()).unwrap(), 2.);
        }
    }
}

#[test]
fn fractional_resource_steps_propagate_original_zero_penalties_without_enumeration() {
    for divisor in [2., 4., 8., 262_144.] {
        for variant in ["plain", "initial", "vertex_wait", "arc_wait", "equality"] {
            let (problem, coefficients) = fixture(divisor, variant);
            let mut forward = options();
            forward
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".into(), 0.));
            let root = mp::solve(problem, forward).unwrap();
            let zero = root
                .snapshot
                .with_resource_knapsack_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 0.,
                    },
                    &coefficients,
                )
                .unwrap()
                .preprocess()
                .unwrap();
            assert_eq!(
                zero.preprocessing_report().forbidden_arcs,
                4,
                "{divisor} {variant}"
            );
            let mut certified = options();
            certified.cg.bound_mode = BoundMode::Certified;
            optimal(&zero.solve(certified).unwrap(), 6.);
            assert!(!zero.enumeration_report().complete);
            optimal(&root.snapshot.solve(options()).unwrap(), 2.);
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn cutoff_resource_rows_preserve_fractional_waiting_equality_and_fixing() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    for divisor in [2., 262_144.] {
        for variant in ["plain", "initial", "vertex_wait", "arc_wait", "equality"] {
            let (problem, coefficients) = fixture(divisor, variant);
            let root = mp::solve(problem, options()).unwrap();
            let child = root
                .snapshot
                .with_resource_knapsack_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 1.,
                    },
                    &coefficients,
                )
                .unwrap();
            let solved = child.solve(options()).unwrap();
            optimal(&solved, 4.);
            let cut = solved
                .snapshot
                .enumerate_below(5., &EnumerationOptions::default())
                .unwrap();
            assert_eq!(
                cut.solve_mip(&MipOptions::default())
                    .unwrap()
                    .mip
                    .solution
                    .unwrap()
                    .objective,
                4.
            );
            assert_eq!(
                cut.solve_mip(&MipOptions {
                    objective_cutoff: 4.,
                    ..Default::default()
                })
                .unwrap()
                .mip
                .status,
                MipStatus::ObjectiveCutoff
            );
            let cheap = root
                .routes
                .iter()
                .position(|r| r.network == 7 && r.route.arcs == [0, 1])
                .unwrap();
            let fixed = child
                .fix_route(cheap, 1.)
                .unwrap()
                .solve(options())
                .unwrap();
            let view = fixed
                .snapshot
                .enumerate_below(5., &EnumerationOptions::default())
                .unwrap();
            assert_eq!(
                view.solve_mip(&MipOptions::default())
                    .unwrap()
                    .mip
                    .solution
                    .unwrap()
                    .objective,
                4.
            );
            optimal(&root.snapshot.solve(options()).unwrap(), 2.);
        }
    }
}
