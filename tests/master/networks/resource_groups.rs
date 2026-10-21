//! Shared grouped-resource branches preserve schedules and physical path caps.
use super::*;
use multi_pricing::{EnumerationOptions, ResourceGroupMember};

fn fixture(waiting: bool) -> (Problem<5>, Vec<ResourceGroupMember>) {
    let mut networks = vec![];
    let mut members = vec![];
    for (id, costs, extra) in [(7, [1., 4.], false), (42, [2., 3.], true)] {
        let mut graph = Graph::<5>::new(if extra { 5 } else { 4 }, 5).unwrap();
        let sink = graph.sink;
        for v in 0..graph.vertices.len() {
            for r in 0..5 {
                graph
                    .set_bounds(v, r, 0., if id == 7 { 3. } else { 8. })
                    .unwrap();
            }
        }
        let mut arc_entries = vec![];
        let mut arcs = vec![];
        for (v, cost, q) in [(1, costs[0], 0.5), (2, costs[1], 2.5)] {
            let arc = graph.add_arc(0, v, cost).unwrap();
            graph.arc_mut(arc).unwrap().consumption[4] = q;
            if !waiting {
                graph.set_bounds(v, 4, 0., q).unwrap();
            }
            graph.add_arc(v, sink, 0.).unwrap();
            arcs.push(arc);
            arc_entries.extend([vec![(0, 1.)], vec![]]);
        }
        if extra {
            graph.add_arc(0, 3, 1.).unwrap();
            graph.add_arc(3, sink, 0.).unwrap();
            arc_entries.extend([vec![(1, 1.)], vec![]]);
        }
        members.push(ResourceGroupMember { network: id, arcs });
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(0., 3.).unwrap(),
            arc_entries,
            route_entries: vec![],
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
        members,
    )
}

#[test]
fn shared_group_children_refresh_schedules_and_survive_enumeration_and_restoration() {
    for waiting in [false, true] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                for complete_parent in [false, true] {
                    let (problem, members) = fixture(waiting);
                    let mut configured = options();
                    configured.cg.bound_mode = mode;
                    configured
                        .pricing_options
                        .push(("BALDESuseBidirectionalSearch".into(), direction));
                    if complete_parent {
                        configured.enumeration = Some(EnumerationOptions::default());
                    }
                    let parent = multi_pricing::solve(problem, configured.clone()).unwrap();
                    optimal(&parent, 2.);
                    for upper in [false, true] {
                        let child = parent
                            .snapshot
                            .with_resource_group_branch(0, 4, 1.5, upper, &members)
                            .unwrap();
                        assert_eq!(child.resource_group_branches().count(), 2);
                        let expected = if upper && !waiting { 4. } else { 2. };
                        let result = child.solve(configured.clone()).unwrap();
                        optimal(&result, expected);
                        for (route, &q) in result.routes.iter().zip(&result.values) {
                            if q <= 1e-6 {
                                continue;
                            }
                            let m = members.iter().find(|m| m.network == route.network).unwrap();
                            for (i, a) in route.route.arcs.iter().enumerate() {
                                if m.arcs.contains(a) {
                                    let x = route.route.resource_profile[i + 1][4];
                                    assert!(if upper {
                                        x >= 1.5 - 1e-6
                                    } else {
                                        x <= 1.5 - 1e-6
                                    });
                                }
                            }
                        }
                        let all = result
                            .snapshot
                            .enumerate(&EnumerationOptions::default())
                            .unwrap();
                        optimal(&all.solve(configured.clone()).unwrap(), expected);
                        #[cfg(feature = "highs")]
                        assert_eq!(
                            all.solve_enumerated_mip(&Default::default())
                                .unwrap()
                                .unwrap()
                                .mip
                                .solution
                                .unwrap()
                                .objective,
                            expected
                        );
                    }
                    assert_eq!(parent.snapshot.resource_group_branches().count(), 0);
                    optimal(&parent.snapshot.solve(configured).unwrap(), 2.);
                }
            }
        }
    }
}

#[test]
fn a_complete_parent_keeps_a_capped_path_when_the_branch_changes_its_schedule() {
    let (problem, members) = fixture(true);
    let parent = multi_pricing::solve(problem, options()).unwrap();
    let capped = parent
        .snapshot
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.25,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![1., 0., 0., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    let complete = capped.enumerate(&EnumerationOptions::default()).unwrap();
    let solved = complete.solve(options()).unwrap();
    optimal(&solved, 2.75);
    let tag = solved
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    assert!(solved
        .snapshot
        .preprocessing_report()
        .route_upper_bounds
        .iter()
        .any(|&(i, u)| i == tag && u == 0.25));
    let branch = solved
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap();
    let result = branch.solve(options()).unwrap();
    optimal(&result, 2.75);
    assert_eq!(result.routes[tag].route.resource_profile[1][4], 1.5 - 1e-6);
    assert!((result.values[tag] - 0.25).abs() < 1e-8);
    optimal(&solved.snapshot.solve(options()).unwrap(), 2.75);
    optimal(&parent.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn shared_group_fixed_copies_and_infeasible_descendants_are_isolated() {
    let (problem, members) = fixture(false);
    let parent = multi_pricing::solve(problem, options()).unwrap();
    let cheap = parent
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    let unrelated = parent
        .routes
        .iter()
        .position(|r| r.network == 42 && r.route.arcs == [4, 5])
        .unwrap();
    let fixed = parent.snapshot.fix_route(cheap, 1.).unwrap();
    let lower = fixed
        .with_resource_group_branch(0, 4, 1.5, false, &members)
        .unwrap();
    optimal(&lower.solve(options()).unwrap(), 2.);
    let branch_first = parent
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, false, &members)
        .unwrap()
        .fix_route(cheap, 1.)
        .unwrap();
    optimal(&branch_first.solve(options()).unwrap(), 2.);
    let upper = parent
        .snapshot
        .fix_route(unrelated, 1.)
        .unwrap()
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap();
    optimal(&upper.solve(options()).unwrap(), 4.);
    let incompatible = fixed
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap()
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    #[cfg(feature = "highs")]
    assert_eq!(
        incompatible
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .status,
        baldes::master::solvers::mip::MipStatus::Infeasible
    );
    let impossible = lower
        .with_resource_group_branch(0, 4, 3., true, &members)
        .unwrap()
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    #[cfg(feature = "highs")]
    assert_eq!(
        impossible
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .status,
        baldes::master::solvers::mip::MipStatus::Infeasible
    );
    optimal(&fixed.solve(options()).unwrap(), 2.);
    optimal(&parent.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn shared_group_proof_rejects_cancellation_and_invalid_support_without_mutation() {
    let (problem, members) = fixture(false);
    let parent = multi_pricing::solve(problem.clone(), options()).unwrap();
    let mut duplicated = members.clone();
    duplicated.push(members[0].clone());
    assert!(parent
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &duplicated)
        .is_err());
    assert!(parent
        .snapshot
        .with_resource_group_branch(0, 5, 1.5, true, &members)
        .is_err());
    let mut unknown = members.clone();
    unknown[0].arcs.push(999);
    assert!(parent
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &unknown)
        .is_err());
    let mut too_small = problem.clone();
    too_small.networks[0].arc_entries[0][0].1 = 0.5;
    let mut no_work = options();
    no_work.cg.time_limit = std::time::Duration::ZERO;
    let node = multi_pricing::solve(too_small, no_work.clone()).unwrap();
    assert!(node
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .is_err());
    let mut cancelling = problem;
    cancelling.networks[1].arc_entries[4].push((0, -1.));
    let node = multi_pricing::solve(cancelling, no_work).unwrap();
    assert!(node
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .is_err());
    assert_eq!(parent.snapshot.resource_group_branches().count(), 0);
    optimal(&parent.snapshot.solve(options()).unwrap(), 2.);
}

#[cfg(feature = "highs")]
#[test]
fn shared_group_cutoff_views_keep_the_restricted_domain() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    for waiting in [false, true] {
        let (problem, members) = fixture(waiting);
        let parent = multi_pricing::solve(problem, options()).unwrap();
        let child = parent
            .snapshot
            .with_resource_group_branch(0, 4, 1.5, true, &members)
            .unwrap()
            .solve(options())
            .unwrap();
        let expected = if waiting { 2. } else { 4. };
        let view = child
            .snapshot
            .enumerate_below(expected + 1., &EnumerationOptions::default())
            .unwrap();
        assert_eq!(
            view.solve_mip(&MipOptions::default())
                .unwrap()
                .mip
                .solution
                .unwrap()
                .objective,
            expected
        );
        assert_eq!(
            view.solve_mip(&MipOptions {
                objective_cutoff: expected,
                ..Default::default()
            })
            .unwrap()
            .mip
            .status,
            MipStatus::ObjectiveCutoff
        );
        optimal(&parent.snapshot.solve(options()).unwrap(), 2.);
    }
}

#[test]
fn exact_feasible_but_unreconstructible_branch_leaves_parent_unchanged() {
    let mut graph = Graph::<5>::new(5, 5).unwrap();
    graph.resources[4] = baldes::graph::ResourceKind::Equality;
    for v in 0..5 {
        for r in 0..5 {
            graph
                .set_bounds(v, r, 0., if r == 4 { 1e16 + 2. } else { 3. })
                .unwrap();
        }
    }
    for (tail, head, q) in [(0, 1, 1e16), (1, 2, 1.), (2, 3, 1.), (3, 4, 0.)] {
        let arc = graph
            .add_arc(tail, head, if tail == 0 { 1. } else { 0. })
            .unwrap();
        graph.arc_mut(arc).unwrap().consumption[4] = q;
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![], vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
        }],
    };
    let parent = multi_pricing::solve(problem, options()).unwrap();
    optimal(&parent, 1.);
    let before = parent
        .routes
        .iter()
        .map(|r| r.route.resource_profile.clone())
        .collect::<Vec<_>>();
    let error = parent
        .snapshot
        .with_resource_group_branch(
            0,
            4,
            1e16 + 2.,
            true,
            &[ResourceGroupMember {
                network: 7,
                arcs: vec![2],
            }],
        )
        .unwrap_err();
    assert!(error.to_string().contains("canonical schedule"), "{error}");
    assert_eq!(parent.snapshot.resource_group_branches().count(), 0);
    let resumed = parent.snapshot.solve(options()).unwrap();
    optimal(&resumed, 1.);
    assert_eq!(
        resumed
            .routes
            .iter()
            .map(|r| r.route.resource_profile.clone())
            .collect::<Vec<_>>(),
        before
    );
}

#[test]
fn resource_load_rows_keep_original_path_semantics_when_a_branch_adds_waiting() {
    use baldes::cuts::knapsack::{KnapsackKind, RouteLoadKnapsackCut};
    use multi_pricing::ResourceKnapsackCoefficients;
    let (problem, members) = fixture(true);
    let parent = multi_pricing::solve(problem, options()).unwrap();
    let coefficients: Vec<_> = [7, 42]
        .into_iter()
        .map(|network| ResourceKnapsackCoefficients {
            network,
            resource: 4,
            scale: 2.,
            cut: RouteLoadKnapsackCut {
                kind: KnapsackKind::Steps,
                numerator: 1,
                denominator: 1,
                rhs: 0,
                lhs: 0.,
                violation: 0.,
                thresholds: vec![2],
                coefficients: std::collections::BTreeMap::from([(2, 1)]),
            },
        })
        .collect();
    let row = Row {
        sense: RowSense::LessEqual,
        rhs: 0.,
    };
    let cut = parent
        .snapshot
        .with_resource_knapsack_row(row.clone(), &coefficients)
        .unwrap();
    assert!(cut
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .is_err());
    let branch = parent
        .snapshot
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap();
    assert!(branch
        .with_resource_knapsack_row(row.clone(), &coefficients)
        .is_err());
    let complete = cut.enumerate(&EnumerationOptions::default()).unwrap();
    let branch = complete
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap();
    let solved = branch.solve(options()).unwrap();
    optimal(&solved, 2.);
    let reverse = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .with_resource_group_branch(0, 4, 1.5, true, &members)
        .unwrap()
        .with_resource_knapsack_row(row, &coefficients)
        .unwrap();
    optimal(&reverse.solve(options()).unwrap(), 2.);
    #[cfg(feature = "highs")]
    assert_eq!(
        solved
            .snapshot
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        2.
    );
    optimal(&parent.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn signed_static_terms_cannot_cancel_the_shared_occurrence_proof() {
    use baldes::master::preprocessing::Bounds;
    use multi_pricing::MasterVariable;
    for (coefficient, bounds) in [
        (
            1.,
            Bounds {
                lower: -1.,
                upper: 0.,
            },
        ),
        (
            -1.,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        ),
    ] {
        let (problem, members) = fixture(false);
        let mut configured = options();
        configured.cg.time_limit = std::time::Duration::ZERO;
        let node = multi_pricing::solve_with_master_variables(
            problem,
            vec![MasterVariable {
                id: 99,
                cost: 100.,
                bounds,
                integer: true,
                entries: vec![(0, coefficient)],
            }],
            configured,
        )
        .unwrap();
        assert!(node
            .snapshot
            .with_resource_group_branch(0, 4, 1.5, true, &members)
            .is_err());
        assert_eq!(node.snapshot.resource_group_branches().count(), 0);
    }
}

#[test]
fn equality_resources_filter_without_introducing_waiting() {
    let (mut problem, members) = fixture(true);
    for n in &mut problem.networks {
        n.graph.resources[4] = baldes::graph::ResourceKind::Equality;
    }
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let parent = multi_pricing::solve(problem.clone(), configured.clone()).unwrap();
        optimal(&parent, 2.);
        for upper in [false, true] {
            let child = parent
                .snapshot
                .with_resource_group_branch(0, 4, 1.5, upper, &members)
                .unwrap();
            optimal(
                &child.solve(configured.clone()).unwrap(),
                if upper { 4. } else { 2. },
            );
        }
    }
}

#[test]
fn public_triangle_group_mapping_survives_actual_shared_tree_descendants() {
    let mut networks = vec![];
    let mut members = vec![];
    for (id, base) in [
        (7, vec![vec![0, 1], vec![1, 2], vec![0], vec![1], vec![2]]),
        (42, vec![vec![0, 2], vec![0], vec![1], vec![2]]),
    ] {
        let routes: Vec<_> = (0..3)
            .flat_map(|t| {
                base.iter()
                    .map(move |r| r.iter().map(|v| v + 3 * t).collect::<Vec<_>>())
            })
            .collect();
        let sink = routes.iter().map(Vec::len).sum::<usize>() + 1;
        let mut graph = Graph::<5>::new(sink + 1, 5).unwrap();
        for v in 0..=sink {
            for r in 0..5 {
                graph.set_bounds(v, r, 0., 2.).unwrap();
            }
        }
        let mut vertex = 1;
        let mut entries = vec![];
        let mut arcs = vec![];
        for route in routes {
            let mut tail = 0;
            for (step, customer) in route.into_iter().enumerate() {
                let arc = graph
                    .add_arc(tail, vertex, if step == 0 { 1. } else { 0. })
                    .unwrap();
                graph.arc_mut(arc).unwrap().consumption[4] = 1.;
                entries.push(vec![(customer, 1.)]);
                if customer == 1 {
                    arcs.push(arc);
                }
                tail = vertex;
                vertex += 1;
            }
            graph.add_arc(tail, sink, 0.).unwrap();
            entries.push(vec![]);
        }
        members.push(ResourceGroupMember { network: id, arcs });
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(0., 9.).unwrap(),
            arc_entries: entries,
            route_entries: vec![],
        });
    }
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            9
        ],
        networks,
    };
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let parent = multi_pricing::solve(problem.clone(), configured.clone()).unwrap();
        optimal(&parent, 4.5);
        for upper in [false, true] {
            let child = parent
                .snapshot
                .with_resource_group_branch(1, 4, 2., upper, &members)
                .unwrap();
            optimal(&child.solve(configured.clone()).unwrap(), 5.);
            let tree = child
                .solve_assignment_tree(multi_pricing::AssignmentTreeOptions {
                    node: configured.clone(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(
                tree.status,
                baldes::master::tree::TreeStatus::Optimal,
                "{tree:?}"
            );
            assert_eq!(tree.incumbent, 6.);
            assert_eq!(tree.bound, 6.);
            assert!(tree.nodes > 1, "must execute descendant branches");
            for fixed in tree.routes {
                let member = members
                    .iter()
                    .find(|m| m.network == fixed.route.network)
                    .unwrap();
                for (i, a) in fixed.route.route.arcs.iter().enumerate() {
                    if member.arcs.contains(a) {
                        let q = fixed.route.route.resource_profile[i + 1][4];
                        assert!(if upper {
                            q >= 2. - 1e-6
                        } else {
                            q <= 2. - 1e-6
                        });
                    }
                }
            }
        }
        optimal(&parent.snapshot.solve(configured).unwrap(), 4.5);
    }
}

#[test]
fn sparse_parallel_member_ids_survive_blocking_and_domain_reindexing() {
    let (mut problem, mut members) = fixture(false);
    let mut graph = Graph::<5>::new(3, 5).unwrap();
    for v in 0..3 {
        for r in 0..5 {
            graph.set_bounds(v, r, 0., 3.).unwrap();
        }
    }
    for (id, q, cost) in [(10, 0.5, 1.), (24, 2.5, 4.)] {
        let arc = graph.add_arc(0, 1, cost).unwrap();
        let a = graph.arc_mut(arc).unwrap();
        a.id = id;
        a.consumption[4] = q;
        a.bounds[4] = Some(Window::new(0., q).unwrap());
    }
    let end = graph.add_arc(1, 2, 0.).unwrap();
    graph.arc_mut(end).unwrap().id = 31;
    problem.networks[0].graph = graph;
    problem.networks[0].arc_entries = vec![vec![(0, 1.)], vec![(0, 1.)], vec![]];
    members[0].arcs = vec![24, 10];
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let parent = multi_pricing::solve(problem.clone(), configured.clone()).unwrap();
        optimal(&parent, 2.);
        for upper in [false, true] {
            let child = parent
                .snapshot
                .with_resource_group_branch(0, 4, 1.5, upper, &members)
                .unwrap();
            let result = child.solve(configured.clone()).unwrap();
            optimal(&result, if upper { 4. } else { 2. });
            let complete = result
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap();
            optimal(
                &complete.solve(configured.clone()).unwrap(),
                if upper { 4. } else { 2. },
            );
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
                if upper { 4. } else { 2. }
            );
        }
        optimal(&parent.snapshot.solve(configured).unwrap(), 2.);
    }
}

#[test]
fn rounded_compatible_profiles_cannot_hide_an_exact_fixed_resource_contradiction() {
    let mut graph = Graph::<5>::new(4, 5).unwrap();
    graph.resources[4] = baldes::graph::ResourceKind::Equality;
    for v in 0..4 {
        for r in 0..5 {
            graph
                .set_bounds(v, r, 0., if r == 4 { 1e16 + 4. } else { 3. })
                .unwrap();
        }
    }
    for (tail, head, consumption) in [(0, 1, 1e16), (1, 2, 3.), (2, 3, 0.)] {
        let arc = graph
            .add_arc(tail, head, if tail == 0 { 1. } else { 0. })
            .unwrap();
        graph.arc_mut(arc).unwrap().consumption[4] = consumption;
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
        }],
    };
    let parent = multi_pricing::solve(problem, options()).unwrap();
    optimal(&parent, 1.);
    let tag = parent
        .routes
        .iter()
        .position(|r| r.route.arcs == [0, 1, 2])
        .unwrap();
    assert_eq!(parent.routes[tag].route.resource_profile[2][4], 1e16 + 4.);
    let members = [ResourceGroupMember {
        network: 7,
        arcs: vec![1],
    }];
    for fixed in [false, true] {
        let source = if fixed {
            parent.snapshot.fix_route(tag, 1.).unwrap()
        } else {
            parent
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap()
        };
        let child = source
            .with_resource_group_branch(0, 4, 1e16 + 4., true, &members)
            .unwrap()
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        #[cfg(feature = "highs")]
        assert_eq!(
            child
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap()
                .mip
                .status,
            baldes::master::solvers::mip::MipStatus::Infeasible
        );
    }
    optimal(&parent.snapshot.solve(options()).unwrap(), 1.);
}
