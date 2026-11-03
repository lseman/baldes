use baldes::pricing::{restrictions::RyanFosterConstraint, Solver};

fn solver() -> Solver<2> {
    let mut s = Solver::<2>::new(5, 1).unwrap();
    for v in 0..5 {
        s.graph.set_bounds(v, 0, 0., 3.).unwrap();
    }
    for (u, v, cost) in [
        (0, 1, -5.),
        (1, 4, 0.),
        (0, 2, -4.),
        (2, 4, 0.),
        (1, 2, -1.),
        (2, 1, -1.),
        (0, 3, -2.),
        (3, 4, 0.),
        (1, 3, 0.),
        (3, 2, 0.),
        (3, 1, 0.),
        (2, 3, 0.),
    ] {
        let id = s.graph.add_arc(u, v, cost).unwrap();
        s.graph.arc_mut(id).unwrap().consumption[0] = if v == 4 { 0. } else { 1. };
    }
    s
}
#[test]
fn ryan_foster_pricing_and_enumeration_obey_both_children() {
    for mode in [0., 1., 2.] {
        for together in [false, true] {
            let mut s = solver();
            s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
            let rule = RyanFosterConstraint {
                first: 1,
                second: 2,
                together,
            };
            s.set_ryan_foster(vec![rule]).unwrap();
            let result = s.price_phase(0, 150, false).unwrap();
            assert!(result.exact);
            assert_eq!(
                result.minimum_reduced_cost,
                if together { -6. } else { -5. },
                "mode={mode} {result:?}"
            );
            assert!(result
                .columns
                .iter()
                .all(|r| rule.satisfied_by(r.nodes.iter().map(|&v| v as usize))));
            let paths = s.enumerate(10.).unwrap();
            assert!(!paths.is_empty());
            assert!(paths
                .iter()
                .all(|r| rule.satisfied_by(r.nodes.iter().map(|&v| v as usize))));
            let mut child = s.fork_for_node();
            child
                .set_ryan_foster(vec![
                    rule,
                    RyanFosterConstraint {
                        together: !together,
                        ..rule
                    },
                ])
                .unwrap();
            assert_eq!(
                child
                    .price_phase(0, 150, false)
                    .unwrap()
                    .minimum_reduced_cost,
                -2.
            );
            assert!(
                s.enumeration.as_ref().unwrap().paths.len
                    > child.enumeration.as_ref().unwrap().paths.len
            );
        }
    }
}

#[test]
fn arc_set_branches_rebuild_coalesced_paths_and_isolate_siblings() {
    use baldes::pricing::restrictions::ArcSetConstraint;
    for mode in [0., 1., 2.] {
        let mut parent = Solver::<2>::new(4, 0).unwrap();
        parent
            .set_option("BALDESuseBidirectionalSearch", mode)
            .unwrap();
        let first = parent.graph.add_arc(0, 1, 0.).unwrap();
        let last = parent.graph.add_arc(1, 3, 0.).unwrap();
        let alternative = parent.graph.add_arc(0, 1, 1.).unwrap();
        assert_eq!(
            parent
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            0.
        );
        assert!(!parent.enumerate(10.).unwrap().is_empty());
        let mut child = parent.fork_for_node();
        let apart = ArcSetConstraint {
            first: vec![first],
            second: vec![last],
            together: false,
        };
        child.set_arc_set_branches(vec![apart.clone()]).unwrap();
        assert!(child.enumeration.is_none());
        assert_eq!(
            child
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            1.
        );
        let paths = child.enumerate(10.).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].arcs, vec![alternative, last]);
        assert!(parent.enumeration.is_some());
        let mut sibling = child.fork_for_node();
        sibling
            .set_arc_set_branches(vec![ArcSetConstraint {
                together: true,
                ..apart
            }])
            .unwrap();
        assert_eq!(
            sibling
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            0.
        );
        let bad = ArcSetConstraint {
            first: vec![999],
            second: vec![last],
            together: false,
        };
        assert!(child.set_arc_set_branches(vec![bad]).is_err());
        assert!(child.enumeration.is_some());
        assert_eq!(
            child
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            1.
        );
        child.set_arc_set_branches(vec![]).unwrap();
        assert_eq!(
            child
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            0.
        );
    }
}

#[cfg(feature = "clp")]
fn triangle_problem() -> baldes::master::model::Problem<2> {
    use baldes::inputs::{Instance, Rounding};
    let instance=Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    baldes::master::model::Problem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.)
        .unwrap()
}

/// A snapshot with no priced column: a restriction applies as on a fresh model.
#[cfg(feature = "clp")]
fn cold(
    problem: baldes::master::model::Problem<2>,
    options: &baldes::master::model::Options,
) -> baldes::master::model::NodeSnapshot<2> {
    let mut cold = options.clone();
    cold.cg.time_limit = std::time::Duration::ZERO;
    baldes::master::model::solve(problem, cold)
        .unwrap()
        .snapshot
}

#[test]
fn randomized_branch_pricing_with_cuts_matches_complete_paths() {
    let mut rng = baldes::rng::Xoroshiro128Plus::from_seed(417u64);
    for case in 0..60 {
        let mut next = || rng.next_u32() as usize;
        let n = 3 + next() % 4;
        let mut s = Solver::<2>::new(n + 2, 1).unwrap();
        for v in 0..n + 2 {
            s.graph.set_bounds(v, 0, 0., n as f64).unwrap();
        }
        for u in 0..n + 1 {
            for v in u + 1..n + 2 {
                if v == u + 1 || next() % 3 != 0 {
                    let id = 20 + 7 * s.graph.arcs.len() as i32;
                    s.graph
                        .add_arc_with_id(id, u, v, (next() % 13) as f64 - 6.)
                        .unwrap();
                    s.graph.arc_mut(id).unwrap().consumption[0] = if v == n + 1 { 0. } else { 1. };
                }
            }
        }
        let c = RyanFosterConstraint {
            first: 1,
            second: n,
            together: case % 2 == 0,
        };
        s.set_ryan_foster(vec![c]).unwrap();
        s.cuts
            .add_strong_k_path_cut(n + 2, &[1, n], 1, -0.5)
            .unwrap();
        s.set_option("BALDESuseBidirectionalSearch", (case % 3) as f64)
            .unwrap();
        fn dfs(
            s: &Solver<2>,
            c: RyanFosterConstraint,
            v: usize,
            vertices: &mut Vec<usize>,
            cost: f64,
            best: &mut f64,
        ) {
            if v == s.graph.sink {
                if c.satisfied_by(vertices.iter().copied()) {
                    let path = s
                        .cuts
                        .route(&s.graph, &vertices[1..vertices.len() - 1])
                        .unwrap();
                    let k = s.cuts.route_coefficient(&s.graph, 0, &path, false).unwrap();
                    *best = best.min(cost + 0.5 * k as f64);
                }
                return;
            }
            for a in s.graph.arcs.iter().filter(|a| a.tail == v) {
                vertices.push(a.head);
                dfs(s, c, a.head, vertices, cost + a.reduced_cost, best);
                vertices.pop();
            }
        }
        let mut expected = 1e12;
        dfs(&s, c, 0, &mut vec![0], 0., &mut expected);
        let result = s.price_phase(0, 150, false).unwrap();
        assert!(result.exact);
        assert_eq!(
            result.minimum_reduced_cost, expected,
            "case={case} {result:?}"
        );
        let before = expected;
        let _ = s.run_fixing_enumeration(0, 0., 100.).unwrap();
        assert_eq!(
            s.price_phase(0, 150, false).unwrap().minimum_reduced_cost,
            before,
            "after enumeration case={case}"
        );
    }
}

#[test]
fn removing_branch_rebuilds_the_unrestricted_pricing_domain() {
    for mode in [0., 1., 2.] {
        let mut s = solver();
        s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
        // Begin with ordinary labels, then activate branch resources on the
        // same prepared engine before returning to unrestricted pricing.
        assert_eq!(
            s.price_phase(0, 150, false).unwrap().minimum_reduced_cost,
            -7.
        );
        let c = RyanFosterConstraint {
            first: 1,
            second: 2,
            together: true,
        };
        s.set_ryan_foster(vec![c]).unwrap();
        assert_eq!(
            s.price_phase(0, 150, false).unwrap().minimum_reduced_cost,
            -6.
        );
        s.run_fixing_enumeration(0, 0., -5.).unwrap();
        s.set_ryan_foster(vec![]).unwrap();
        assert!(s.enumeration.is_none());
        // NG pricing again permits 0->1->2->1->sink, which RF binary resources
        // had excluded by making both endpoints elementary.
        assert_eq!(
            s.price_phase(0, 150, false).unwrap().minimum_reduced_cost,
            -7.
        );
    }
}

#[test]
fn accumulated_resource_windows_cover_schedules_and_restore_pricing() {
    use baldes::graph::ResourceKind;
    use baldes::pricing::restrictions::AccumulatedResourceBranch;
    for mode in [0., 1., 2.] {
        for kind in [ResourceKind::Disposable, ResourceKind::Equality] {
            for upper in [false, true] {
                let mut s = solver();
                s.graph.resources[0] = kind;
                s.graph.elementary = true;
                if kind == ResourceKind::Equality {
                    s.graph.set_bounds(4, 0, 3., 3.).unwrap();
                }
                s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
                let arc = s
                    .graph
                    .arcs
                    .iter()
                    .find(|a| a.tail == 2 && a.head == 1)
                    .unwrap()
                    .id;
                let before = s.price_phase(0, 150, false).unwrap();
                let baseline = s.graph.clone();
                let rule = AccumulatedResourceBranch {
                    arc,
                    resource: 0,
                    threshold: 2.5,
                    upper,
                };
                let index = s.add_accumulated_resource_branch(rule).unwrap();
                assert_eq!(index, 0);
                assert!(s.enumeration.is_none());
                let priced = s.price_phase(0, 150, false).unwrap();
                assert!(priced.exact);
                let enumerated = s
                    .enumerate(100.)
                    .unwrap_or_else(|e| panic!("mode={mode} kind={kind:?} upper={upper}: {e}"));
                assert!(!enumerated.is_empty());
                for route in priced.columns.iter().chain(&enumerated) {
                    for (i, &id) in route.arcs.iter().enumerate() {
                        if id == arc {
                            let q = route.resource_profile[i + 1][0];
                            assert!(rule.satisfied_by(q), "mode={mode} {kind:?} {route:?}");
                        }
                    }
                }
                let expected = enumerated.iter().map(|r| r.cost).fold(1e12, f64::min);
                assert_eq!(
                    priced.minimum_reduced_cost, expected,
                    "mode={mode} {kind:?}"
                );
                let child = s.fork_for_node();
                s.clear_accumulated_resource_branches().unwrap();
                assert_eq!(s.graph, baseline);
                assert_ne!(child.graph, s.graph);
                assert_eq!(
                    s.price_phase(0, 150, false).unwrap().minimum_reduced_cost,
                    before.minimum_reduced_cost
                );
                assert!(s
                    .add_accumulated_resource_branch(AccumulatedResourceBranch {
                        resource: 2,
                        ..rule
                    })
                    .is_err());
                assert!(s
                    .add_accumulated_resource_branch(AccumulatedResourceBranch {
                        threshold: f64::NAN,
                        ..rule
                    })
                    .is_err());
                assert!(s.accumulated_resource_branches().is_empty());
            }
        }
    }
}

#[cfg(feature = "clp")]
#[cfg(feature = "clp")]
#[test]
fn typed_children_match_fresh_models_and_survive_snapshot_setup() {
    use baldes::master::{
        cg::Status,
        model::{self, EnumerationOptions, NodeSnapshot, Options, RyanFosterMember},
        preprocessing::Bounds,
    };
    #[derive(Clone, Copy, Debug)]
    enum Branch {
        Apart,
        Together,
        AtLeastTwoVehicles,
        AssignedCustomer,
    }
    let restrict = |node: &NodeSnapshot<2>, branch| {
        let pair = |together| {
            node.with_ryan_foster(
                &[RyanFosterMember {
                    network: 0,
                    first: 1,
                    second: 2,
                }],
                together,
            )
        };
        match branch {
            Branch::Apart => pair(false),
            Branch::Together => pair(true),
            Branch::AtLeastTwoVehicles => node.with_multiplicity(
                0,
                Bounds {
                    lower: 2.,
                    upper: 3.,
                },
            ),
            // Customer 1 covers shared row 0.
            Branch::AssignedCustomer => node.branch_assignment(0, 0, true),
        }
        .unwrap()
    };
    for enumerate in [false, true] {
        let problem = triangle_problem();
        let mut options = Options::default();
        options.cg.stabilize = false;
        let parent = model::solve(problem.clone(), options.clone()).unwrap();
        let snapshot = if enumerate {
            parent
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap()
        } else {
            parent.snapshot.solve(options.clone()).unwrap().snapshot
        };
        let count = snapshot.enumeration_report().routes;
        for branch in [
            Branch::Apart,
            Branch::Together,
            Branch::AtLeastTwoVehicles,
            Branch::AssignedCustomer,
        ] {
            let child = restrict(&snapshot, branch).solve(options.clone()).unwrap();
            let fresh = restrict(&cold(problem.clone(), &options), branch)
                .solve(options.clone())
                .unwrap();
            assert!(
                (child.master.bound - fresh.master.bound).abs() < 1e-6,
                "{branch:?}: child={child:?} fresh={fresh:?}"
            );
            assert_eq!(snapshot.enumeration_report().routes, count);
            for (route, &x) in child.routes.iter().zip(&child.values) {
                if x <= 1e-6 {
                    continue;
                }
                let nodes = &route.route.nodes;
                let (a, b) = (nodes.contains(&1), nodes.contains(&2));
                match branch {
                    Branch::Apart => assert!(!(a && b)),
                    Branch::Together => assert_eq!(a, b),
                    _ => {}
                }
            }
        }
        let forbidden = cold(problem.clone(), &options)
            .branch_assignment(0, 0, false)
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(forbidden.master.status, Status::Infeasible, "{forbidden:?}");
    }
}

#[cfg(feature = "clp")]
#[test]
fn resource_threshold_branches_match_fresh_and_inherited_masters() {
    use baldes::master::{
        cg::BoundMode,
        model::{self, EnumerationOptions, Options, ResourceGroupMember},
    };
    for (enumerate, mode) in [
        (false, BoundMode::Numerical),
        (true, BoundMode::Numerical),
        (false, BoundMode::Certified),
    ] {
        let p = triangle_problem();
        let graph = &p.networks[0].graph;
        // Arrivals at vertex 1: the arcs its cover row charges.
        let arrivals: Vec<i32> = graph
            .arcs
            .iter()
            .filter(|a| a.head == 1)
            .map(|a| a.id)
            .collect();
        let mut options = Options::default();
        options.cg.bound_mode = mode;
        options.cg.stabilize = false;
        let parent = model::solve(p.clone(), options.clone()).unwrap();
        let snapshot = if enumerate {
            parent
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap()
        } else {
            parent.snapshot.solve(options.clone()).unwrap().snapshot
        };
        for (threshold, upper) in [(1.5, false), (1.5, true), (-1., false)] {
            // Vertex 1's cover row is shared row 0.
            let branch = |node: &model::NodeSnapshot<2>| {
                node.with_resource_group_branch(
                    0,
                    0,
                    threshold,
                    upper,
                    &[ResourceGroupMember {
                        network: 0,
                        arcs: arrivals.clone(),
                    }],
                )
                .unwrap()
            };
            let child = branch(&snapshot).solve(options.clone()).unwrap();
            let fresh = branch(&cold(p.clone(), &options))
                .solve(options.clone())
                .unwrap();
            assert_eq!(
                child.master.status, fresh.master.status,
                "threshold={threshold} upper={upper}"
            );
            assert!(
                child.master.lp == fresh.master.lp
                    || (child.master.lp - fresh.master.lp).abs() < 1e-6,
                "threshold={threshold} upper={upper}: child={} fresh={}",
                child.master.lp,
                fresh.master.lp
            );
            assert_eq!(snapshot.resource_group_branches().count(), 0);
            assert_eq!(child.snapshot.resource_group_branches().count(), 1);
        }
    }
}
#[test]
fn equality_bucket_jumps_cannot_invent_resource_consumption() {
    use baldes::graph::ResourceKind;
    for mode in [0., 1., 2.] {
        let mut s = Solver::<2>::new(3, 1).unwrap();
        s.graph.resources[0] = ResourceKind::Equality;
        for v in 0..3 {
            s.graph.set_bounds(v, 0, 0., 3.).unwrap();
        }
        s.graph.set_bounds(2, 0, 3., 3.).unwrap();
        let arc = s.graph.add_arc(0, 1, -1.).unwrap();
        s.graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        s.graph.add_arc(1, 2, 0.).unwrap();
        s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
        let result = s.price_phase(0, 50, false).unwrap();
        assert!(result.exact);
        assert!(result.columns.is_empty(), "mode={mode}: {result:?}");
        assert!(s.enumerate(100.).unwrap().is_empty());
    }
}

#[test]
fn resource_boundary_belongs_to_both_sides_and_empty_intersections_forbid_only_the_arc() {
    use baldes::pricing::restrictions::AccumulatedResourceBranch;
    for mode in [0., 1., 2.] {
        for upper in [false, true] {
            let mut s = solver();
            s.graph.elementary = true;
            s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
            let arc = s
                .graph
                .arcs
                .iter()
                .find(|a| a.tail == 0 && a.head == 1)
                .unwrap()
                .id;
            let rule = AccumulatedResourceBranch {
                arc,
                resource: 0,
                threshold: 1.,
                upper,
            };
            s.graph.arc_mut(arc).unwrap().consumption[0] = rule.boundary();
            s.add_accumulated_resource_branch(rule).unwrap();
            let result = s.price_phase(0, 100, false).unwrap();
            assert!(result.columns.iter().any(|r| r.arcs.first() == Some(&arc)));
            for route in result
                .columns
                .iter()
                .filter(|r| r.arcs.first() == Some(&arc))
            {
                assert_eq!(route.resource_profile[1][0], rule.boundary());
            }
            // Both an inherited restrictive window and an empty intersection
            // must be undone, in reverse order, without weakening other arcs.
            s.add_accumulated_resource_branch(AccumulatedResourceBranch {
                threshold: if upper { 4. } else { -1. },
                ..rule
            })
            .unwrap();
            assert!(s
                .price_phase(0, 100, false)
                .unwrap()
                .columns
                .iter()
                .all(|r| !r.arcs.contains(&arc)));
            assert!(s
                .enumerate(100.)
                .unwrap()
                .iter()
                .all(|r| !r.arcs.contains(&arc)));
            s.clear_accumulated_resource_branches().unwrap();
            assert!(s
                .price_phase(0, 100, false)
                .unwrap()
                .columns
                .iter()
                .any(|r| r.arcs.first() == Some(&arc)));
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn ryan_foster_equal_trial_scores_advance_like_native() {
    use baldes::cuts::separation::{ArcData, GraphData, Parameters, VertexData};
    use baldes::master::model::{
        format::Model, BranchingRegistration, MappedEvaluationKind, RankOneIncidenceConfig,
        RyanFosterBranching,
    };
    use baldes::master::tree::{
        self, ModelTreeOptions, NetworkBranchingPolicy, StrongBranchingOptions,
    };
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/ryan_foster_strong_root_native.json"
    ))
    .unwrap();
    let case: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/ryan_foster_candidate_model_native.json"
    ))
    .unwrap();
    assert_eq!(capture["exit_code"], 0);
    let native = capture["hits"].as_array().unwrap();
    assert_eq!(native[0]["score"], native[2]["score"]);
    let mut problem = Model::from_json(&case["model"].to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut graphs = vec![];
    for network in &mut problem.networks {
        let graph = &mut network.graph;
        graph.elementary = true;
        let membership: Vec<Vec<usize>> = (0..10)
            .map(|v| {
                if v == 1 {
                    vec![1]
                } else if (3..=8).contains(&v) {
                    vec![v - 1]
                } else {
                    vec![]
                }
            })
            .collect();
        graph
            .configure_ng(8, membership, vec![(0..8).collect(); 10])
            .unwrap();
        for arc in &mut graph.arcs {
            arc.elements = if arc.head == 1 && arc.tail != 4 && arc.tail != 5 {
                vec![0]
            } else {
                vec![]
            };
            arc.memory = (0..8).collect();
        }
        let vertices = (0..10)
            .map(|id| {
                let sets = if id == 1 {
                    vec![1]
                } else if (3..=8).contains(&id) {
                    vec![id - 1]
                } else {
                    vec![]
                };
                VertexData {
                    id,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                }
            })
            .collect();
        let arcs = graph
            .arcs
            .iter()
            .map(|arc| {
                let sets = if arc.head == 1 && arc.tail != 4 && arc.tail != 5 {
                    vec![0]
                } else {
                    vec![]
                };
                ArcData {
                    id: arc.id,
                    tail: arc.tail as i32,
                    head: arc.head as i32,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                }
            })
            .collect();
        graphs.push(GraphData {
            id: network.id,
            source: 0,
            sink: 9,
            elem_set_count: 8,
            packing_set_count: 8,
            covering_set_count: 8,
            vertices,
            arcs,
            ..Default::default()
        });
    }
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.node.preprocessing.enabled = false;
    options.node.separation.rank_one_incidence = Some(RankOneIncidenceConfig {
        rows: (0..8).collect(),
        graphs,
        parameters: Parameters {
            max_num_rows: 5,
            max_num_per_round: 1,
            memory_type: 1,
            arc_memory_type: 2,
            ls_num_iterations: 0,
            violation_tolerance: 1_000_000.,
            ..Default::default()
        },
    });
    options.node_limit = 50;
    options
        .branching_registry
        .register(BranchingRegistration::Policy(
            NetworkBranchingPolicy::Phased,
        ))
        .unwrap();
    options
        .branching_registry
        .register(BranchingRegistration::RyanFoster(RyanFosterBranching::new(
            100.,
        )))
        .unwrap();
    options
        .branching_registry
        .register(BranchingRegistration::StrongBranching(Some(
            StrongBranchingOptions::default(),
        )))
        .unwrap();
    let root = baldes::master::model::solve(problem.clone(), options.node.clone()).unwrap();
    assert_eq!(root.cuts, 0);
    assert!((root.master.lp - 3.7).abs() < 1e-8);
    let result = tree::solve(problem, options).unwrap();
    for expected in &native[..5] {
        let name = expected["name"].as_str().unwrap();
        let mut actual: Vec<_> = result
            .mapped_branch_evaluations
            .iter()
            .filter(|e| {
                e.depth == 0
                    && e.stage == 1
                    && e.kind == MappedEvaluationKind::RestrictedMaster
                    && e.name == name
            })
            .map(|e| e.estimate)
            .collect();
        let mut bounds: Vec<_> = expected["bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        actual.sort_by(f64::total_cmp);
        bounds.sort_by(f64::total_cmp);
        assert_eq!(actual.len(), 2, "{name}");
        assert!(
            actual.iter().zip(bounds).all(|(a, b)| (a - b).abs() < 1e-8),
            "{name}: {actual:?}"
        );
    }
    let mut next = vec![];
    for evaluation in &result.mapped_branch_evaluations {
        if evaluation.depth == 0
            && evaluation.stage == 2
            && evaluation.kind == MappedEvaluationKind::HeuristicPricing
            && !next.contains(&evaluation.name)
        {
            next.push(evaluation.name.clone());
        }
    }
    let expected: Vec<_> = native[5..8]
        .iter()
        .map(|h| h["name"].as_str().unwrap())
        .collect();
    assert_eq!(next, expected);
}
