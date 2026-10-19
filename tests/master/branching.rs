use baldes::{
    pricing::{branching::RyanFosterConstraint, Solver},
    Graph,
};

fn solver() -> Solver {
    let mut s = Solver::new(5, 1).unwrap();
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

#[cfg(feature = "clp")]
fn triangle_problem() -> baldes::master::root::RootProblem {
    use baldes::{
        master::root::RootProblem,
        inputs::{Instance, Rounding},
    };
    let instance=Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap()
}

#[cfg(feature = "clp")]
#[test]
fn typed_children_match_fresh_models_and_survive_snapshot_setup() {
    use baldes::master::{
        branching::BranchConstraint as B,
        root::{self, RootOptions},
        RowSense,
    };
    for enumerate in [false, true] {
        let problem = triangle_problem();
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = enumerate;
        options.cg.stabilize = false;
        options.cg.multiplicity = 3.;
        options.incumbent = 100.;
        let parent = root::solve(problem.clone(), options.clone()).unwrap();
        let snapshot = parent.snapshot.as_ref().unwrap();
        let count = snapshot.enumerated_routes();
        for branch in [
            B::RyanFoster(RyanFosterConstraint {
                first: 1,
                second: 2,
                together: false,
            }),
            B::RyanFoster(RyanFosterConstraint {
                first: 1,
                second: 2,
                together: true,
            }),
            B::NumberOfVehicles {
                graph_id: 0,
                sense: RowSense::GreaterEqual,
                rhs: 2.,
            },
            B::PackingSetAssignment {
                vertex: 1,
                graph_id: 0,
                assigned: true,
            },
        ] {
            let child =
                root::inherit_branches(snapshot, &[branch.clone()], options.clone()).unwrap();
            let mut cold = problem.clone();
            cold.branch(branch.clone()).unwrap();
            let fresh = root::solve(cold.clone(), options.clone()).unwrap();
            assert!(
                (child.master.bound - fresh.master.bound).abs() < 1e-6,
                "{branch:?}: child={child:?} fresh={fresh:?}"
            );
            assert_eq!(snapshot.branch_depth(), 0);
            assert_eq!(snapshot.enumerated_routes(), count);
            assert_eq!(child.snapshot.as_ref().unwrap().branch_depth(), 1);
            for (&tag, &x) in child.master.column_tags.iter().zip(&child.values) {
                if tag >= 0 && x > 1e-6 && matches!(branch, B::RyanFoster(_)) {
                    assert_eq!(
                        branch.coefficient(&problem, &child.routes[tag as usize]),
                        0.
                    );
                }
            }
        }
        let mut contradictory = problem.clone();
        contradictory
            .branch(B::PackingSetAssignment {
                vertex: 1,
                graph_id: 0,
                assigned: false,
            })
            .unwrap();
        let forbidden = root::solve(contradictory, options.clone()).unwrap();
        assert!(
            forbidden.preprocessing.infeasible
                || forbidden.master.pruned_by_bound
                || forbidden.master.artificials.iter().any(|&c| forbidden
                    .values
                    .get(c)
                    .copied()
                    .unwrap_or(0.)
                    > 1e-6)
        );
    }
}

#[cfg(feature = "clp")]
#[test]
fn candidate_families_use_network_route_counts_and_pair_weights() {
    use baldes::master::{
        branching::{BranchConstraint as B, BranchingOptions},
        root::{self, RootOptions},
        tree::candidates::candidates,
    };
    let p = triangle_problem();
    let mut options = RootOptions::default();
    options.enumerate = false;
    options.cg.stabilize = false;
    options.cg.multiplicity = 3.;
    let out = root::solve(p.clone(), options).unwrap();
    let native = candidates(&p, &out, &BranchingOptions::default());
    assert_eq!(native.len(), 1);
    assert_eq!(native[0].value, 1.5);
    assert!(matches!(
        native[0].down,
        B::NumberOfVehicles { rhs: 1., .. }
    ));
    let rf = candidates(
        &p,
        &out,
        &BranchingOptions {
            number_of_vehicles: false,
            ..Default::default()
        },
    );
    assert!(!rf.is_empty());
    assert!(rf.iter().all(|c| matches!(c.down, B::RyanFoster(_))));
    assert!(rf.iter().all(|c| (c.value - 0.5).abs() < 1e-6));
    let arc = candidates(&p, &out, &BranchingOptions::arc_flow_only());
    assert!(!arc.is_empty());
    assert!(arc.iter().all(|c| matches!(c.down, B::ArcFlow { .. })));
}

#[test]
fn randomized_branch_pricing_with_cuts_matches_complete_paths() {
    let mut seed = 417u64;
    for case in 0..60 {
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        let n = 3 + next() % 4;
        let mut s = Solver::new(n + 2, 1).unwrap();
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
            s: &Solver,
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
