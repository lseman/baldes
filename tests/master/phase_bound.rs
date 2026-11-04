#![cfg(feature = "clp")]

use baldes::master::{
    cg::{self, Answer, Column, NoController, Options, Status},
    clp::ClpMaster,
    RowSense,
};

#[test]
fn minimum_cut_rounds_delay_tailing_off_but_do_not_repeat_empty_separation() {
    struct Shrink {
        rounds: usize,
        available: usize,
    }
    impl cg::Controller<ClpMaster> for Shrink {
        fn separates(&self) -> bool {
            true
        }
        fn separate(&mut self, master: &mut ClpMaster, _: &[i32]) -> baldes::Result<usize> {
            if self.rounds >= self.available {
                return Ok(0);
            }
            self.rounds += 1;
            master.add_row(
                RowSense::LessEqual,
                1. - self.rounds as f64 * 0.1,
                &[(0, 1.)],
            )?;
            Ok(1)
        }
    }
    for (minimum, available, expected) in [(0, 5, 1), (3, 5, 3), (3, 1, 1)] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(1., &[(0, 1.)]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        let mut controller = Shrink {
            rounds: 0,
            available,
        };
        let result = cg::optimize(
            &mut master,
            &[],
            |_, _| {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut controller,
            Options {
                stabilize: false,
                min_cut_rounds: minimum,
                max_cut_rounds: 5,
                tailing_off_threshold: 1e6,
                tailing_off_counter: 1,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::Optimal);
        assert_eq!(result.cut_rounds, expected);
        assert_eq!(controller.rounds, expected);
    }
}

#[test]
fn an_exact_phase_cannot_promote_an_incomplete_multi_domain_batch() {
    struct Partial;
    impl cg::Controller<ClpMaster> for Partial {
        fn pricing_complete(&self) -> bool {
            false
        }
        fn pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            Ok(Some(2.))
        }
        fn certified_pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            panic!("partial pricing must not schedule a closing certificate")
        }
    }
    let exact = [baldes::pricing::phases::Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    }];
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        let out = cg::optimize(
            &mut master,
            &exact,
            |_, _| {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut Partial,
            Options {
                stabilize: false,
                max_iterations: 1,
                bound_mode: mode,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.status, Status::Stalled);
        assert!(!out.history[0].exact);
        assert_eq!(out.numerical_bound, f64::NEG_INFINITY);
        assert_eq!(out.certified_bound, f64::NEG_INFINITY);
    }
}

#[test]
fn numerical_mode_skips_certification_and_certified_mode_requests_only_a_closing_pass() {
    use cg::{BoundMode, Controller};
    use std::cell::Cell;
    struct Proof {
        calls: Cell<usize>,
    }
    impl Controller<ClpMaster> for Proof {
        fn certified_pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            self.calls.set(self.calls.get() + 1);
            Ok(Some(1.))
        }
    }
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        let mut proof = Proof {
            calls: Cell::new(0),
        };
        let mut calls = 0;
        let exact = [baldes::pricing::phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }];
        let out = cg::optimize(
            &mut master,
            &exact,
            |_, duals| {
                calls += 1;
                let rc = 1. - duals[0];
                Ok(Answer {
                    columns: if calls == 1 {
                        vec![Column {
                            cost: 1.,
                            reduced_cost: rc,
                            entries: vec![(0, 1.)],
                            tag: 1,
                            identity: "cheaper".into(),
                            acceptable: true,
                        }]
                    } else {
                        vec![]
                    },
                    minimum: rc,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut proof,
            Options {
                bound_mode: mode,
                stabilize: false,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(out.status, Status::Optimal);
        assert_eq!(out.lp, 1.);
        assert_eq!(
            calls, 1,
            "the preceding exact bound closes the next LP without repricing"
        );
        assert_eq!(out.bound_mode, mode);
        assert_eq!(out.numerical_bound, 1.);
        assert_eq!(out.bound, 1.);
        assert_eq!(proof.calls.get(), usize::from(mode == BoundMode::Certified));
        assert_eq!(
            out.certified_bound,
            if mode == BoundMode::Certified {
                1.
            } else {
                f64::NEG_INFINITY
            }
        );
    }
}

#[test]
fn numerical_warm_bounds_never_authorize_certified_pruning() {
    use cg::{BoundMode, Controller};
    struct Unknown;
    impl Controller<ClpMaster> for Unknown {
        fn incumbent(&self) -> f64 {
            1.
        }
        // None explicitly means that no independent certificate is available.
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let parent = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoController,
        Options {
            stabilize: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(parent.bound, 1.);
    assert_eq!(parent.certified_bound, f64::NEG_INFINITY);
    let child = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut Unknown,
        Options {
            bound_mode: BoundMode::Certified,
            stabilize: false,
            warm_start: Some(parent.warm_start),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(child.status, Status::Optimal);
    assert_eq!(child.lp, 1.);
    assert_eq!(child.numerical_bound, 1.);
    assert_eq!(child.bound, f64::NEG_INFINITY);
    assert_eq!(child.certified_bound, f64::NEG_INFINITY);
    assert!(!child.pruned_by_bound);
    assert_eq!(child.warm_start.bound_mode, BoundMode::Certified);
    assert_eq!(child.warm_start.bound, f64::NEG_INFINITY);
}

#[test]
fn root_snapshots_and_tree_modes_do_not_promote_an_unsupported_certificate() {
    use baldes::{
        graph::ResourceKind,
        master::{
            cg::BoundMode,
            model::{self, Problem},
            tree::{self, ModelTreeOptions, TreeStatus},
        },
        Graph,
    };
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.edit().resources[0] = ResourceKind::Equality;
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    graph.set_bounds(2, 0, 0., 0.).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = Problem::set_partitioning(graph, 1.).unwrap();
    let mut options = model::Options::default();
    options.gap_enumeration = false;
    options.cg.stabilize = false;
    let parent = model::solve(problem.clone(), options.clone()).unwrap();
    assert_eq!(parent.master.bound, 2.);
    options.cg.bound_mode = BoundMode::Certified;
    let child = parent.snapshot.solve(options.clone()).unwrap();
    assert_eq!(child.master.lp, parent.master.lp);
    assert_eq!(child.master.numerical_bound, 2.);
    // The network path certifies this equality-resource domain; any
    // certificate must not exceed the only route's cost.
    assert!(child.master.bound <= 2., "{child:?}");
    assert!(!child.master.pruned_by_bound);
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        options.cg.bound_mode = mode;
        let out = tree::solve(
            problem.clone(),
            ModelTreeOptions {
                node: options.clone(),
                diving: None,
                ..ModelTreeOptions::default()
            },
        )
        .unwrap();
        assert_eq!(out.bound_mode, mode);
        assert_eq!(out.incumbent, 2.);
        assert!(out.bound <= 2., "{out:?}");
        if mode == BoundMode::Numerical {
            assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
            assert_eq!(out.bound, 2.);
        } else if out.status == TreeStatus::Optimal {
            // Closing in certified mode requires the supported certificate.
            assert_eq!(out.bound, 2.);
        }
    }
}

#[test]
fn certified_pruning_uses_the_independent_bound_and_preserves_its_mode_in_warm_starts() {
    use cg::{BoundMode, Controller};
    struct Proof;
    impl Controller<ClpMaster> for Proof {
        fn incumbent(&self) -> f64 {
            1.
        }
        fn certified_pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            Ok(Some(1.))
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let out = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: -0.5,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut Proof,
        Options {
            bound_mode: BoundMode::Certified,
            stabilize: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(out.pruned_by_bound);
    assert_eq!(out.bound, 1.);
    assert_eq!(out.certified_bound, 1.);
    assert_eq!(out.numerical_bound, 0.5);
    let resumed = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: -0.5,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoController,
        Options {
            bound_mode: BoundMode::Numerical,
            stabilize: false,
            max_iterations: 1,
            warm_start: Some(out.warm_start),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(resumed.bound_mode, BoundMode::Numerical);
    assert_eq!(resumed.bound, 0.5);
    assert_eq!(resumed.certified_bound, 1.);
}

#[test]
fn original_feasible_columns_can_recover_an_exhausted_penalized_master() {
    use baldes::Result;
    use cg::Controller;
    struct Recovery {
        calls: usize,
    }
    impl Controller<ClpMaster> for Recovery {
        fn restore_feasible_columns(&mut self, duals: &[f64]) -> Result<Vec<Column>> {
            self.calls += 1;
            Ok(vec![
                Column {
                    cost: 16.,
                    reduced_cost: 16. - duals[0],
                    entries: vec![(0, 1.)],
                    acceptable: true,
                    tag: 71,
                    identity: "original-feasible".into(),
                },
                Column {
                    cost: 0.,
                    reduced_cost: -duals[0],
                    entries: vec![(0, 1.)],
                    acceptable: false,
                    tag: 72,
                    identity: "ineligible".into(),
                },
            ])
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    let mut options = Options::default();
    options.stabilize = false;
    options.artificial_cost = 1.;
    options.max_penalty_updates = 1;
    let mut controller = Recovery { calls: 0 };
    let output = cg::optimize(
        &mut master,
        &[],
        |phase, duals| {
            Ok(Answer {
                columns: vec![],
                minimum: 16. - duals[0],
                exact: phase == 0,
                bound_minimum: None,
            })
        },
        &mut controller,
        options,
    )
    .unwrap();
    assert_eq!(output.status, Status::Optimal);
    assert_eq!(output.lp, 16.);
    assert!(output.bound <= 16. && controller.calls > 0);
    assert!(output
        .artificials
        .iter()
        .all(|&i| master.values()[i] <= 1e-6));
    assert!(output.column_tags.contains(&71) && !output.column_tags.contains(&72));
}

#[test]
fn large_original_routes_recover_before_a_rank_cut_certificate_closes_the_root() {
    use baldes::{
        inputs::{Instance, Rounding},
        master::{
            model::{self, Problem},
            tree::{self, ModelTreeOptions},
        },
    };
    let scale = 1_000_000.;
    let diagonal = (2f64 * scale * scale).sqrt();
    let nearest = 4. * scale + diagonal;
    let z = nearest - 4. * scale;
    let error = (4. * scale - (nearest - z)) + (diagonal - z);
    let instance=Instance::parse("large triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100000000 0\n1 1000000 0 1 0 100000000 0\n2 0 1000000 1 0 100000000 0\n3 1000000 1000000 1 0 100000000 0\n").unwrap();
    let mut options = model::Options::default();
    options.separation.rank_one.enabled = true;
    options.cg.bound_mode = baldes::master::cg::BoundMode::Certified;
    // Phase I needs artificial costs on the routes' scale (about 4e6).
    options.cg.artificial_cost = 1e8;
    let problem = Problem::set_partitioning(
        instance.graph(5, Rounding::None).unwrap(),
        (instance.customers.len() - 1) as f64,
    )
    .unwrap();
    let output = model::solve(problem.clone(), options.clone()).unwrap();
    assert_eq!(output.master.status, Status::Optimal, "{:?}", output.master);
    // The exact original optimum is 4*scale plus the supplied floating
    // diagonal. TwoSum's residual decides which side of `nearest` it lies.
    assert!(output.master.bound < nearest || output.master.bound == nearest && error >= 0.);
    assert!(output.cuts > 0);
    let tree = tree::solve(
        problem,
        ModelTreeOptions {
            node: options,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.incumbent > nearest || tree.incumbent == nearest && error <= 0.);
    assert!(tree.incumbent - tree.bound < 1e-6, "{tree:?}");
}

#[test]
fn an_integer_certificate_does_not_terminate_fractional_lp_pricing() {
    use baldes::{pricing::phases::Phase, Result};
    use cg::Controller;
    struct IntegerCover;
    impl Controller<ClpMaster> for IntegerCover {
        fn certified_pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> Result<Option<f64>> {
            // x + 2y + 2z = 1 with integer variables has only x=1,
            // of cost one. Its LP can use z=1/2, of cost one half.
            Ok(Some(1.))
        }
    }
    for warm in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(1., &[(0, 1.)]).unwrap();
        let mut options = Options::default();
        options.bound_mode = cg::BoundMode::Certified;
        options.stabilize = false;
        if warm {
            options.warm_start = Some(cg::WarmStart {
                bound: 1.,
                bound_mode: cg::BoundMode::Certified,
                numerical_bound: f64::NEG_INFINITY,
                certified_bound: 1.,
                stabilization: None,
                proximal: None,
                generated_pool: vec![],
                columns: vec![Some(Column {
                    cost: 1.,
                    reduced_cost: 0.,
                    entries: vec![(0, 1.)],
                    acceptable: true,
                    tag: 0,
                    identity: "integer-singleton".into(),
                })],
                artificials: vec![],
                cut_artificials: vec![],
                initial_rows: 1,
            });
        }
        let mut calls = 0;
        let output = cg::optimize(
            &mut master,
            &[Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }],
            |_, duals| {
                calls += 1;
                let cost = if calls == 1 { 1.5 } else { 1. };
                Ok(Answer {
                    columns: vec![Column {
                        cost,
                        reduced_cost: cost - 2. * duals[0],
                        entries: vec![(0, 2.)],
                        acceptable: true,
                        tag: calls,
                        identity: format!("cost-{cost}"),
                    }],
                    minimum: 1. - 2. * duals[0],
                    exact: true,
                    bound_minimum: Some(1. - 2. * duals[0]),
                })
            },
            &mut IntegerCover,
            options,
        )
        .unwrap();
        assert_eq!(output.status, Status::Optimal);
        assert_eq!(output.lp, 0.5);
        assert_eq!(output.bound, 1.);
        assert!(calls >= 2);
    }
}

#[test]
fn finite_artificial_penalty_exhaustion_does_not_prove_infeasibility() {
    for (updates, expected) in [(0, Status::Stalled), (25, Status::Optimal)] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        let mut options = Options::default();
        options.stabilize = false;
        options.artificial_cost = 1.;
        options.max_penalty_updates = updates;
        let result = cg::optimize(
            &mut master,
            &[],
            |phase, duals| {
                let reduced_cost = 1e6 - duals[0];
                Ok(Answer {
                    columns: vec![Column {
                        cost: 1e6,
                        reduced_cost,
                        entries: vec![(0, 1.)],
                        acceptable: true,
                        tag: 0,
                        identity: "feasible-real-column".into(),
                    }],
                    minimum: reduced_cost,
                    exact: phase == 0,
                    bound_minimum: None,
                })
            },
            &mut NoController,
            options,
        )
        .unwrap();
        // Quantity one of the returned real column is an exact feasible
        // witness regardless of which finite penalty the master prefers.
        assert_eq!(result.status, expected);
        if expected == Status::Optimal {
            assert_eq!(result.lp, 1e6);
        }
    }
}

#[test]
fn heuristic_stage_value_skips_pricing_resets_and_never_certifies_the_global_bound() {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    let mut options = Options::default();
    options.stabilize = false;
    options.artificial_cost = 100.;
    let mut phases = Vec::new();
    let result = cg::optimize(
        &mut master,
        &[],
        |phase, duals| {
            phases.push(phase);
            let (cost, entries) = if phase == 2 {
                let row = if phases.len() == 1 { 0 } else { 1 };
                (if row == 0 { 10. } else { 8. }, vec![(row, 1.)])
            } else {
                (if phase == 1 { 12. } else { 5. }, vec![(0, 1.), (1, 1.)])
            };
            let mut reduced_cost = cost;
            for &(row, coefficient) in &entries {
                reduced_cost -= coefficient * duals[row];
            }
            Ok(Answer {
                columns: vec![Column {
                    cost,
                    reduced_cost,
                    entries,
                    acceptable: true,
                    tag: phase as i32,
                    identity: format!("phase-{phase}-cost-{cost}"),
                }],
                minimum: reduced_cost,
                exact: phase == 0,
                bound_minimum: None,
            })
        },
        &mut NoController,
        options,
    )
    .unwrap();
    assert_eq!(phases, [2, 2, 1, 0]);
    assert_eq!(result.status, Status::Optimal);
    assert!((result.lp - 5.).abs() < 1e-9);
    assert!((result.bound - 5.).abs() < 1e-9);
    assert_eq!(result.pricing_calls, 4);
    assert_eq!(result.iterations, 7);
}

#[test]
fn resource_rounding_retry_does_not_publish_a_root_feasibility_proof() {
    use baldes::{
        master::{
            model::{self, Problem},
            tree::{self, ModelTreeOptions},
        },
        Graph,
    };
    let upper = f64::from_bits(1f64.to_bits() - 1);
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for customer in [1, 2] {
        graph.add_elementarity_set(&[customer]).unwrap();
    }
    graph.edit().initial[0] = Some(0.);
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., upper).unwrap();
        graph.set_ng(v, &[0, 1]).unwrap();
    }
    for (u, v, cost) in [
        (0, 1, 0.7),
        (1, 3, 0.3),
        (0, 2, 0.7),
        (2, 3, 0.3),
        (1, 2, 0.),
        (2, 1, 0.),
    ] {
        let id = graph.add_arc(u, v, cost).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = cost;
    }
    let problem = Problem::set_partitioning(graph, 2.).unwrap();
    let mut options = model::Options::default();
    options.gap_enumeration = false;
    options.cg.max_penalty_updates = 0;
    options.cg.time_limit = std::time::Duration::from_secs(1);
    let output = model::solve(problem.clone(), options.clone()).unwrap();
    // Every route overshoots the window by an ulp in exact arithmetic.
    assert_ne!(output.master.status, Status::Optimal, "{output:?}");
    let tree = tree::solve(
        problem,
        ModelTreeOptions {
            node: options,
            node_limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.incumbent.is_infinite(), "{tree:?}");
}

#[test]
fn rounded_resource_feasibility_does_not_publish_an_integer_incumbent() {
    use baldes::{
        master::{
            model::{self, Problem},
            tree::{self, ModelTreeOptions},
        },
        Graph,
    };
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.edit().initial[0] = Some(0.);
    graph.add_elementarity_set(&[1]).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 1.).unwrap();
        graph.set_ng(v, &[0]).unwrap();
    }
    for (u, v, c) in [(0, 1, 1.), (1, 2, f64::EPSILON / 2.)] {
        let id = graph.add_arc(u, v, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = c;
    }
    let mut options = model::Options::default();
    options.gap_enumeration = false;
    options
        .pricing_options
        .push(("BALDESuseBidirectionalSearch".into(), 0.));
    let problem = Problem::set_partitioning(graph, 1.).unwrap();
    let output = model::solve(problem.clone(), options.clone()).unwrap();
    // The floating labeler accepts the route; exact replay must not.
    assert!(output.routes.iter().any(|r| r.route.arcs == [0, 1]));
    let tree = tree::solve(
        problem,
        ModelTreeOptions {
            node: options,
            node_limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.incumbent.is_infinite(), "{tree:?}");
}

#[test]
fn ipm_master_handles_edits_limits_and_independent_forks() {
    use baldes::master::{
        clp::{ClpMaster, MasterSolver},
        LpMaster, LpSolveOutcome, RowSense,
    };
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Ipm);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    assert!((master.objective - 3.).abs() < 1e-7);
    let first_iterations = master.ipm_iterations;
    assert!(first_iterations > 0);
    assert_eq!(master.simplex_iterations, 0);
    master.add_column(1., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    assert!((master.objective - 1.).abs() < 1e-7);
    assert!(master.ipm_iterations > first_iterations);
    assert!((master.duals[0] - 1.).abs() < 1e-7);
    assert!((master.reduced_costs[0] - 2.).abs() < 1e-7);
    let mut fork = master.try_clone().unwrap();
    assert_eq!(fork.solver(), MasterSolver::Ipm);
    assert_eq!(fork.ipm_iterations, 0);
    fork.set_upper(1, 0.).unwrap();
    assert!(fork.solve());
    assert!((fork.objective - 3.).abs() < 1e-7);
    assert!((master.objective - 1.).abs() < 1e-7);
    assert_eq!(
        fork.solve_with_time_limit(std::time::Duration::ZERO)
            .unwrap(),
        LpSolveOutcome::TimeLimit
    );
    assert!(!fork.solved);
    assert!(fork.solve());
    // An interior point has no simplex basis: cleanup must retain these rows.
    assert!(!fork.row_is_basic(0).unwrap());
}

#[test]
fn ipm_column_generation_tightens_from_lp_bounds_and_finishes_accurately() {
    use baldes::master::clp::MasterSolver;
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Ipm);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let output = cg::optimize(
        &mut master,
        &[],
        |_, duals| {
            let rc = 1. - duals[0];
            Ok(Answer {
                columns: if rc < -1e-6 {
                    vec![Column {
                        cost: 1.,
                        reduced_cost: rc,
                        entries: vec![(0, 1.)],
                        acceptable: true,
                        tag: 1,
                        identity: "cheap".into(),
                    }]
                } else {
                    vec![]
                },
                minimum: rc.min(0.),
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoController,
        Options {
            stabilize: false,
            max_iterations: 20,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.status, Status::Optimal, "{output:?}");
    assert!((output.lp - 1.).abs() < 1e-7, "{output:?}");
    assert_eq!(output.history[0].master_gap_tolerance, Some(0.5));
    let first_upper = output.history[0].lp;
    let expected = ((first_upper - 1.) / (1. + first_upper.abs()) / 10.).clamp(1e-9, 1.);
    assert!((output.history[1].master_gap_tolerance.unwrap() - expected).abs() < 1e-8);
    assert!(output
        .history
        .iter()
        .any(|h| h.master_gap_tolerance.is_some_and(|e| e < 0.5)));
    assert_eq!(output.history.last().unwrap().master_gap_tolerance, None);
    assert!(output.bound <= 1. + 1e-7);
}

#[test]
fn ipm_pricing_accuracy_does_not_leak_into_standalone_solves() {
    use baldes::master::{clp::MasterSolver, LpMaster, LpSolveOutcome};
    use std::time::Duration;
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Ipm);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    for cost in [1., 2., 8e4] {
        master.add_column(cost, &[(0, 1.)]).unwrap();
    }
    assert_eq!(
        master
            .solve_for_column_generation(Duration::from_secs(1), Some(0.5))
            .unwrap(),
        LpSolveOutcome::Suboptimal
    );
    assert!(master.objective > 1. + 1e-8);
    assert!(master.solve());
    assert!((master.objective - 1.).abs() < 1e-7);
    assert_eq!(
        master
            .solve_for_column_generation(Duration::ZERO, Some(0.5))
            .unwrap(),
        LpSolveOutcome::TimeLimit
    );
    assert!(!master.solved);
}

#[test]
fn ipm_cut_round_resets_the_primal_accuracy_bound() {
    use baldes::master::clp::MasterSolver;
    struct CutOnce {
        added: bool,
    }
    impl cg::Controller<ClpMaster> for CutOnce {
        fn separates(&self) -> bool {
            true
        }
        fn separate(&mut self, master: &mut ClpMaster, _: &[i32]) -> baldes::Result<usize> {
            // Separators must receive the accurately solved point.
            let expected = if self.added { 5.5 } else { 1. };
            assert!((master.objective - expected).abs() < 1e-7);
            if self.added {
                return Ok(0);
            }
            master.add_row(RowSense::LessEqual, 0.5, &[(0, 1.)])?;
            self.added = true;
            Ok(1)
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Ipm);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let output = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut CutOnce { added: false },
        Options {
            stabilize: false,
            max_iterations: 20,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.status, Status::Optimal, "{output:?}");
    assert!((output.lp - 5.5).abs() < 1e-7);
    assert_eq!(output.cut_rounds, 1);
    let reset = output
        .history
        .windows(2)
        .find(|pair| pair[1].lp > pair[0].lp + 1.)
        .unwrap();
    assert_eq!(reset[1].master_gap_tolerance, Some(0.5));
    assert_eq!(output.history.last().unwrap().master_gap_tolerance, None);
}

#[test]
fn hgs_seeded_ipm_root_and_tree_match_the_small_routing_optimum() {
    use baldes::inputs::{HgsColumns, HgsWarmStart, Instance, Rounding};
    use baldes::master::{
        clp::MasterSolver,
        model::{self, FixedRoute, NetworkRoute, Problem},
        tree,
    };
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for columns in [HgsColumns::Best, HgsColumns::All, HgsColumns::Top(1)] {
        let graph = instance.graph(1, Rounding::None).unwrap();
        let problem = Problem::set_partitioning(graph.clone(), 3.).unwrap();
        let seed = instance
            .seed_hgs(
                &graph,
                Rounding::None,
                &HgsWarmStart {
                    time_limit: std::time::Duration::from_millis(30),
                    columns,
                    seed: 7,
                },
                false,
                &[],
            )
            .unwrap();
        assert!(seed.summary.inserted_columns > 0);
        if columns == HgsColumns::Top(1) {
            assert_eq!(seed.summary.inserted_columns, 1);
        }
        assert!(seed.incumbent.is_some());
        assert!(seed
            .routes
            .iter()
            .all(|r| r.nodes.first() == Some(&0) && r.nodes.last() == Some(&4)));
        let mut options = model::Options::default();
        options.separation.rank_one.enabled = true;
        options.master_solver = MasterSolver::Ipm;
        options.gap_enumeration = false;
        options.cg.time_limit = std::time::Duration::from_secs(5);
        options.seed_routes = seed
            .routes
            .iter()
            .map(|route| NetworkRoute {
                network: 0,
                route: route.clone(),
            })
            .collect();
        let root = model::solve(problem.clone(), options.clone()).unwrap();
        assert_eq!(root.master.status, baldes::master::cg::Status::Optimal);
        assert!(
            (root.master.lp - (4. + 2_f64.sqrt())).abs() < 1e-6,
            "{}",
            root.master.lp
        );
        // IPM disables dual stabilization.
        assert!(root.master.history.iter().all(|h| h.alpha == 0.));
        let mut resumed = options.clone();
        resumed.seed_routes.clear();
        let child = root.snapshot.solve(resumed).unwrap();
        assert!((child.master.lp - root.master.lp).abs() < 1e-6);
        let mut controls = tree::ModelTreeOptions::default();
        controls.initial_solution =
            seed.incumbent
                .zip(seed.solution.clone())
                .map(|(cost, routes)| {
                    (
                        cost,
                        routes
                            .into_iter()
                            .map(|route| FixedRoute {
                                route: NetworkRoute { network: 0, route },
                                quantity: 1.,
                            })
                            .collect(),
                    )
                });
        controls.node = options;
        controls.time_limit = std::time::Duration::from_secs(5);
        let output = tree::solve(problem, controls).unwrap();
        assert!(
            (output.incumbent - (4. + 2_f64.sqrt())).abs() < 1e-6,
            "{output:?}"
        );
    }
}

#[test]
fn hgs_seeding_rejects_integer_approximation_false_feasibility() {
    use baldes::inputs::{HgsColumns, HgsWarmStart, Instance, Rounding};
    let instance = Instance::parse("one\nVEHICLE\nNUMBER CAPACITY\n1 1\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1.001 0 100 0\n").unwrap();
    let graph = instance.capacity_graph(1, Rounding::None).unwrap();
    let warm = HgsWarmStart {
        time_limit: std::time::Duration::from_millis(10),
        columns: HgsColumns::All,
        seed: 0,
    };
    let seed = instance
        .seed_hgs(&graph, Rounding::None, &warm, false, &[])
        .unwrap();
    assert!(seed.routes.is_empty());
    assert!(seed.incumbent.is_none());
    assert!(seed.solution.is_none());
    assert!(instance
        .seed_hgs(
            &graph,
            Rounding::None,
            &HgsWarmStart {
                time_limit: std::time::Duration::ZERO,
                ..warm
            },
            false,
            &[],
        )
        .is_err());
}

#[test]
fn hgs_capacity_seeds_ignore_file_windows_and_deduplicate_existing_columns() {
    use baldes::inputs::{HgsColumns, HgsWarmStart, Instance, Rounding};
    let instance = Instance::parse("one\nVEHICLE\nNUMBER CAPACITY\n1 1\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1 0\n1 10 0 1 0 1 0\n").unwrap();
    let graph = instance.capacity_graph(1, Rounding::None).unwrap();
    let mut existing = vec![];
    for (mode, expected) in [(HgsColumns::Best, 1), (HgsColumns::All, 0)] {
        let seed = instance
            .seed_hgs(
                &graph,
                Rounding::None,
                &HgsWarmStart {
                    time_limit: std::time::Duration::from_millis(10),
                    columns: mode,
                    seed: 0,
                },
                false,
                &existing,
            )
            .unwrap();
        assert_eq!(seed.summary.inserted_columns, expected);
        assert_eq!(seed.incumbent, Some(20.));
        existing.extend(seed.routes);
    }
    assert_eq!(existing.len(), 1);
}

#[cfg(feature = "highs")]
#[test]
fn highs_routing_master_keeps_solution_basis_and_limits_independent() {
    use baldes::master::{clp::MasterSolver, LpMaster, LpSolveOutcome};
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Highs);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 5., &[]).unwrap();
    master.add_column(3., &[(0, 1.), (1, 1.)]).unwrap();
    master.objective_offset = 4.;
    assert!(master.solve());
    assert!((master.objective - 7.).abs() < 1e-8);
    assert!(master.row_is_basic(1).unwrap());
    master.add_column(1., &[(0, 1.), (1, 1.)]).unwrap();
    assert!(master.solve());
    assert!((master.objective - 5.).abs() < 1e-8);
    assert!((master.duals[0] - 1.).abs() < 1e-8);
    assert!((master.reduced_costs[0] - 2.).abs() < 1e-8);
    let mut fork = master.try_clone().unwrap();
    assert_eq!(fork.solver(), MasterSolver::Highs);
    assert!(fork.row_is_basic(1).unwrap());
    fork.set_upper(1, 0.).unwrap();
    assert!(fork.solve());
    assert!((fork.objective - 7.).abs() < 1e-8);
    assert!((master.objective - 5.).abs() < 1e-8);
    fork.remove_cuts(&[1], &[]).unwrap();
    assert!(fork.row_is_basic(0).is_ok());
    assert!(fork.solve());
    assert_eq!(
        fork.solve_with_time_limit(std::time::Duration::ZERO)
            .unwrap(),
        LpSolveOutcome::TimeLimit
    );
    assert!(!fork.solved);
    assert!(fork.solve());
    fork.set_rhs(&[-1.]).unwrap();
    assert_eq!(
        fork.solve_with_time_limit(std::time::Duration::from_secs(1))
            .unwrap(),
        LpSolveOutcome::PrimalInfeasible
    );
    assert!(!fork.solved);
}

#[cfg(feature = "highs")]
#[test]
fn highs_solves_routing_roots_cuts_and_inherited_tree_nodes() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        clp::MasterSolver,
        model::{self, Problem},
        preprocessing::Bounds,
        tree,
    };
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for stabilize in [false, true] {
        let problem =
            Problem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
        let mut options = model::Options::default();
        options.separation.rank_one.enabled = true;
        options.master_solver = MasterSolver::Highs;
        options.cg.stabilize = stabilize;
        options.cg.time_limit = std::time::Duration::from_secs(5);
        options.gap_enumeration = false;
        let result = model::solve(problem.clone(), options.clone()).unwrap();
        assert_eq!(result.master.status, Status::Optimal);
        assert!(
            (result.master.lp - (4. + 2_f64.sqrt())).abs() < 1e-6,
            "{}",
            result.master.lp
        );
        let inherited = result.snapshot.solve(options.clone()).unwrap();
        assert!((inherited.master.lp - result.master.lp).abs() < 1e-6);
        let branched = result
            .snapshot
            .with_multiplicity(
                0,
                Bounds {
                    lower: 3.,
                    upper: 3.,
                },
            )
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(branched.master.status, Status::Optimal);
        assert!(
            (branched.master.lp - (4. + 2. * 2_f64.sqrt())).abs() < 1e-6,
            "{}",
            branched.master.lp
        );
        let mut controls = tree::ModelTreeOptions::default();
        controls.node = options;
        controls.time_limit = std::time::Duration::from_secs(5);
        let tree = tree::solve(problem, controls).unwrap();
        assert!(
            (tree.incumbent - (4. + 2_f64.sqrt())).abs() < 1e-6,
            "{tree:?}"
        );
    }
}

#[cfg(not(feature = "highs"))]
#[test]
fn unavailable_highs_master_is_rejected_explicitly() {
    use baldes::master::clp::MasterSolver;
    assert!(MasterSolver::Highs
        .ensure_available()
        .unwrap_err()
        .to_string()
        .contains("highs build feature"));
}

#[cfg(feature = "highs")]
#[test]
fn highs_recovers_the_r201_hundred_customer_warm_basis_failure() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{clp::MasterSolver, model};
    let instance = Instance::read("instances/Solomon/R201.txt").unwrap();
    let mut options = model::Options::default();
    options.master_solver = MasterSolver::Highs;
    options.cg.time_limit = std::time::Duration::from_secs(30);
    let result = instance.solve(8, Rounding::None, options).unwrap();
    // The retained HiGHS basis used to return Unknown at master solve 430.
    // Warm reoptimization must complete pricing and agree with the Clp root LP.
    assert_eq!(result.master.status, Status::Optimal);
    assert!(
        (result.master.lp - 1140.130144533555).abs() < 1e-6,
        "{}",
        result.master.lp
    );
}

#[test]
fn closing_exact_bound_keeps_feasible_master_before_empty_enumeration() {
    use cg::Controller;
    struct ClosingBound {
        enumeration_calls: usize,
    }
    impl Controller<ClpMaster> for ClosingBound {
        fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
            Ok(true)
        }
        fn incumbent(&self) -> f64 {
            1.
        }
        fn pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            Ok(Some(1.))
        }
        fn enumerated_universe(
            &mut self,
            _: &[f64],
            _: f64,
        ) -> baldes::Result<Option<Vec<Column>>> {
            self.enumeration_calls += 1;
            // No column can strictly improve the incumbent. Replacing the
            // feasible master with this empty universe loses its LP point.
            Ok((self.enumeration_calls == 1).then(Vec::new))
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let mut controller = ClosingBound {
        enumeration_calls: 0,
    };
    let out = cg::optimize(
        &mut master,
        &[baldes::pricing::phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut controller,
        Options {
            stabilize: false,
            known: vec![Column {
                cost: 1.,
                reduced_cost: 0.,
                entries: vec![(0, 1.)],
                acceptable: true,
                tag: 0,
                identity: "incumbent-column".into(),
            }],
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(out.status, Status::Optimal);
    assert!(out.pruned_by_bound);
    assert_eq!(out.lp, 1.);
    assert_eq!(out.bound, 1.);
    assert_eq!(controller.enumeration_calls, 0);
    assert_eq!(master.values()[0], 1.);
    assert!(out.warm_start.columns.iter().any(Option::is_some));
}

#[test]
fn pricing_timeout_before_checkpoint_preserves_bounds_and_resumable_columns() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        for inherited in [false, true] {
            let mut master = ClpMaster::new().unwrap();
            master.add_row(RowSense::Equal, 1., &[]).unwrap();
            master.add_column(2., &[(0, 1.)]).unwrap();
            let warm = inherited.then(|| cg::WarmStart {
                bound: 1.,
                bound_mode: mode,
                numerical_bound: 1.,
                certified_bound: if mode == cg::BoundMode::Certified {
                    1.
                } else {
                    f64::NEG_INFINITY
                },
                stabilization: None,
                proximal: None,
                generated_pool: vec![],
                columns: vec![Some(Column {
                    cost: 2.,
                    entries: vec![(0, 1.)],
                    reduced_cost: 0.,
                    tag: 7,
                    identity: "existing".into(),
                    acceptable: true,
                })],
                artificials: vec![],
                cut_artificials: vec![],
                initial_rows: 1,
            });
            let out = cg::optimize(
                &mut master,
                &[],
                |_, _| {
                    Err(baldes::core::error::OrError {
                        kind: baldes::core::error::ErrorKind::Runtime,
                        message: "pricing hard time limit exceeded".into(),
                    })
                },
                &mut NoController,
                Options {
                    stabilize: false,
                    bound_mode: mode,
                    warm_start: warm,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(out.status, Status::TimeLimit);
            assert_eq!(out.pricing_calls, 1);
            assert_eq!(out.bound, if inherited { 1. } else { f64::NEG_INFINITY });
            assert_eq!(out.warm_start.bound, out.bound);
            assert_eq!(out.lp, 2.);
            assert!(out.history.is_empty());
            assert_eq!(out.warm_start.columns.len(), master.costs.len());
            // A timed-out solve must remain usable, including column identities.
            let resumed = cg::optimize(
                &mut master,
                &[],
                |_, _| {
                    Ok(Answer {
                        columns: vec![],
                        minimum: 0.,
                        exact: true,
                        bound_minimum: None,
                    })
                },
                &mut NoController,
                Options {
                    stabilize: false,
                    bound_mode: mode,
                    warm_start: Some(out.warm_start),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(resumed.status, Status::Optimal);
            assert_eq!(resumed.lp, 2.);
        }
    }
}

#[test]
fn pricing_timeout_defers_to_rollback_and_other_errors_still_propagate() {
    struct Rollback(bool);
    impl cg::Controller<ClpMaster> for Rollback {
        fn pricing_timeout_requires_rollback(&self) -> bool {
            self.0
        }
    }
    for (rollback, error) in [
        (
            true,
            baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "pricing hard time limit exceeded".into(),
            },
        ),
        (
            false,
            baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "resource or bucket arithmetic or state failure".into(),
            },
        ),
        (
            false,
            baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::InvalidArgument,
                message: "invalid pricing time limit".into(),
            },
        ),
        (
            false,
            baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "pricing label limit exceeded".into(),
            },
        ),
    ] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        let actual = cg::optimize(
            &mut master,
            &[],
            |_, _| Err(error.clone()),
            &mut Rollback(rollback),
            Options {
                stabilize: false,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(actual.kind, error.kind);
        assert_eq!(actual.message, error.message);
    }
}

#[test]
fn network_tree_pricing_timeout_retains_a_validated_initial_incumbent() {
    use baldes::{
        graph::Graph,
        master::{model, tree},
    };
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let route = baldes::labeling::Route {
        nodes: vec![0, 1, 2],
        arcs: vec![0, 1],
        original_cost: 2.,
        cost: 2.,
        resources: vec![0.],
        resource_profile: vec![vec![0.]; 3],
        coupled_values: vec![],
    };
    let problem = model::Problem::set_partitioning(graph, 1.).unwrap();
    let mut options = tree::ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.node.gap_enumeration = false;
    options
        .node
        .pricing_options
        .push(("BALDEShardTimeThresholdInPricing".into(), 0.));
    options.initial_solution = Some((
        2.,
        vec![model::FixedRoute {
            route: model::NetworkRoute { network: 0, route },
            quantity: 1.,
        }],
    ));
    let out = tree::solve(problem, options).unwrap();
    assert_eq!(out.status, tree::TreeStatus::TimeLimit);
    assert_eq!(out.incumbent, 2.);
    assert_eq!(out.routes.len(), 1);
    assert!(out.bound <= 2.);
}
