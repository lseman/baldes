#![cfg(feature = "clp")]

use baldes::master::{
    clp::ClpMaster,
    column_generation::{self, Answer, Column, NoController, Options, Status},
    RowSense,
};

#[test]
fn original_feasible_columns_can_recover_an_exhausted_penalized_master() {
    use baldes::Result;
    use column_generation::Controller;
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
    let output = column_generation::optimize(
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
        master::root::RootOptions,
        inputs::{Instance, Rounding},
    };
    let scale = 1_000_000.;
    let diagonal = (2f64 * scale * scale).sqrt();
    let nearest = 4. * scale + diagonal;
    let z = nearest - 4. * scale;
    let error = (4. * scale - (nearest - z)) + (diagonal - z);
    let upper = if error > 0. {
        f64::from_bits(nearest.to_bits() + 1)
    } else {
        nearest
    };
    let instance=Instance::parse("large triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100000000 0\n1 1000000 0 1 0 100000000 0\n2 0 1000000 1 0 100000000 0\n3 1000000 1000000 1 0 100000000 0\n").unwrap();
    let mut options = RootOptions::default().with_rank_one_cuts();
    options.incumbent = upper;
    let output = instance.solve_root(5, Rounding::None, options).unwrap();
    assert_eq!(output.master.status, Status::Optimal);
    // The exact original optimum is 4*scale plus the supplied floating
    // diagonal. TwoSum's residual decides which side of `nearest` it lies.
    assert!(output.master.bound < nearest || output.master.bound == nearest && error >= 0.);
    assert!(output.incumbent > nearest || output.incumbent == nearest && error <= 0.);
    assert!(output.incumbent - output.master.bound < 1e-6);
    assert!(output.cuts > 0);
}

#[test]
fn an_integer_certificate_does_not_terminate_fractional_lp_pricing() {
    use baldes::{pricing::phases::Phase, Result};
    use column_generation::Controller;
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
        options.stabilize = false;
        if warm {
            options.warm_start = Some(column_generation::WarmStart {
                bound: 1.,
                stabilization: None,
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
        let output = column_generation::optimize(
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
        let result = column_generation::optimize(
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
        ).unwrap();
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
    let result = column_generation::optimize(
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
    use baldes::{Graph, master::root::{self, RootOptions, RootProblem}};
    let upper = f64::from_bits(1f64.to_bits() - 1);
    let mut graph = Graph::new(4, 1).unwrap();
    for customer in [1,2] { graph.add_elementarity_set(&[customer]).unwrap(); }
    graph.initial[0] = Some(0.);
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., upper).unwrap();
        graph.set_ng(v,&[0,1]).unwrap();
    }
    for (u,v,cost) in [(0,1,0.7),(1,3,0.3),(0,2,0.7),(2,3,0.3),(1,2,0.),(2,1,0.)] {
        let id=graph.add_arc(u,v,cost).unwrap();
        graph.arc_mut(id).unwrap().consumption[0]=cost;
    }
    let problem=RootProblem::set_partitioning(graph,2.).unwrap();
    let mut options=RootOptions::default();
    options.enumerate=false;
    options.cg.max_penalty_updates=0;
    options.cg.time_limit=std::time::Duration::from_secs(1);
    let output=root::solve(problem,options).unwrap();
    assert_eq!(output.master.status,Status::Stalled);
    assert!(output.incumbent.is_infinite());
}

#[test]
fn rounded_resource_feasibility_does_not_publish_an_integer_incumbent() {
    use baldes::{Graph, master::root::{self, RootOptions, RootProblem}};
    let mut graph=Graph::new(3,1).unwrap();
    graph.initial[0]=Some(0.);
    graph.add_elementarity_set(&[1]).unwrap();
    for v in 0..3 {graph.set_bounds(v,0,0.,1.).unwrap();graph.set_ng(v,&[0]).unwrap();}
    for (u,v,c) in [(0,1,1.),(1,2,f64::EPSILON/2.)] {
        let id=graph.add_arc(u,v,1.).unwrap();graph.arc_mut(id).unwrap().consumption[0]=c;
    }
    let mut options=RootOptions::default();options.enumerate=false;
    options.pricing_options.push(("BALDESuseBidirectionalSearch".into(),0.));
    let output=root::solve(RootProblem::set_partitioning(graph,1.).unwrap(),options).unwrap();
    assert!(output.routes.iter().any(|r|r.arcs==[0,1]));
    assert!(output.incumbent.is_infinite());
}
