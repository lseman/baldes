#![cfg(feature = "clp")]

use baldes::master::{
    clp::ClpMaster,
    column_generation::{self, Answer, Column, NoController, Options, Status},
    RowSense,
};

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
