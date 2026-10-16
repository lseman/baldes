#![cfg(feature = "clp")]

use baldes::master::{
    clp::ClpMaster,
    column_generation::{self, Answer, Column, NoController, Options, Status},
    RowSense,
};

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
