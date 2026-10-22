use super::*;
use crate::master::solvers::clp::ClpMaster;

fn run(model: &ClpMaster, cutoff: f64) -> DeluxingOutput {
    filter(
        model,
        cutoff,
        &DeluxingOptions {
            clusters: 3,
            min_extra_fixes: 0,
            ..Default::default()
        },
        ClpMaster::new,
    )
    .unwrap()
}

#[test]
fn initial_fixing_retains_cutoff_equality_and_source_state() {
    let mut m = ClpMaster::new().unwrap();
    m.add_row(RowSense::Equal, 1., &[]).unwrap();
    for cost in [1., 2., 5.] {
        m.add_column(cost, &[(0, 1.)]).unwrap();
    }
    assert!(m.solve());
    let old = (m.costs.clone(), m.values.clone(), m.duals.clone());
    let report = run(&m, 2.);
    assert_eq!(report.fixed_columns, [2]);
    assert_eq!((m.costs.clone(), m.values.clone(), m.duals.clone()), old);
}

#[test]
fn zero_budget_and_invalid_options() {
    let mut m = ClpMaster::new().unwrap();
    m.add_column(1., &[]).unwrap();
    let options = DeluxingOptions {
        time_limit: Duration::ZERO,
        ..Default::default()
    };
    let r = filter(&m, 2., &options, ClpMaster::new).unwrap();
    assert!(r.stopped_early);
    assert!(r.fixed_columns.is_empty());
    assert!(filter(&m, f64::INFINITY, &options, ClpMaster::new).is_err());
    assert!(filter(
        &m,
        1.,
        &DeluxingOptions {
            clusters: 0,
            ..options
        },
        ClpMaster::new
    )
    .is_err());
}

#[test]
fn exhaustive_integer_witnesses_survive_mixed_rows_bounds_and_offsets() {
    // Deterministic generated small models, checked against every integer
    // assignment. Include negative costs/RHS, mixed senses and multiplicities.
    for seed in 0..60usize {
        let n = 5;
        let mut m = ClpMaster::new().unwrap();
        m.objective_offset = -3.;
        let senses = [RowSense::Equal, RowSense::LessEqual, RowSense::GreaterEqual];
        let rhs = [(seed % 4) as f64, (seed % 5) as f64, -2.];
        for (&s, &b) in senses.iter().zip(&rhs) {
            m.add_row(s, b, &[]).unwrap();
        }
        let mut matrix = vec![];
        for j in 0..n {
            let a: Vec<_> = (0..3)
                .map(|r| ((seed * 13 + j * 7 + r * 11) % 5) as f64 - 2.)
                .collect();
            let entries: Vec<_> = a.iter().copied().enumerate().collect();
            let c = m
                .add_column(((seed * 3 + j * 11) % 13) as f64 - 4., &entries)
                .unwrap();
            m.set_upper(c, 2.75).unwrap();
            matrix.push(a);
        }
        let cutoff = (seed % 8) as f64 - 2.;
        let report = run(&m, cutoff);
        for encoded in 0..3usize.pow(n as u32) {
            let mut code = encoded;
            let mut x = vec![0.; n];
            for v in &mut x {
                *v = (code % 3) as f64;
                code /= 3;
            }
            let activity: Vec<f64> = (0..3)
                .map(|r| (0..n).map(|j| matrix[j][r] * x[j]).sum())
                .collect();
            let feasible = activity.iter().enumerate().all(|(r, &a)| match senses[r] {
                RowSense::Equal => a == rhs[r],
                RowSense::LessEqual => a <= rhs[r],
                RowSense::GreaterEqual => a >= rhs[r],
            });
            let cost = m.objective_offset + dot(&m.costs, &x);
            if feasible && cost <= cutoff {
                assert!(
                    report.fixed_columns.iter().all(|&j| x[j] == 0.),
                    "seed {seed}, x={x:?}, fixed={:?}",
                    report.fixed_columns
                );
            }
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn filtered_mip_preserves_optimum_and_qualifies_cutoff_proof() {
    use crate::master::solvers::{
        highs,
        mip::{MipOptions, MipStatus},
    };
    let mut m = ClpMaster::new().unwrap();
    m.add_row(RowSense::Equal, 1., &[]).unwrap();
    m.add_column(2., &[(0, 1.)]).unwrap();
    m.add_column(20., &[(0, 1.)]).unwrap();
    let options = MipOptions {
        objective_cutoff: 5.,
        ..Default::default()
    };
    let (out, report) =
        highs::solve_integer_with_deluxing(&m, &options, &DeluxingOptions::default()).unwrap();
    assert_eq!(out.status, MipStatus::Optimal);
    assert_eq!(out.solution.unwrap().values, [1., 0.]);
    assert_eq!(report.fixed_columns, [1]);
    let options = MipOptions {
        objective_cutoff: 1.,
        ..options
    };
    let (out, _) =
        highs::solve_integer_with_deluxing(&m, &options, &DeluxingOptions::default()).unwrap();
    assert_eq!(out.status, MipStatus::ObjectiveCutoff);
    assert_eq!(out.bound, 1.);
}

#[test]
fn auxiliary_round_fixes_a_column_missed_by_initial_reduced_costs() {
    let mut m = ClpMaster::new().unwrap();
    m.add_row(RowSense::Equal, 1., &[]).unwrap();
    m.add_row(RowSense::Equal, 1., &[]).unwrap();
    m.add_column(1., &[(0, 1.)]).unwrap();
    m.add_column(1., &[(1, 1.)]).unwrap();
    // LP reduced cost 2.5 is below UB-LB=3, but this column overcovers
    // the first equality whenever its integer value is positive.
    m.add_column(4.5, &[(0, 2.)]).unwrap();
    for clustering_limit in [0, 10_000] {
        let report = filter(
            &m,
            5.,
            &DeluxingOptions {
                clusters: 3,
                min_extra_fixes: 0,
                clustering_limit,
                ..Default::default()
            },
            ClpMaster::new,
        )
        .unwrap();
        assert_eq!(report.initial_fixes, 0);
        assert_eq!(report.fixed_columns, [2]);
        assert!(report.auxiliary_solves > 0);
    }
}

#[test]
fn empty_pool_and_fixed_empty_column_are_supported() {
    let mut m = ClpMaster::new().unwrap();
    assert!(run(&m, 1.).fixed_columns.is_empty());
    let c = m.add_column(10., &[]).unwrap();
    assert!(m.column_entries(c).unwrap().is_empty());
    m.set_upper(c, 0.75).unwrap();
    let report = run(&m, 1.);
    assert_eq!(report.fixed_columns, [c]);
    assert_eq!(report.initial_fixes, 1);
}
