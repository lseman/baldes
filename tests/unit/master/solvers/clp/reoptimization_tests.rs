use super::*;

#[test]
fn scaling_defaults_to_off() {
    let root = crate::master::routing::RootOptions::default();
    assert_eq!(root.clp_options.scaling, ClpScaling::Off);
    assert!(root.clp_options.keep_work_areas);
    assert_eq!(ClpOptions::default().scaling, ClpScaling::Off);
    assert_eq!(ClpScaling::default(), ClpScaling::Off);
}

fn verify(master: &mut ClpMaster, expected: f64) {
    assert_eq!(
        master
            .solve_limited(100_000, Duration::from_secs(60))
            .unwrap(),
        ClpSolveOutcome::Optimal
    );
    assert!(
        (master.objective - expected).abs() < 1e-7,
        "{} != {expected}",
        master.objective
    );
    let mut activity = vec![0.; master.rhs.len()];
    let mut objective = master.objective_offset;
    for column in 0..master.costs.len() {
        let x = master.values[column];
        assert!(x >= -1e-7 && x <= master.upper[column] + 1e-7);
        objective += master.costs[column] * x;
        let mut reduced = master.costs[column];
        for (row, a) in master.column_entries(column).unwrap() {
            activity[row] += a * x;
            reduced -= a * master.duals[row];
        }
        assert!((reduced - master.reduced_costs[column]).abs() < 1e-7);
    }
    assert!((objective - expected).abs() < 1e-7);
    for (row, &value) in activity.iter().enumerate() {
        let rhs = master.rhs[row];
        assert!(match master.senses[row] {
            RowSense::Equal => (value - rhs).abs() < 1e-7,
            RowSense::LessEqual => value <= rhs + 1e-7,
            RowSense::GreaterEqual => value >= rhs - 1e-7,
        });
    }
}

#[test]
fn reoptimization_policies_preserve_edits_forks_and_restoration() {
    for scaling in [
        ClpScaling::Auto,
        ClpScaling::Geometric,
        ClpScaling::Ruiz,
        ClpScaling::Equilibrium,
        ClpScaling::Off,
    ] {
        for keep_work_areas in [false, true] {
            let options = ClpOptions {
                scaling,
                keep_work_areas,
            };
            let mut master = ClpMaster::new().unwrap();
            master.set_clp_options(options);
            master.add_row(RowSense::GreaterEqual, 2., &[]).unwrap();
            master.add_column(3., &[(0, 1.)]).unwrap();
            master.add_column(2., &[(0, 1.)]).unwrap();
            master.set_upper(0, 3.).unwrap();
            master.set_upper(1, 4.).unwrap();
            verify(&mut master, 4.);
            master.add_column(1., &[(0, 1.)]).unwrap();
            verify(&mut master, 2.);
            master.add_row(RowSense::LessEqual, 1., &[(2, 1.)]).unwrap();
            verify(&mut master, 3.);
            let mut sibling = master.try_clone().unwrap();
            assert_eq!(sibling.clp_options(), options);
            sibling.set_upper(1, 0.).unwrap();
            verify(&mut sibling, 4.);
            verify(&mut master, 3.);
            master.set_upper(2, 0.).unwrap();
            verify(&mut master, 4.);
            master.set_rhs(&[3., 1.]).unwrap();
            verify(&mut master, 6.);
            master.set_coefficient(0, 1, 2.).unwrap();
            verify(&mut master, 3.);
            master.set_cost(0, 0.5).unwrap();
            verify(&mut master, 1.5);
            master.remove(&[1], &[2]).unwrap();
            verify(&mut master, 1.5);
            master.set_upper(0, 0.).unwrap();
            verify(&mut master, 3.);
            master.set_cost(1, 4.).unwrap();
            verify(&mut master, 6.);
            let mut target = master.try_clone().unwrap();
            let target_options = ClpOptions {
                scaling: if scaling == ClpScaling::Off {
                    ClpScaling::Geometric
                } else {
                    ClpScaling::Off
                },
                keep_work_areas: !keep_work_areas,
            };
            target.set_clp_options(target_options);
            target.set_upper(0, 3.).unwrap();
            assert!(target.continue_from(&master).unwrap());
            assert_eq!(target.clp_options(), target_options);
            verify(&mut target, 1.5);
            verify(&mut master, 6.);
        }
    }
}

#[test]
fn scaling_policies_solve_original_scaled_rows() {
    extern "C" {
        fn Clp_scalingFlag(model: *mut Model) -> i32;
    }
    for scaling in [
        ClpScaling::Auto,
        ClpScaling::Geometric,
        ClpScaling::Ruiz,
        ClpScaling::Equilibrium,
        ClpScaling::Off,
    ] {
        assert_eq!(
            format!("{:?}", scaling)
                .to_lowercase()
                .parse::<ClpScaling>()
                .unwrap(),
            scaling
        );
        for keep_work_areas in [false, true] {
            let mut master = ClpMaster::new().unwrap();
            master.set_clp_options(ClpOptions {
                scaling,
                keep_work_areas,
            });
            // Verify that Clp accepted the requested native mode, including
            // the additional Ruiz mode, before checking the solution.
            assert_eq!(
                unsafe { Clp_scalingFlag(master.model.unwrap().as_ptr()) },
                scaling.code()
            );
            master.add_row(RowSense::Equal, 1., &[]).unwrap();
            master.add_column(1., &[(0, 1e-4)]).unwrap();
            master.add_column(2., &[(0, 1e4)]).unwrap();
            verify(&mut master, 2e-4);
            assert!(master.solve());
            assert!((master.objective - 2e-4).abs() < 1e-9);
            for next in [
                ClpScaling::Geometric,
                ClpScaling::Ruiz,
                ClpScaling::Auto,
                ClpScaling::Equilibrium,
            ] {
                master.set_clp_options(ClpOptions {
                    scaling: next,
                    keep_work_areas,
                });
                assert_eq!(
                    unsafe { Clp_scalingFlag(master.model.unwrap().as_ptr()) },
                    next.code()
                );
                verify(&mut master, 2e-4);
            }
        }
    }
    assert!("unknown".parse::<ClpScaling>().is_err());
}

#[test]
fn native_matrix_validation_merges_duplicates_across_columns_and_repeated_solves() {
    // Exercise the native validation path directly: the public Rust editor
    // canonicalizes entries before passing them to Clp.
    for scaling in [0, 1, 3] {
        let master = ClpMaster::new().unwrap();
        let model = master.model.unwrap().as_ptr();
        let lower = [3., 4.];
        let upper = [f64::MAX; 4];
        let starts = [0, 3, 5, 7, 7];
        let rows = [0, 0, 1, 0, 1, 1, 1];
        let values = [0.4, 0.6, 1e-30, 1., 1., 1.5, 0.5];
        let costs = [2., 3., 1., 9.];
        // SAFETY: this owned native model is live, all sparse array sizes
        // match the supplied row/column counts, and no pointers escape.
        unsafe {
            Clp_scaling(model, scaling);
            Clp_addRows(
                model,
                2,
                lower.as_ptr(),
                upper.as_ptr(),
                [0, 0, 0].as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            );
            Clp_addColumns(
                model,
                4,
                [0.; 4].as_ptr(),
                upper.as_ptr(),
                costs.as_ptr(),
                starts.as_ptr(),
                rows.as_ptr(),
                values.as_ptr(),
            );
            assert_eq!(
                std::slice::from_raw_parts(Clp_getVectorLengths(model), 4),
                &[3, 2, 2, 0]
            );
            for _ in 0..2 {
                assert_eq!(baldes_clp_dual_limited(model, 1000, 60., 1), 0);
                assert_eq!(Clp_status(model), 0);
                assert!((Clp_objectiveValue(model) - 8.).abs() < 1e-9);
                let solution = std::slice::from_raw_parts(Clp_primalColumnSolution(model), 4);
                for (&actual, expected) in solution.iter().zip([3., 0., 2., 0.]) {
                    assert!((actual - expected).abs() < 1e-9);
                }
                assert_eq!(
                    std::slice::from_raw_parts(Clp_getVectorLengths(model), 4),
                    &[1, 2, 1, 0]
                );
            }
        }
    }
}
