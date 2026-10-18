use baldes::core::options::{OptionKind, OptionSupport, Options, OPTION_TABLE};
use baldes::pricing::Solver;

#[test]
fn recovered_defaults_remain_readable_and_unsupported_overrides_are_atomic() {
    let mut options = Options::default();
    let mut solver = Solver::new(2, 0).unwrap();
    assert_eq!(OPTION_TABLE.len(), 58);
    let mut unsupported = 0;
    for info in OPTION_TABLE {
        assert_eq!(options.get(info.name).unwrap(), info.default);
        assert_eq!(Options::default_of(info.name).unwrap(), info.default);
        assert_eq!(Options::support(info.name).unwrap(), info.support);
        options.set(info.name, info.default).unwrap();
        solver.set_option(info.name, info.default).unwrap();
        if let OptionSupport::Unsupported(reason) = info.support {
            unsupported += 1;
            assert!(!reason.is_empty());
            let changed = if info.kind == OptionKind::Bool {
                1. - info.default
            } else {
                info.default + 1.
            };
            for error in [
                options.set(info.name, changed).unwrap_err(),
                solver.set_option(info.name, changed).unwrap_err(),
            ] {
                assert!(error.message.contains(info.name));
                assert!(error.message.contains(reason));
            }
            assert_eq!(options.get(info.name).unwrap(), info.default);
            assert_eq!(solver.options.get(info.name).unwrap(), info.default);
        }
    }
    assert_eq!(
        unsupported,
        if cfg!(all(feature = "clp", feature = "highs")) {
            17
        } else {
            19
        }
    );
    assert!(Options::support("unknown").is_err());
}

#[test]
fn failed_solver_setters_preserve_options_controls_phases_and_prepared_engine() {
    let mut solver = Solver::new(2, 0).unwrap();
    solver.graph.add_arc(0, 1, -2.).unwrap();
    solver.price_phase(0, 10, true).unwrap();
    let values: Vec<_> = OPTION_TABLE
        .iter()
        .map(|o| solver.options.get(o.name).unwrap())
        .collect();
    let phases = solver.phases.clone();
    let time = solver.controls.time_limit;
    let columns = solver.controls.max_columns;
    for (name, value) in [
        ("RCSPhardTimeThresholdInPricing", f64::MAX),
        ("RCSPmaxNumOfColsPerExactIteration", 0.),
        ("RCSPlabelDominanceStrategy", 2.),
        ("RCSPuseBidirectionalSearch", 3.),
        ("RCSPdynamicBucketSteps", 2.),
        ("RCSPcapacityCutsSeparator", 0.),
        ("RCSPcapCutsMaxNumPerRound", 0.),
        ("RCSPmaxNumOfLabelsInEnumeration", -1.),
        ("RCSPmaxNumOfColsPerIteration", 0.),
    ] {
        assert!(solver.set_option(name, value).is_err(), "{name}");
        assert_eq!(
            OPTION_TABLE
                .iter()
                .map(|o| solver.options.get(o.name).unwrap())
                .collect::<Vec<_>>(),
            values
        );
        assert_eq!(solver.controls.time_limit, time);
        assert_eq!(solver.controls.max_columns, columns);
        assert_eq!(solver.phases, phases);
        assert!(solver.engine.is_some());
    }
    assert_eq!(
        solver
            .price_phase(0, 10, true)
            .unwrap()
            .minimum_reduced_cost,
        -2.
    );
}

#[test]
fn named_dominance_option_reaches_labeling_and_the_c_setter_stays_synchronized() {
    let mut solver = Solver::new(2, 0).unwrap();
    solver.graph.add_arc(0, 1, -2.).unwrap();
    solver.set_option("RCSPlabelDominanceStrategy", 1.).unwrap();
    assert_eq!(solver.dominance_mode, 1);
    assert_eq!(
        solver
            .price_phase(0, 10, true)
            .unwrap()
            .minimum_reduced_cost,
        -2.
    );
    assert_eq!(solver.engine.as_ref().unwrap().controls.dominance_mode, 1);
    // SAFETY: Local live handle, NUL-terminated names, and writable output.
    unsafe {
        use baldes::api::*;
        let p = or_rcsp_new(2, 0);
        let mut value = -1.;
        assert_eq!(or_rcsp_set_label_dominance_strategy(p, 1), 0);
        assert_eq!(
            or_rcsp_get_option(p, c"RCSPlabelDominanceStrategy".as_ptr(), &mut value),
            0
        );
        assert_eq!(value, 1.);
        assert_eq!(or_rcsp_set_label_dominance_strategy(p, 2), -1);
        assert_eq!(
            or_rcsp_get_option(p, c"RCSPlabelDominanceStrategy".as_ptr(), &mut value),
            0
        );
        assert_eq!(value, 1.);
        or_rcsp_delete(p);
    }
}

#[test]
fn c_api_reports_unsupported_changes_without_losing_the_recovered_default() {
    use baldes::api::*;
    use std::ffi::{CStr, CString};
    // SAFETY: Locally owned handles and valid name/output pointers.
    unsafe {
        let p = or_rcsp_new(2, 0);
        for info in OPTION_TABLE {
            if let OptionSupport::Unsupported(_) = info.support {
                let name = CString::new(info.name).unwrap();
                let changed = if info.kind == OptionKind::Bool {
                    1. - info.default
                } else {
                    info.default + 1.
                };
                assert_eq!(or_rcsp_set_option(p, name.as_ptr(), changed), -1);
                assert!(CStr::from_ptr(or_rcsp_last_error())
                    .to_str()
                    .unwrap()
                    .contains("unsupported RCSP option"));
                let mut value = f64::NAN;
                assert_eq!(or_rcsp_get_option(p, name.as_ptr(), &mut value), 0);
                assert_eq!(value, info.default);
                assert_eq!(or_rcsp_set_option(p, name.as_ptr(), info.default), 0);
            }
        }
        assert_eq!(or_rcsp_option_count(), 58);
        or_rcsp_delete(p);
    }
}

#[cfg(feature = "clp")]
#[test]
fn capacity_option_controls_separation_in_fresh_and_inherited_nodes() {
    use baldes::master::root::{self, CapacityCutConfig, RootOptions};
    use baldes::vrptw::{Instance, Rounding};
    let instance = Instance::parse("option-test\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let mut options =
        RootOptions::default().with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.));
    options.enumerate = false;
    options.capture_state = true;
    options
        .pricing_options
        .push(("RCSPuseCapacityCuts".into(), 0.));
    let parent = instance
        .solve_root(8, Rounding::None, options.clone())
        .unwrap();
    assert_eq!(parent.cuts, 0);
    let snapshot = parent.snapshot.as_ref().unwrap();
    options.pricing_options.clear();
    // An omitted child override preserves the parent's disabled separator.
    let disabled = root::inherit_branches(snapshot, &[], options.clone()).unwrap();
    assert_eq!(disabled.cuts, 0);
    assert!((disabled.master.lp - parent.master.lp).abs() < 1e-8);
    options
        .pricing_options
        .push(("RCSPuseCapacityCuts".into(), 1.));
    options
        .pricing_options
        .push(("RCSPcapCutsMaxNumPerRound".into(), 1.));
    let enabled = root::inherit_branches(snapshot, &[], options.clone()).unwrap();
    let fresh = instance.solve_root(8, Rounding::None, options).unwrap();
    assert!(enabled.cuts > 0);
    assert!(enabled.master.lp > parent.master.lp + 0.1);
    assert!((enabled.master.lp - fresh.master.lp).abs() < 1e-8);
}

#[cfg(feature = "clp")]
#[test]
fn capacity_per_round_option_caps_the_number_of_generated_rows() {
    use baldes::master::root::RootOptions;
    use baldes::vrptw::{Instance, Rounding};
    // Two distant odd customer groups each need their own capacity cut.
    let instance = Instance::parse("two-triangles\nVEHICLE\nNUMBER CAPACITY\n6 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1000 0\n1 10 10 1 0 1000 0\n2 11 10 1 0 1000 0\n3 10 11 1 0 1000 0\n4 -10 -10 1 0 1000 0\n5 -11 -10 1 0 1000 0\n6 -10 -11 1 0 1000 0\n").unwrap();
    let mut options = RootOptions::default()
        .with_capacity_cuts(instance.capacity_cut_config(Rounding::None, true));
    options.enumerate = false;
    options.cg.stabilize = false;
    options.cg.max_cut_rounds = 1;
    let full = instance
        .solve_root(8, Rounding::None, options.clone())
        .unwrap();
    options
        .pricing_options
        .push(("RCSPcapCutsMaxNumPerRound".into(), 1.));
    let limited = instance.solve_root(8, Rounding::None, options).unwrap();
    assert_eq!(limited.cuts, 1);
    assert!(full.cuts > limited.cuts);
}
