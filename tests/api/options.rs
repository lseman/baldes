use baldes::core::options::{OptionSupport, Options, OPTION_TABLE};
use baldes::pricing::Solver;

#[test]
fn public_metadata_lists_only_supported_canonical_names_and_consistent_indices() {
    let mut options = Options::default();
    let mut solver = Solver::new(2, 0).unwrap();
    let expected = if cfg!(all(feature = "clp", feature = "highs")) { 41 } else { 39 };
    assert_eq!(OPTION_TABLE.len(), expected);
    let mut names = std::collections::HashSet::new();
    for (index, info) in OPTION_TABLE.iter().enumerate() {
        assert!(info.name.starts_with("BALDES"));
        assert!(names.insert(info.name));
        assert!(matches!(info.support, OptionSupport::Implemented(_)));
        assert_eq!(Options::index(info.name).unwrap(), index);
        assert_eq!(options.get(info.name).unwrap(), info.default);
        assert_eq!(Options::default_of(info.name).unwrap(), info.default);
        assert_eq!(Options::support(info.name).unwrap(), info.support);
        options.set(info.name, info.default).unwrap();
        solver.set_option(info.name, info.default).unwrap();
        let legacy = format!("RCSP{}", info.name.strip_prefix("BALDES").unwrap());
        assert_eq!(Options::canonical_name(&legacy).unwrap(), info.name);
        assert_eq!(Options::index(&legacy).unwrap(), index);
        assert_eq!(options.get(&legacy).unwrap(), info.default);
        solver.set_option(&legacy, info.default).unwrap();
    }
    assert!(Options::support("unknown").is_err());
}

#[test]
fn unregistered_options_are_rejected_without_changing_supported_values() {
    let mut options = Options::default();
    let mut solver = Solver::new(2, 0).unwrap();
    let name = "BALDESnotARealOption";
    assert!(Options::support(name).is_err());
    assert!(Options::canonical_name(name).is_err());
    assert!(Options::index(name).is_err());
    assert!(Options::default_of(name).is_err());
    assert!(options.get(name).is_err());
    for value in [0., 1.] {
        assert!(options.set(name, value).unwrap_err().message.contains("unknown BALDES option"));
        assert!(solver.set_option(name, value).is_err());
    }
    for info in OPTION_TABLE {
        assert_eq!(options.get(info.name).unwrap(), info.default);
        assert_eq!(solver.options.get(info.name).unwrap(), info.default);
    }
    for name in ["BALDESmaxNumOfEnumSolutionsForMIP", "BALDESmaxNumOfEnumSolsForEndOfNodeMIP"] {
        let enabled = cfg!(all(feature = "clp", feature = "highs"));
        assert_eq!(Options::index(name).is_ok(), enabled);
        assert_eq!(Options::support(name).is_ok(), enabled);
        assert_eq!(options.get(name).is_ok(), enabled);
    }
    #[cfg(feature = "clp")]
    {
        let policy = baldes::master::root::EnumerationMipPolicy::from_options(&options).unwrap();
        assert_eq!(policy.normal_route_limit, 10_000);
        assert_eq!(policy.end_of_node_route_limit, 10_000);
    }
}

#[test]
fn legacy_aliases_share_storage_and_reach_runtime_validation() {
    let mut options = Options::default();
    options.set("RCSPnumberOfBucketsPerVertex", 17.).unwrap();
    assert_eq!(options.get("BALDESnumberOfBucketsPerVertex").unwrap(), 17.);
    options.set("BALDESnumberOfBucketsPerVertex", 23.).unwrap();
    assert_eq!(options.get("RCSPnumberOfBucketsPerVertex").unwrap(), 23.);
    let mut solver = Solver::new(2, 0).unwrap();
    solver.set_option("RCSPlabelDominanceStrategy", 1.).unwrap();
    assert_eq!(solver.dominance_mode, 1);
    assert!(solver.set_option("RCSPlabelDominanceStrategy", 2.).is_err());
    assert_eq!(solver.dominance_mode, 1);
    assert_eq!(solver.options.get("BALDESlabelDominanceStrategy").unwrap(), 1.);
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
        ("BALDEShardTimeThresholdInPricing", f64::MAX),
        ("BALDESmaxNumOfColsPerExactIteration", 0.),
        ("BALDESlabelDominanceStrategy", 2.),
        ("BALDESuseBidirectionalSearch", 3.),
        ("BALDESdynamicBucketSteps", 2.),
        ("BALDEScapacityCutsSeparator", 0.),
        ("BALDEScapCutsMaxNumPerRound", 0.),
        ("BALDESmaxNumOfLabelsInEnumeration", -1.),
        ("BALDESmaxNumOfColsPerIteration", 0.),
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
    solver.set_option("BALDESlabelDominanceStrategy", 1.).unwrap();
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
            or_rcsp_get_option(p, c"BALDESlabelDominanceStrategy".as_ptr(), &mut value),
            0
        );
        assert_eq!(value, 1.);
        assert_eq!(or_rcsp_set_label_dominance_strategy(p, 2), -1);
        assert_eq!(
            or_rcsp_get_option(p, c"BALDESlabelDominanceStrategy".as_ptr(), &mut value),
            0
        );
        assert_eq!(value, 1.);
        or_rcsp_delete(p);
    }
}

#[test]
fn c_api_lists_supported_canonical_names_and_accepts_legacy_inputs() {
    use baldes::api::*;
    use std::ffi::CStr;
    // SAFETY: Live local handles, static NUL-terminated names and writable outputs.
    unsafe {
        assert_eq!(or_rcsp_option_count() as usize, OPTION_TABLE.len());
        for (i, info) in OPTION_TABLE.iter().enumerate() {
            assert_eq!(CStr::from_ptr(or_rcsp_option_name(i as i32)).to_str().unwrap(), info.name);
        }
        assert!(or_rcsp_option_name(-1).is_null());
        assert!(or_rcsp_option_name(OPTION_TABLE.len() as i32).is_null());
        let p = or_rcsp_new(2, 0);
        assert_eq!(or_rcsp_set_option(p, c"RCSPlabelDominanceStrategy".as_ptr(), 1.), 0);
        let mut value = f64::NAN;
        assert_eq!(or_rcsp_get_option(p, c"BALDESlabelDominanceStrategy".as_ptr(), &mut value), 0);
        assert_eq!(value, 1.);
        assert_eq!(or_rcsp_set_option(p, c"BALDESlabelDominanceStrategy".as_ptr(), 0.), 0);
        assert_eq!(or_rcsp_get_option(p, c"RCSPlabelDominanceStrategy".as_ptr(), &mut value), 0);
        assert_eq!(value, 0.);
        assert_eq!(or_rcsp_set_option(p, c"BALDESnotARealOption".as_ptr(), 0.), -1);
        assert!(CStr::from_ptr(or_rcsp_last_error()).to_str().unwrap().contains("unknown BALDES option"));
        value = 123.;
        assert_eq!(or_rcsp_get_option(p, c"BALDESnotARealOption".as_ptr(), &mut value), -1);
        assert_eq!(value, 123.);
        or_rcsp_delete(p);
    }
}

#[cfg(feature = "clp")]
#[test]
fn capacity_option_controls_separation_in_fresh_and_inherited_nodes() {
    use baldes::master::root::{self, CapacityCutConfig, RootOptions};
    use baldes::inputs::{Instance, Rounding};
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
        .push(("BALDEScapCutsMaxNumPerRound".into(), 1.));
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
    use baldes::inputs::{Instance, Rounding};
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
        .push(("BALDEScapCutsMaxNumPerRound".into(), 1.));
    let limited = instance.solve_root(8, Rounding::None, options).unwrap();
    assert_eq!(limited.cuts, 1);
    assert!(full.cuts > limited.cuts);
}
