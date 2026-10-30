use super::*;

#[test]
fn coupled_envelope_admission_encloses_original_columns_under_signed_duals() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let (problem, registry) = format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let values = case["activity_values_by_path"].as_array().unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.preprocessing.enabled = false;
        registry
            .configure_boxes(
                &mut options,
                &[OriginalVariableBound {
                    network: 42,
                    id: 37,
                    lower: values
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .fold(f64::INFINITY, f64::min),
                    upper: Some(
                        values
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .fold(f64::NEG_INFINITY, f64::max),
                    ),
                }],
            )
            .unwrap();
        let mut output = solve(problem, options).unwrap();
        if let Some(entry) = output.snapshot.state.oracles[0]
            .activity_rows
            .first()
            .cloned()
        {
            // Preserve the actual kernel aggregation order under cancellation.
            // Folding these coefficients algebraically into a single dual
            // adjustment would erase the row's floating-point residual.
            for coefficient in [4_503_599_627_370_496., -4_503_599_627_370_496.] {
                output.snapshot.state.oracles[0]
                    .activity_rows
                    .push(AccumulatedActivityEntry {
                        resource: entry.resource,
                        row: entry.row,
                        coefficient,
                    });
            }
        }
        let state = &output.snapshot.state;
        let oracle = &state.oracles[0];
        let groups = &state.grouped_variable_boxes[&42];
        for dual in [-2., -0.5, 0., 0.5, 2.] {
            let duals: Vec<_> = (0..state.shared_offset + state.shared_rows.len())
                .map(|r| if r % 2 == 0 { dual } else { -dual })
                .collect();
            let bound =
                grouped_boxes::coupled_admission(oracle, groups, &duals, state.shared_offset)
                    .unwrap();
            for arcs in [vec![0, 1], vec![2, 3]] {
                let route = Route {
                    coupled_values: Vec::new(),
                    nodes: vec![],
                    arcs,
                    cost: 0.,
                    original_cost: 0.,
                    resources: vec![],
                    resource_profile: vec![],
                };
                let positions: Vec<_> = route.arcs.iter().map(|id| oracle.index[id]).collect();
                let mut raw = positions
                    .iter()
                    .map(|&p| oracle.network.graph.arcs[p].cost)
                    .sum::<f64>()
                    + oracle
                        .network
                        .graph
                        .coupled_kernel()
                        .unwrap()
                        .replay(&positions)
                        .unwrap()
                        .unwrap();
                for (r, a) in entries(oracle, &route, state.shared_offset).unwrap() {
                    raw -= duals[r] * a;
                }
                assert!(
                    bound.minimum <= raw,
                    "{} dual={dual} lower={} raw={raw}",
                    case["name"],
                    bound.minimum
                );
            }
        }
    }
}

#[test]
fn required_coupled_box_admission_retains_positive_required_quantity() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let case = &corpus["cases"][0];
    let mut source = case["model"].clone();
    source["rows"] = serde_json::json!([]);
    source["networks"][0]["multiplicity"] = serde_json::json!([1, 1]);
    for variable in source["pricing_variables"].as_array_mut().unwrap() {
        variable["entries"] = serde_json::json!([]);
        if variable["id"] != 37 {
            variable["cost"] = 0.into();
        }
    }
    let (problem, registry) = format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.preprocessing.enabled = false;
        options.cg.bound_mode = mode;
        registry.configure_pricing(&mut options);
        let baseline = solve(problem.clone(), options.clone()).unwrap();
        registry
            .configure_boxes(
                &mut options,
                &[OriginalVariableBound {
                    network: 42,
                    id: 37,
                    lower: 2.,
                    upper: Some(11.),
                }],
            )
            .unwrap();
        let boxed = solve(problem.clone(), options).unwrap();
        assert_eq!(boxed.master.status, cg::Status::Optimal);
        assert_eq!(boxed.master.lp, 2.);
        assert_eq!(boxed.master.bound, 2.);
        assert!(boxed.oracle_calls[0].1 > 0);
        assert!(
            boxed.oracle_calls[0].1 <= baseline.oracle_calls[0].1,
            "boxed={:?} plain={:?}",
            boxed.oracle_calls,
            baseline.oracle_calls
        );
        let state = &boxed.snapshot.state;
        let oracle = &state.oracles[0];
        let groups = &state.grouped_variable_boxes[&42];
        let duals = vec![0.; state.shared_offset + state.shared_rows.len()];
        let bound =
            grouped_boxes::coupled_admission(oracle, groups, &duals, state.shared_offset).unwrap();
        assert_eq!(bound.minimum, 2.);
        assert_eq!(bound.correction, 2.);
        assert!(!bound.zero_quantity);
    }
}
