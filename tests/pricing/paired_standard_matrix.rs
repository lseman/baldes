use baldes::master::{
    cg::Status,
    model::{self, format::Model},
};

#[test]
fn registered_two_standard_slots_and_equality_match_native_owner_objectives() {
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-standard-dispatch.json"
    ))
    .unwrap();
    let matrix: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-standard-matrix.json"
    ))
    .unwrap();
    assert_eq!(matrix["reference_sha256"], baseline["reference_sha256"]);
    for case in matrix["cases"].as_array().unwrap() {
        assert!(case["factory"]["returned_nonnull"].as_bool().unwrap());
        let mut input = baseline["cases"][0]["model"].clone();
        let graph = &mut input["networks"][0];
        graph["resources"].as_array_mut().unwrap().truncate(1);
        let profiles = case["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 2);
        for profile in profiles {
            graph["resources"].as_array_mut().unwrap().push(serde_json::json!({
                "kind": if profile["non_disposable"].as_bool().unwrap_or(false) { "equality" } else { "disposable" },
                "bounds": profile["bounds"],
            }));
        }
        let sink = graph["sink"].as_u64().unwrap();
        for arc in graph["arcs"].as_array_mut().unwrap() {
            let quantity = if arc["head"].as_u64().unwrap() == sink {
                0.
            } else {
                1.
            };
            let consumption = arc["consumption"].as_array_mut().unwrap();
            consumption.truncate(1);
            consumption.extend([serde_json::json!(quantity), serde_json::json!(quantity)]);
        }
        let problem = Model::from_json(&input.to_string())
            .unwrap()
            .problem::<3>()
            .unwrap();
        let mut options = model::Options::default();
        options.cg.stabilize = false;
        let output = model::solve(problem, options).unwrap();
        assert_eq!(output.master.status, Status::Optimal, "{}", case["name"]);
        assert_eq!(
            output.master.lp,
            case["native_result"]["bcRecRootLpVal"].as_f64().unwrap(),
            "{}",
            case["name"]
        );
        if profiles[1]["non_disposable"].as_bool().unwrap_or(false) {
            for route in &output.routes {
                assert_eq!(route.route.resources[2], 2.);
            }
        }
    }
}

#[test]
fn paired_pricing_checks_each_of_three_independent_standard_resources() {
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-standard-dispatch.json"
    ))
    .unwrap();
    // This exercises generic scalar composition beyond the pinned Std2
    // instantiation; only the separate two-slot test claims native parity.
    for tight_slot in 1..=3 {
        let mut input = baseline["cases"][0]["model"].clone();
        let graph = &mut input["networks"][0];
        let ordinary = graph["resources"][1].clone();
        graph["resources"]
            .as_array_mut()
            .unwrap()
            .extend([ordinary.clone(), ordinary]);
        for bounds in graph["resources"][tight_slot]["bounds"]
            .as_array_mut()
            .unwrap()
        {
            bounds[1] = serde_json::json!(1.);
        }
        for arc in graph["arcs"].as_array_mut().unwrap() {
            let quantity = arc["consumption"][1].clone();
            arc["consumption"]
                .as_array_mut()
                .unwrap()
                .extend([quantity.clone(), quantity]);
        }
        let problem = Model::from_json(&input.to_string())
            .unwrap()
            .problem::<4>()
            .unwrap();
        let mut options = model::Options::default();
        options.cg.stabilize = false;
        let output = model::solve(problem, options).unwrap();
        assert_eq!(
            output.master.status,
            Status::Optimal,
            "tight slot {tight_slot}"
        );
        assert_eq!(output.master.lp, 7.5, "tight slot {tight_slot}");
        for route in &output.routes {
            assert!(
                route.route.resources[tight_slot] <= 1.,
                "tight slot {tight_slot}"
            );
        }
    }
}
