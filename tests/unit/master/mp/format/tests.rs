use super::*;

#[test]
fn network_pricing_schedule_round_trips_and_configures_root_and_tree() {
    for (name, policy) in [
        ("full-sweep", NetworkPricingSchedule::FullSweep),
        ("cyclic", NetworkPricingSchedule::Cyclic),
        ("sorted", NetworkPricingSchedule::Sorted),
    ] {
        let options =
            SolveOptions::from_json(&format!(r#"{{"network_pricing_schedule":"{name}"}}"#))
                .unwrap();
        assert_eq!(
            options.node_options().unwrap().network_pricing_schedule,
            policy
        );
        assert_eq!(
            options
                .tree_options()
                .unwrap()
                .node
                .network_pricing_schedule,
            policy
        );
        assert_eq!(
            serde_json::to_value(options).unwrap()["network_pricing_schedule"],
            name
        );
    }
    assert_eq!(
        SolveOptions::from_json("{}")
            .unwrap()
            .network_pricing_schedule,
        NetworkPricingSchedule::FullSweep
    );
    assert!(SolveOptions::from_json(r#"{"network_pricing_schedule":"unknown"}"#).is_err());
}
#[test]
fn multiplicity_json_round_trips_unbounded_upper_without_clipping_finite_counts() {
    let fixture = include_str!("../../../../../examples/networks/triangle.json");
    let mut model = Model::from_json(fixture).unwrap();
    for upper in [1e13, f64::INFINITY] {
        model.networks[0].multiplicity = [0., upper];
        let text = serde_json::to_string(&model).unwrap();
        let parsed = Model::from_json(&text).unwrap();
        assert_eq!(parsed.networks[0].multiplicity, [0., upper]);
        assert_eq!(
            parsed.problem::<2>().unwrap().networks[0]
                .multiplicity
                .upper,
            upper
        );
    }
    let mut bad = serde_json::to_value(&model).unwrap();
    bad["networks"][0]["multiplicity"][0] = serde_json::Value::Null;
    assert!(Model::from_json(&bad.to_string()).is_err());
    for bounds in [[-1., 1.], [2., 1.]] {
        bad["networks"][0]["multiplicity"] = serde_json::json!(bounds);
        assert!(Model::from_json(&bad.to_string())
            .unwrap()
            .problem::<2>()
            .is_err());
    }
}
#[test]
fn model_json_preserves_f64_costs_at_representable_boundaries() {
    let fixture = include_str!("../../../../../examples/networks/triangle.json");
    let mut model = Model::from_json(fixture).unwrap();
    for bits in [
        0x3ff0000000000001,
        0x3fb999999999999a,
        0x4340000000000001,
        0x0000000000000001,
    ] {
        let expected = f64::from_bits(bits);
        model.networks[0].arcs[0].cost = expected;
        let text = serde_json::to_string(&model).unwrap();
        let parsed = Model::from_json(&text).unwrap();
        assert_eq!(parsed.networks[0].arcs[0].cost.to_bits(), bits);
    }
}
