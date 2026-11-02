use super::*;

fn keys(values: &serde_json::Value) -> Vec<ordering::PriorityKey> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|key| ordering::PriorityKey {
            family: key["family"].as_u64().unwrap() as usize,
            family_priority: key["outer"].as_f64().unwrap(),
            rule: key["rule"].as_u64().unwrap() as u8,
            variable_priority: key["priority"].as_f64().unwrap(),
            added_level: key["level"].as_i64().unwrap() as i32,
        })
        .collect()
}

#[test]
fn registered_family_publication_matches_original_at_each_requested_quota() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("component_family_publication_capture.json")).unwrap();
    assert_eq!(
        fixture["binary_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 1008);
    for case in fixture["cases"].as_array().unwrap() {
        let publication = ComponentPublication {
            keys: keys(&case["inputs"]),
        };
        let selected = publication
            .publish(case["quota"].as_u64().unwrap() as usize, None)
            .unwrap();
        assert_eq!(serde_json::json!(selected), case["native"], "{case}");
    }
}

#[test]
fn shared_publisher_compatibility_view_keeps_all_120_original_controls() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("component_publication_capture.json")).unwrap();
    assert_eq!(fixture.as_array().unwrap().len(), 120);
    for case in fixture.as_array().unwrap() {
        let keys = keys(&case["inputs"]);
        let mut work = ordering::Work::new(None);
        let selected =
            ordering::publish(&keys, case["quota"].as_u64().unwrap() as usize, &mut work).unwrap();
        assert_eq!(serde_json::json!(selected), case["native"], "{case}");
    }
}

#[test]
fn interrupted_publication_never_returns_a_partial_family() {
    let publication = ComponentPublication {
        keys: vec![ordering::PriorityKey {
            family: 0,
            family_priority: 1.,
            rule: 0,
            variable_priority: 1.,
            added_level: 1,
        }],
    };
    assert!(publication
        .publish(1, Some(Instant::now() - std::time::Duration::from_secs(1)))
        .is_none());
}
