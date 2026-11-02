use super::*;
use serde_json::{json, Value};

fn generate(quota: usize, families: &[(f64, usize)]) -> (Vec<Vec<(usize, usize)>>, f64, Value) {
    let mut priorities: Vec<_> = families.iter().map(|&(p, _)| p).collect();
    priorities.sort_by(|a, b| b.total_cmp(a));
    priorities.dedup();
    let mut calls = Vec::new();
    let (batches, cutoff) = fresh_family_batches(quota, &priorities, None, |priority, quota| {
        families
            .iter()
            .enumerate()
            .filter(|(_, (p, _))| *p == priority)
            .map(|(family, &(_, count))| {
                calls.push(json!({"family":family,"quota":quota,"created":count.min(quota)}));
                (0..count.min(quota))
                    .map(|member| (family, member))
                    .collect()
            })
            .collect()
    });
    (batches, cutoff, json!(calls))
}

#[test]
fn original_strong_callsite_generation_and_allocation_match_72_cases() {
    let fixture: Value =
        serde_json::from_str(include_str!("family_allocation/quota-comparison.json")).unwrap();
    assert_eq!(
        fixture["binary_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 72);
    for case in cases {
        let quota = case["quota"].as_u64().unwrap() as usize;
        let families: Vec<_> = case["families"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| (f[0].as_f64().unwrap(), f[1].as_u64().unwrap() as usize))
            .collect();
        let (batches, cutoff, calls) = generate(quota, &families);
        assert_eq!(calls, case["generation"]["calls"], "{case}");
        assert_eq!(cutoff, case["generation"]["cutoff"].as_f64().unwrap());
        assert_eq!(
            batches.iter().map(Vec::len).sum::<usize>(),
            case["generation"]["count"].as_u64().unwrap() as usize
        );
        let selected = allocate_family_batches(batches, quota, |_| false);
        assert_eq!(json!(selected), case["selection"]["selected"], "{case}");
        assert_eq!(
            quota - selected.len(),
            case["selection"]["quota"].as_u64().unwrap() as usize
        );
    }
}

#[test]
fn original_history_name_exclusion_and_fresh_duplicate_controls_match() {
    let cases: Value = serde_json::from_str(include_str!(
        "family_allocation/quota-duplicate-controls.json"
    ))
    .unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 3);
    for case in cases.as_array().unwrap() {
        let history = case["history_seed"].as_bool().unwrap();
        let duplicate = case["duplicate_fresh_names"].as_bool().unwrap();
        let (batches, _, _) = generate(4, &[(10., 4), (10., 4)]);
        let quota = 4 - usize::from(history);
        let selected = allocate_family_batches(batches, quota, |&candidate| {
            history && (duplicate || candidate == (0, 0))
        });
        assert_eq!(json!(selected), case["observed"]["selected"]);
        assert_eq!(
            quota - selected.len(),
            case["observed"]["quota"].as_u64().unwrap() as usize
        );
    }
}

#[test]
fn original_zero_quota_and_empty_family_controls_match() {
    let cases: Value =
        serde_json::from_str(include_str!("family_allocation/quota-empty-controls.json")).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 2);
    for case in cases.as_array().unwrap() {
        let quota = case["quota"].as_u64().unwrap() as usize;
        let count = if quota == 0 { 3 } else { 0 };
        let (batches, cutoff, calls) = generate(quota, &[(10., count), (10., count)]);
        assert_eq!(calls, case["observed"]["calls"]);
        assert_eq!(cutoff, case["observed"]["cutoff"].as_f64().unwrap());
        assert!(batches.is_empty());
        assert!(allocate_family_batches(batches, quota, |_| false).is_empty());
    }
}

#[test]
fn adaptive_budget_is_applied_before_every_family_in_a_bucket() {
    let mut budget = CandidateBudget::new(
        4,
        0.5,
        super::super::SubtreeEstimate {
            size: 2.,
            priority: 11.,
        },
    );
    let mut seen = Vec::new();
    let (batches, cutoff) =
        fresh_family_batches(budget.quota, &[10.], Some(&mut budget), |_, quota| {
            seen.push(quota);
            vec![vec![0; quota], vec![1; quota]]
        });
    assert_eq!(seen, [4]);
    assert_eq!(cutoff, 10.);
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [4, 4]);
    assert_eq!(allocate_family_batches(batches, 4, |_| false), [0, 0, 1, 1]);
}
