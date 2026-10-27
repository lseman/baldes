use super::*;
use crate::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, PreparedClampFilter, PreparedSlopeRelationArc, SlopeFront,
};

fn case() -> (CoupledResource, serde_json::Value) {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-paired-feasibility-preparation.json"
    ))
    .unwrap();
    let case = data["cases"][6].clone();
    let definition =
        serde_json::from_value(case["model"]["networks"][0]["coupled_resources"][0].clone())
            .unwrap();
    (definition, case)
}
fn bits(pairs: &[(f64, f64)]) -> Vec<(u64, u64)> {
    pairs
        .iter()
        .map(|&(a, b)| (a.to_bits(), b.to_bits()))
        .collect()
}

#[test]
fn rounded_affine_lower_maximum_matches_the_actual_modern_forward_state() {
    let (definition, case) = case();
    let CoupledResource::SlopePaired { initial, arcs, .. } = &definition else {
        panic!()
    };
    let front = SlopeFront::from_prepared(initial.clone());
    let relation = PreparedSlopeRelationArc::new(&arcs[1], NATIVE).unwrap();
    let FrontierOutcome::Complete(result) = relation.extend(&front, FrontierBudget::default())
    else {
        panic!()
    };
    let event = case["reference"]["forward_extensions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["arc"] == 1 && event["returned"] == true)
        .unwrap();
    let expected: Vec<(f64, f64)> =
        serde_json::from_value(event["after"]["pairs"].clone()).unwrap();
    assert_eq!(event["after"]["tags"], 0);
    assert_eq!(result.components().len(), 1);
    assert_eq!(bits(result.components()[0].pairs()), bits(&expected));
    assert!(result.used_tolerant_alignment());
    let next = PreparedSlopeRelationArc::new(&arcs[2], NATIVE).unwrap();
    let FrontierOutcome::Complete(final_front) = next.extend(&result, FrontierBudget::default())
    else {
        panic!()
    };
    let event = case["reference"]["forward_extensions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["arc"] == 2 && event["returned"] == true)
        .unwrap();
    let mut expected: Vec<(f64, f64)> =
        serde_json::from_value(event["after"]["pairs"].clone()).unwrap();
    // The binary encodes a point with a redundant zero-length final segment;
    // the public prepared-function representation stores just the point.
    assert_eq!(expected, [(6.5, 2.), (0., 2.)]);
    expected.truncate(1);
    assert_eq!(final_front.components().len(), 1);
    assert_eq!(bits(final_front.components()[0].pairs()), bits(&expected));
}

#[test]
fn rounded_maximum_keeps_strict_domain_and_budget_guards() {
    let (definition, _) = case();
    let CoupledResource::SlopePaired { arcs, .. } = definition else {
        panic!()
    };
    let input = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 1.), (1., 7.)]).unwrap(),
    );
    let clamp = PreparedClampFilter::new(&arcs[1].lower, &arcs[1].upper).unwrap();
    assert!(matches!(
        clamp.apply(&input, STRICT, FrontierBudget::default()),
        FrontierOutcome::Unknown(_)
    ));
    for budget in [
        FrontierBudget {
            max_operations: 0,
            ..FrontierBudget::default()
        },
        FrontierBudget {
            max_pairs: 1,
            ..FrontierBudget::default()
        },
        FrontierBudget {
            max_elapsed: Some(std::time::Duration::ZERO),
            ..FrontierBudget::default()
        },
    ] {
        assert!(matches!(
            clamp.apply(&input, NATIVE, budget),
            FrontierOutcome::Unknown(_)
        ));
    }
}

#[test]
fn an_endpoint_witness_proves_feasibility_without_promoting_an_objective_or_bad_domain() {
    let (definition, _) = case();
    assert!(full_certify(&definition, &[1, 2], 1., &mut 1_000_000, None).is_none());
    assert_eq!(
        certify(&definition, &[1, 2], 1., &mut 1_000_000, None),
        Some((true, [0., 0.]))
    );
    let mut cost = definition.clone();
    let CoupledResource::SlopePaired { mode, .. } = &mut cost else {
        panic!()
    };
    *mode = PairedResourceMode::CostUnbounded;
    assert!(left_endpoint_feasible(&cost, &[1, 2], 1., &mut 1_000_000, None).is_none());
    let mut clipped = definition.clone();
    let CoupledResource::SlopePaired { initial_base, .. } = &mut clipped else {
        panic!()
    };
    initial_base.lower = 1.;
    assert!(left_endpoint_feasible(&clipped, &[1, 2], 1., &mut 1_000_000, None).is_none());
    let mut infeasible = definition.clone();
    let CoupledResource::SlopePaired { arcs, .. } = &mut infeasible else {
        panic!()
    };
    arcs[1].upper = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 1.), (0., 7.)]).unwrap();
    assert!(left_endpoint_feasible(&infeasible, &[1, 2], 1., &mut 1_000_000, None).is_none());
    assert!(certify(&definition, &[1, 2], 1., &mut 0, None).is_none());
}
