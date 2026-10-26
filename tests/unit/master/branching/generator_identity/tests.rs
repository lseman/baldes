use super::*;
use crate::master::{branching::AccumulatedResourceGroupBranch, RowSense};
#[test]
fn labels_follow_both_sides_and_exact_resource_identities() {
    let mut names = BranchGeneratorNames::default();
    let down = BranchConstraint::ArcFlow {
        arc: 17,
        sense: RowSense::LessEqual,
        rhs: 0.,
    };
    let up = BranchConstraint::ArcFlow {
        arc: 17,
        sense: RowSense::GreaterEqual,
        rhs: 1.,
    };
    names.insert(&down, "original_x_17");
    assert_eq!(names.get(&up), Some("original_x_17"));
    assert_eq!(names.remove(&up), Some("original_x_17".into()));
    assert_eq!(names.get(&down), None);
    let a = AccumulatedResourceGroupBranch::new(1, vec![7], 0, 2., false).unwrap();
    let b = AccumulatedResourceGroupBranch::new(
        1,
        vec![7],
        0,
        f64::from_bits(2f64.to_bits() + 1),
        false,
    )
    .unwrap();
    let ca = BranchConstraint::AccumulatedResourceGroup(a.clone());
    let cb = BranchConstraint::AccumulatedResourceGroup(b);
    names.insert(&ca, "same_display");
    names.insert(&cb, "same_display");
    assert_eq!(names.entries.len(), 2);
    assert_eq!(
        names.get(&BranchConstraint::AccumulatedResourceGroup(
            a.with_upper(true)
        )),
        Some("same_display")
    );
    names.remove(&ca);
    assert_eq!(names.get(&cb), Some("same_display"));
}

#[test]
fn registered_aggregate_flow_uses_its_original_name_for_history_ties() {
    let constraint = BranchConstraint::AggregateFlow {
        name: "x_1_20_1_OspV".into(),
        arcs: vec![(3, 1.), (9, -1.)],
        sense: RowSense::LessEqual,
        rhs: 0.,
    };
    let names = BranchGeneratorNames::default();
    assert_eq!(names.label(&constraint), "x_1_20_1_OspV");

    let mut names = BranchGeneratorNames::default();
    names.insert(&constraint, "explicit_override");
    assert_eq!(names.label(&constraint), "explicit_override");
}
