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
