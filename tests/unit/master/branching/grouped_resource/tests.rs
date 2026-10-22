use super::*;
use crate::master::tree::candidates::resource_groups::tests::fixture;

#[test]
fn blocked_members_remain_optional_and_old_row_decompositions_are_frozen() {
    let (mut problem, routes, arcs) = fixture();
    let packing = problem.arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
    let upper =
        AccumulatedResourceGroupBranch::new(packing, arcs.clone(), 4, 1. + 1e-6, true).unwrap();
    let row = problem.rows.len();
    problem
        .branch(BranchConstraint::AccumulatedResourceGroup(upper.clone()))
        .unwrap();
    let value = |p: &RootProblem<5>, r: &Route, row| {
        p.base_entries(r)
            .unwrap()
            .iter()
            .filter(|(i, _)| *i == row)
            .map(|(_, c)| c)
            .sum::<f64>()
    };
    assert_eq!(value(&problem, &routes[0].0, row), 1.);
    assert_eq!(value(&problem, &routes[1].0, row), 0.);
    let lower = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 2. + 1e-6, false).unwrap();
    problem
        .branch(BranchConstraint::AccumulatedResourceGroup(lower))
        .unwrap();
    assert_eq!(value(&problem, &routes[0].0, row), 1.);
    assert_eq!(value(&problem, &routes[1].0, row), 0.);
    assert_eq!(value(&problem, &routes[1].0, row + 1), 1.);
}

#[test]
fn group_disjunction_requires_an_actual_unit_packing_proof() {
    let (mut problem, _, arcs) = fixture();
    let packing = problem.arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1., false).unwrap();
    group.validate(&problem).unwrap();
    problem.rows[packing].rhs = 2.;
    assert!(group.validate(&problem).is_err());
    problem.rows[packing].rhs = 1.;
    problem.arc_entries[0].push((packing, -0.25));
    assert!(group.validate(&problem).is_err());
    assert!(AccumulatedResourceGroupBranch::new(0, vec![1, 1], 0, 1., false).is_err());
}
