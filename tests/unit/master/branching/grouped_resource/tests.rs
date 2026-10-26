use super::*;
use crate::master::tree::candidates::resource_groups::tests::fixture;

#[test]
fn blocked_members_remain_optional_and_old_row_decompositions_are_frozen() {
    let (mut problem, routes, arcs) = fixture();
    let packing =
        problem.model.networks[0].arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
    let upper =
        AccumulatedResourceGroupBranch::new(packing, arcs.clone(), 4, 1. + 1e-6, true).unwrap();
    let row = problem.model.rows.len();
    problem
        .branch(BranchConstraint::AccumulatedResourceGroup(upper.clone()))
        .unwrap();
    let value = |p: &RoutingContext<5>, r: &Route, row| {
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
    let packing =
        problem.model.networks[0].arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1., false).unwrap();
    group.validate(&problem).unwrap();
    problem.model.rows[packing].rhs = 2.;
    assert!(group.validate(&problem).is_err());
    problem.model.rows[packing].rhs = 1.;
    problem.model.networks[0].arc_entries[0].push((packing, -0.25));
    assert!(group.validate(&problem).is_err());
    assert!(AccumulatedResourceGroupBranch::new(0, vec![1, 1], 0, 1., false).is_err());
}

#[test]
fn repeated_group_violations_have_one_coefficient_including_blocked_arcs() {
    let (mut problem, routes, arcs) = fixture();
    let packing =
        problem.model.networks[0].arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1. + 1e-6, true).unwrap();
    let row = problem.model.rows.len();
    problem
        .branch(BranchConstraint::AccumulatedResourceGroup(group))
        .unwrap();
    let mut repeated = routes[0].0.clone();
    repeated.arcs.extend_from_within(..);
    // Concatenate head states as well as arcs; the first profile entry is root.
    let heads = repeated.resource_profile[1..].to_vec();
    repeated.resource_profile.extend(heads);
    let branch = problem.branches().last().unwrap();
    assert_eq!(branch.coefficient(&problem, &repeated), 1.);
    let coefficient: f64 = problem
        .base_entries(&repeated)
        .unwrap()
        .iter()
        .filter(|(r, _)| *r == row)
        .map(|(_, c)| c)
        .sum();
    assert_eq!(coefficient, 1.);
    if !crate::master::branching::enforcement::ACCUMULATED_RESOURCE_IN_PRICING {
        assert!(problem.model.networks[0]
            .arc_entries
            .iter()
            .flatten()
            .all(|(r, _)| *r != row));
    }
}

#[test]
#[cfg(feature = "accumulated-resource-master")]
fn empty_single_arc_window_keeps_master_only_identity_and_no_arc_entries() {
    let (mut problem, _, arcs) = fixture();
    let row = problem.model.rows.len();
    problem
        .branch(BranchConstraint::AccumulatedResource(
            AccumulatedResourceBranch {
                arc: arcs[0],
                resource: 4,
                threshold: 100.,
                upper: true,
            },
        ))
        .unwrap();
    assert!(matches!(
        problem.branches().last(),
        Some(BranchConstraint::AccumulatedResource(_))
    ));
    assert!(problem.model.networks[0]
        .arc_entries
        .iter()
        .flatten()
        .all(|(r, _)| *r != row));
}
