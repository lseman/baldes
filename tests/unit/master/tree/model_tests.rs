use super::{
    bound_priority_bound, bound_priority_less, bound_priority_values, phased_fallback_selection,
    BoundPriorityKey, IntegerBoundRounding, TreeFrontier, TreeSearchPolicy,
};

fn checkpoint() -> crate::master::model::NodeSnapshot<2> {
    let problem = crate::master::model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = crate::master::model::Options::default();
    options.cg.stabilize = false;
    crate::master::model::solve(problem, options)
        .unwrap()
        .snapshot
}

fn checkpoint_output() -> crate::master::model::Output<2> {
    let problem = crate::master::model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = crate::master::model::Options::default();
    options.cg.stabilize = false;
    crate::master::model::solve(problem, options).unwrap()
}

#[test]
fn native_strategy3_prefers_depth_then_higher_bound_but_keeps_normal_overflow() {
    assert_eq!(
        TreeSearchPolicy::parse("depth-bound-priority").unwrap(),
        TreeSearchPolicy::DepthBoundPriority
    );
    let root = checkpoint();
    for open_limit in [0, 1000] {
        let mut frontier = TreeFrontier::new(
            TreeSearchPolicy::DepthBoundPriority,
            open_limit,
            IntegerBoundRounding::Unrounded,
        );
        for (depth, bound) in [(1, 0.5), (1, 1.25), (2, 0.25)] {
            let mut node = root.fork().unwrap();
            node.state.node_depth = depth;
            node.warm.bound = bound;
            node.warm.numerical_bound = bound;
            frontier.push(node, 1).unwrap();
        }
        assert_eq!(frontier.pop().unwrap().unwrap().node.state.node_depth, 2);
        let remaining: Vec<_> = (0..2)
            .map(|_| frontier.pop().unwrap().unwrap().node.warm.numerical_bound)
            .collect();
        assert_eq!(
            remaining,
            if open_limit == 0 {
                vec![0.5, 1.25]
            } else {
                vec![1.25, 0.5]
            }
        );
    }
}

#[test]
fn native_frontiers_rank_by_numerical_channel_in_certified_mode() {
    let root = checkpoint();
    for policy in [
        TreeSearchPolicy::Default,
        TreeSearchPolicy::BestBound,
        TreeSearchPolicy::BoundPriority,
    ] {
        for open_limit in [0, 1, 1000] {
            let mut frontier =
                TreeFrontier::new(policy, open_limit, IntegerBoundRounding::Unrounded);
            // Both channels are valid lower bounds for this 1.5 LP. Their
            // preferences disagree, exposing accidental certificate ranking.
            for (numerical, certified) in [(1.25, 0.5), (0.75, 1.)] {
                let mut node = root.fork().unwrap();
                node.state.node_depth = 1;
                node.warm.bound_mode = crate::master::cg::BoundMode::Certified;
                node.warm.bound = certified;
                node.warm.certified_bound = certified;
                node.warm.numerical_bound = numerical;
                frontier.push(node, 1).unwrap();
            }
            let first = frontier.pop().unwrap().unwrap();
            assert_eq!(first.node.warm.numerical_bound, 0.75);
            assert_eq!(first.node.warm.bound, 1.);
            assert_eq!(first.node.warm.certified_bound, 1.);
            assert_eq!(
                frontier.pop().unwrap().unwrap().node.warm.numerical_bound,
                1.25
            );
        }
    }
}

#[test]
fn native_strategy2_parent_rank_follows_treatment_instead_of_creation_order() {
    let root = checkpoint();
    let mut frontier = TreeFrontier::new(
        TreeSearchPolicy::BoundPriority,
        1000,
        IntegerBoundRounding::Unrounded,
    );
    for bound in [3., 2.] {
        let mut node = root.fork().unwrap();
        node.state.node_depth = 1;
        node.warm.bound = bound;
        node.warm.numerical_bound = bound;
        frontier.push(node, 1).unwrap();
    }
    let earlier_treated = frontier.pop().unwrap().unwrap();
    let later_treated = frontier.pop().unwrap().unwrap();
    assert!(earlier_treated.serial > later_treated.serial);
    assert!(earlier_treated.branch_order < later_treated.branch_order);

    // Reverse creation order again. With equal bounds and depths, the earlier
    // treated parent must win before the grandchildren's creation serials.
    for parent in [&later_treated, &earlier_treated] {
        let mut child = parent.node.fork().unwrap();
        child.state.node_depth = 2;
        child.warm.bound = 4.;
        child.warm.numerical_bound = 4.;
        frontier.push(child, parent.branch_order).unwrap();
    }
    assert_eq!(
        frontier.pop().unwrap().unwrap().parent_branch_order,
        earlier_treated.branch_order
    );
}

#[test]
fn native_prepared_son_rank_is_reserved_before_ordinary_son_treatment() {
    let mut prepared = checkpoint_output();
    prepared.snapshot.state.node_depth = 1;
    prepared.snapshot.warm.bound = 3.;
    prepared.snapshot.warm.numerical_bound = 3.;
    let prepared_node = prepared.snapshot.fork().unwrap();
    let mut prepared_sibling = checkpoint_output();
    prepared_sibling.snapshot.state.node_depth = 1;
    prepared_sibling.snapshot.warm.bound = 4.;
    prepared_sibling.snapshot.warm.numerical_bound = 4.;
    let prepared_sibling_node = prepared_sibling.snapshot.fork().unwrap();

    let mut ordinary_node = checkpoint();
    ordinary_node.state.node_depth = 1;
    ordinary_node.warm.bound = 2.;
    ordinary_node.warm.numerical_bound = 2.;

    let mut frontier = TreeFrontier::new(
        TreeSearchPolicy::BoundPriority,
        1000,
        IntegerBoundRounding::Unrounded,
    );
    frontier
        .push_evaluated(prepared_node, 1, Some(prepared))
        .unwrap();
    frontier.push(ordinary_node, 1).unwrap();
    frontier
        .push_evaluated(prepared_sibling_node, 1, Some(prepared_sibling))
        .unwrap();

    // Bound priority treats the ordinary son first, but both exact trial sons
    // reserved ranks in their native admission order.
    let ordinary = frontier.pop().unwrap().unwrap();
    let prepared = frontier.pop().unwrap().unwrap();
    let prepared_sibling = frontier.pop().unwrap().unwrap();
    assert_eq!(ordinary.node.warm.numerical_bound, 2.);
    assert_eq!(ordinary.branch_order, 4);
    assert_eq!(prepared.node.warm.numerical_bound, 3.);
    assert_eq!(prepared.branch_order, 2);
    assert_eq!(prepared_sibling.node.warm.numerical_bound, 4.);
    assert_eq!(prepared_sibling.branch_order, 3);

    let mut prepared_descendant = prepared.node.fork().unwrap();
    prepared_descendant.state.node_depth = 2;
    prepared_descendant.warm.numerical_bound = 4.;
    let mut prepared_sibling_descendant = prepared_sibling.node.fork().unwrap();
    prepared_sibling_descendant.state.node_depth = 2;
    prepared_sibling_descendant.warm.numerical_bound = 4.;
    frontier
        .push(prepared_descendant, prepared.branch_order)
        .unwrap();
    frontier
        .push(prepared_sibling_descendant, prepared_sibling.branch_order)
        .unwrap();
    assert_eq!(
        frontier.pop().unwrap().unwrap().parent_branch_order,
        prepared.branch_order
    );
}

#[test]
fn native_phased_fallback_admits_floor_ceiling_split_upper_first() {
    let root = checkpoint();
    let mut lower = root.fork().unwrap();
    lower.warm.bound = 0.;
    let mut upper = root.fork().unwrap();
    upper.warm.bound = 1.;

    let selection = phased_fallback_selection(Some((
        vec![lower, upper],
        super::branching::ChildOrder::LowerThenUpper,
    )))
    .unwrap();
    assert_eq!(selection.children.len(), 2);
    assert_eq!(selection.children[0].warm.bound, 1.);
    assert_eq!(selection.children[1].warm.bound, 0.);
}

#[test]
fn native_phased_fallback_preserves_custom_disjunction_order() {
    let root = checkpoint();
    let mut first = root.fork().unwrap();
    first.warm.bound = 0.;
    let mut second = root.fork().unwrap();
    second.warm.bound = 1.;

    let selection = phased_fallback_selection(Some((
        vec![first, second],
        super::branching::ChildOrder::Custom,
    )))
    .unwrap();
    assert_eq!(selection.children.len(), 2);
    assert_eq!(selection.children[0].warm.bound, 0.);
    assert_eq!(selection.children[1].warm.bound, 1.);
}

#[test]
fn native_strategy2_unknown_numerical_channel_does_not_use_certified_bound() {
    let mut node = checkpoint();
    node.warm.numerical_bound = f64::NEG_INFINITY;
    node.warm.certified_bound = 3.25;
    node.warm.bound = 3.25;
    assert_eq!(
        bound_priority_bound(&node, IntegerBoundRounding::Unrounded),
        -1e12
    );
    node.warm.numerical_bound = 1.5;
    assert_eq!(
        bound_priority_bound(&node, IntegerBoundRounding::Unrounded),
        1.5
    );
}

#[test]
fn native_strategy2_uses_guarded_integer_rounded_numerical_bound() {
    let almost_above_integer = 1.000_000_5;
    assert_eq!(
        IntegerBoundRounding::default(),
        IntegerBoundRounding::Unrounded
    );
    assert_eq!(
        bound_priority_values(1.5, IntegerBoundRounding::Unrounded),
        1.5
    );
    assert_eq!(
        bound_priority_values(almost_above_integer, IntegerBoundRounding::GuardedTolerance,),
        1.0
    );
    assert_eq!(
        bound_priority_values(almost_above_integer, IntegerBoundRounding::ExactCeil,),
        2.0
    );
    assert_eq!(
        bound_priority_values(f64::NEG_INFINITY, IntegerBoundRounding::GuardedTolerance,),
        -1e12
    );
    assert_eq!(
        bound_priority_values(f64::INFINITY, IntegerBoundRounding::Unrounded,),
        1e12
    );
}

#[test]
fn native_strategy2_uses_depth_parent_rank_then_creation_serial_for_ties() {
    let shallow = BoundPriorityKey {
        dual_bound: 2.0,
        depth: 1,
        parent_branch_order: 1,
        serial: 1,
    };
    let deep = BoundPriorityKey {
        depth: 2,
        ..shallow
    };
    assert!(bound_priority_less(shallow, deep));
    assert!(!bound_priority_less(deep, shallow));

    let later_parent = BoundPriorityKey {
        depth: 2,
        parent_branch_order: 2,
        serial: 2,
        ..shallow
    };
    let earlier_parent = BoundPriorityKey {
        parent_branch_order: 1,
        ..later_parent
    };
    assert!(bound_priority_less(later_parent, earlier_parent));
    assert!(!bound_priority_less(earlier_parent, later_parent));

    let later_node = BoundPriorityKey {
        serial: 3,
        ..earlier_parent
    };
    let earlier_node = BoundPriorityKey {
        serial: 2,
        ..earlier_parent
    };
    assert!(bound_priority_less(later_node, earlier_node));
    assert!(!bound_priority_less(earlier_node, later_node));
}

#[test]
fn strategy2_keeps_a_proven_empty_child_closed_when_numerical_channel_is_unknown() {
    use crate::master::{cg::BoundMode, model::Options, preprocessing::Bounds};
    let root = checkpoint();
    let node = root
        .with_multiplicity(
            7,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap()
        .with_multiplicity(
            42,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    let mut options = Options::default();
    options.cg.bound_mode = BoundMode::Certified;
    options.cg.stabilize = false;
    options.cg.max_penalty_updates = 0;
    let evaluated = node.solve(options).unwrap();
    assert!(super::model_strong::proven_infeasible(&evaluated));
    let mut child = evaluated.snapshot.fork().unwrap();
    assert_eq!(child.warm.bound_mode, BoundMode::Certified);
    assert!(child.state.phase_one_infeasible);
    child.warm.numerical_bound = f64::NEG_INFINITY;
    let mut frontier = TreeFrontier::new(
        TreeSearchPolicy::BoundPriority,
        1000,
        IntegerBoundRounding::Unrounded,
    );
    frontier.push_evaluated(child, 1, Some(evaluated)).unwrap();
    let selected = frontier.pop().unwrap().unwrap();
    assert!(selected.proven_infeasible());
    assert_eq!(selected.strategy2_dual_bound, 1e12);
    assert_eq!(selected.numerical_bound(), f64::INFINITY);
}
