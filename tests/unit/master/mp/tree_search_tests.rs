use super::{
    native_strategy2_bound_values, native_strategy2_less, NativeIntegerBoundRounding,
    NativeStrategy2Key,
};

#[test]
fn native_strategy2_uses_guarded_integer_rounded_numerical_bound() {
    let almost_above_integer = 1.000_000_5;
    assert_eq!(
        native_strategy2_bound_values(
            almost_above_integer,
            10.0,
            NativeIntegerBoundRounding::GuardedTolerance,
        ),
        1.0
    );
    assert_eq!(
        native_strategy2_bound_values(
            almost_above_integer,
            10.0,
            NativeIntegerBoundRounding::ExactCeil,
        ),
        2.0
    );
    assert_eq!(
        native_strategy2_bound_values(
            f64::NEG_INFINITY,
            3.25,
            NativeIntegerBoundRounding::GuardedTolerance,
        ),
        4.0
    );
    assert_eq!(
        native_strategy2_bound_values(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            NativeIntegerBoundRounding::GuardedTolerance,
        ),
        -1e12
    );
}

#[test]
fn native_strategy2_uses_depth_parent_rank_then_creation_serial_for_ties() {
    let shallow = NativeStrategy2Key {
        dual_bound: 2.0,
        depth: 1,
        parent_branch_order: 1,
        serial: 1,
    };
    let deep = NativeStrategy2Key {
        depth: 2,
        ..shallow
    };
    assert!(native_strategy2_less(shallow, deep));
    assert!(!native_strategy2_less(deep, shallow));

    let later_parent = NativeStrategy2Key {
        depth: 2,
        parent_branch_order: 2,
        serial: 2,
        ..shallow
    };
    let earlier_parent = NativeStrategy2Key {
        parent_branch_order: 1,
        ..later_parent
    };
    assert!(native_strategy2_less(later_parent, earlier_parent));
    assert!(!native_strategy2_less(earlier_parent, later_parent));

    let later_node = NativeStrategy2Key {
        serial: 3,
        ..earlier_parent
    };
    let earlier_node = NativeStrategy2Key {
        serial: 2,
        ..earlier_parent
    };
    assert!(native_strategy2_less(later_node, earlier_node));
    assert!(!native_strategy2_less(earlier_node, later_node));
}
