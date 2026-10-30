use super::*;
#[test]
fn local_callback_bounds_differ_from_aggregate_rows_under_multiplicity() {
    // owner-bound-propagation-extended: local input -4,100, aggregate0,100.
    let mut optional = OwnerBounds::new(
        Some([-4., 100.]),
        Bounds {
            lower: 0.,
            upper: 2.,
        },
        [0., 0.],
    );
    assert_eq!(
        optional.aggregate,
        Bounds {
            lower: 0.,
            upper: 100.
        }
    );
    optional.update(
        optional.aggregate,
        Bounds {
            lower: 0.,
            upper: 2.,
        },
    );
    assert_eq!(optional.local.lower, -4.);
    let mut singleton = OwnerBounds::new(
        Some([-4., 100.]),
        Bounds {
            lower: 0.,
            upper: 1.,
        },
        [0., 0.],
    );
    singleton.update(
        singleton.aggregate,
        Bounds {
            lower: 0.,
            upper: 1.,
        },
    );
    assert_eq!(singleton.local.lower, 0.);
    // A positive aggregate bound need not induce the unsupported local bound.
    optional.update(
        Bounds {
            lower: 4.,
            upper: 100.,
        },
        Bounds {
            lower: 0.,
            upper: 2.,
        },
    );
    assert_eq!(optional.local.lower, -4.);
}
#[test]
fn fixed_selected_contributions_reduce_aggregate_bounds_and_preserve_local_input() {
    // fixVariableValue subtracts fixed activity from both global bounds;
    // local callback bounds are subsequently derived with remaining multiplicity.
    let mut owner = OwnerBounds::new(
        Some([0., 100.]),
        Bounds {
            lower: 0.,
            upper: 1.,
        },
        [3., 3.],
    );
    assert_eq!(
        owner.aggregate,
        Bounds {
            lower: 0.,
            upper: 97.
        }
    );
    assert_eq!(owner.local.upper, 100.);
    owner.update(
        owner.aggregate,
        Bounds {
            lower: 0.,
            upper: 1.,
        },
    );
    assert_eq!(owner.local.upper, 97.);
}
#[test]
fn missing_input_is_unbounded_and_unbounded_upper_cannot_bypass_callback_rejection() {
    let owner = OwnerBounds::new(
        None,
        Bounds {
            lower: 0.,
            upper: 1.,
        },
        [0., 0.],
    );
    assert_eq!(
        owner.aggregate,
        Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY
        }
    );
    // Native initVars maps +inf to its 1e12 infinity sentinel before callback.
    assert!(
        crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported([
            4.,
            f64::INFINITY
        ])
    );
    assert!(
        !crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported([
            0.,
            f64::INFINITY
        ])
    );
}
