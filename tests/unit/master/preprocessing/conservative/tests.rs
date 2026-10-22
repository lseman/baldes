use super::*;
#[test]
fn signed_propagation_preserves_all_exact_grid_solutions() {
    let constraints = [
        Constraint {
            entries: vec![(0, 2.), (1, -1.)],
            sense: RowSense::Equal,
            rhs: 1.,
        },
        Constraint {
            entries: vec![(1, 1.), (2, 1.)],
            sense: RowSense::LessEqual,
            rhs: 2.,
        },
    ];
    let mut bounds = [Bounds {
        lower: 0.,
        upper: 4.,
    }; 3];
    let report = propagate_conservative(&mut bounds, &constraints, 10000).unwrap();
    assert!(report.complete && !report.infeasible && report.bound_updates > 0);
    for x in 0..=16 {
        for y in 0..=16 {
            for z in 0..=16 {
                let q = [x as f64 / 4., y as f64 / 4., z as f64 / 4.];
                if 2. * q[0] - q[1] == 1. && q[1] + q[2] <= 2. {
                    assert!(q
                        .iter()
                        .zip(bounds)
                        .all(|(&v, b)| b.lower <= v && v <= b.upper));
                }
            }
        }
    }
    let report = propagate_conservative(&mut bounds, &constraints, 0).unwrap();
    assert!(!report.complete && !report.infeasible);
}
#[test]
fn cancellation_does_not_invent_a_zero_bound() {
    let mut bounds = [
        Bounds {
            lower: 1e16,
            upper: 1e16,
        },
        Bounds {
            lower: 1.,
            upper: 1.,
        },
        Bounds {
            lower: 0.,
            upper: 2.,
        },
    ];
    // x-y+z <= 1e16 permits z=1. Subtracting a rounded full activity
    // instead of outward prefix/suffix sums could incorrectly infer zero.
    let constraints = [Constraint {
        entries: vec![(0, 1.), (1, -1.), (2, 1.)],
        sense: RowSense::LessEqual,
        rhs: 1e16,
    }];
    let report = propagate_conservative(&mut bounds, &constraints, 1000).unwrap();
    assert!(!report.infeasible && bounds[2].upper >= 1.);
}
