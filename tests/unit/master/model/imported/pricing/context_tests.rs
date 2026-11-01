use super::candidate_tests::{bounds, column, session};
use super::*;

fn assert_bounds(actual: Bounds, lower: f64, upper: f64) {
    assert_eq!((actual.lower, actual.upper), (lower, upper));
}

#[test]
fn import_pricing_exposes_child_bounds_without_conflating_pattern_and_aggregate_domains() {
    let parent = session()
        .with_envelope(
            7,
            ColumnEnvelope {
                cost_lower: 1.,
                coordinates: vec![(17, bounds(0., 1.)), (0, bounds(-1., 0.))],
            },
        )
        .unwrap();
    let child = parent
        .with_variable_bounds(0, bounds(-1., 1.))
        .unwrap()
        .with_variable_bounds(0, bounds(-10., 0.5))
        .unwrap()
        .with_multiplicity(7, bounds(0., 3.))
        .unwrap()
        .with_column_bounds(42, 2, bounds(0., 2.))
        .unwrap();
    for point in [
        child.pricing(7, Options::default()).unwrap(),
        child.feasibility_pricing(7, Options::default()).unwrap(),
    ] {
        assert_bounds(point.original_bounds(0).unwrap(), -1., 0.5);
        assert_bounds(point.original_bounds(17).unwrap(), -100., 100.);
        assert_bounds(point.multiplicity_bounds(), 1., 3.);
        assert_bounds(point.column_bounds(2).unwrap(), 0., f64::INFINITY);
        // This pattern box remains [-1,0], even though aggregate x may be
        // positive through another group. Neither bound is inferred from LPs.
        let envelope = point.envelope().unwrap();
        assert_bounds(envelope.coordinates[1].1, -1., 0.);
        assert_eq!(envelope.cost_lower, 1.);
        let other = point.for_subproblem(42).unwrap();
        assert_bounds(other.column_bounds(2).unwrap(), 0., 2.);
        assert_bounds(other.multiplicity_bounds(), 0., 9.);
        assert!(other.envelope().is_none());
        assert!(point.original_bounds(999).is_err());
        assert!(point.column_bounds(999).is_err());
    }
    let parent_point = parent.pricing(7, Options::default()).unwrap();
    assert_bounds(parent_point.original_bounds(0).unwrap(), -2., 2.);
    assert_bounds(parent_point.multiplicity_bounds(), 1., 4.);
    assert_bounds(
        parent_point
            .for_subproblem(42)
            .unwrap()
            .column_bounds(2)
            .unwrap(),
        0.,
        3.,
    );
}

#[test]
fn import_pricing_context_survives_extension_and_does_not_publish_finite_domain_deductions() {
    let parent = session();
    let child = parent
        .with_variable_bounds(0, bounds(-1., 1.))
        .unwrap()
        .with_multiplicity(7, bounds(1., 2.))
        .unwrap();
    let view = child.certificate_view().unwrap();
    let mut new_column = column(77, 1., vec![]);
    new_column.bounds = bounds(0., 0.5);
    let extended = view
        .accept(ColumnBatch {
            columns: vec![new_column],
            complete: false,
        })
        .unwrap();
    let point = extended.pricing(Options::default()).unwrap();
    assert_bounds(point.original_bounds(0).unwrap(), -1., 1.);
    assert_bounds(point.original_bounds(17).unwrap(), -100., 100.);
    assert_bounds(point.multiplicity_bounds(), 1., 2.);
    assert_bounds(point.column_bounds(77).unwrap(), 0., 0.5);
    assert!(child
        .pricing(7, Options::default())
        .unwrap()
        .column_bounds(77)
        .is_err());
    assert_bounds(
        parent
            .pricing(7, Options::default())
            .unwrap()
            .multiplicity_bounds(),
        1.,
        4.,
    );
}
