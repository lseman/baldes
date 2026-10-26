use super::*;
fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
}
fn line(value: f64, slope: f64, end: f64) -> PreparedSlopeFunction {
    function(&[(value, 0.), (slope, end)])
}
fn complete(outcome: FrontierOutcome<SlopeFront>) -> SlopeFront {
    match outcome {
        FrontierOutcome::Complete(front) => front,
        other => panic!("expected complete front: {other:?}"),
    }
}
fn value(front: &SlopeFront, x: f64) -> Option<f64> {
    front.components().iter().find_map(|component| {
        component
            .evaluate(x, EvaluationPolicy::StrictDomain)
            .unwrap()
    })
}
#[test]
fn general_sloped_lower_and_upper_split_at_all_verified_crossings() {
    let front = SlopeFront::from_prepared(line(0., 0., 4.));
    let lower = line(-1., 1., 4.);
    let upper = line(3., -1., 4.);
    let result = complete(clamp_filter(
        &front,
        &lower,
        &upper,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(result.components().len(), 1);
    assert_eq!(result.components()[0].domain(), (0., 2.));
    for (x, y) in [(0., 0.), (0.5, 0.), (1., 0.), (1.5, 0.5), (2., 1.)] {
        assert_eq!(value(&result, x), Some(y));
    }
    assert_eq!(value(&result, 2.5), None);
    assert!(matches!(
        result.provenance().as_ref(),
        FrontierProvenance::ClampFilter { .. }
    ));
}
#[test]
fn upper_filter_preserves_disconnected_components_and_lower_is_a_clamp() {
    let hill = SlopeFront::from_prepared(function(&[(0., 0.), (2., 1.), (-2., 2.)]));
    let low = line(-100., 0., 2.);
    let high = line(1., 0., 2.);
    let result = complete(clamp_filter(
        &hill,
        &low,
        &high,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    let domains: Vec<_> = result
        .components()
        .iter()
        .map(PreparedSlopeFunction::domain)
        .collect();
    assert_eq!(domains, vec![(0., 0.5), (1.5, 2.)]);
    assert_eq!(value(&result, 1.), None);
    let valley = SlopeFront::from_prepared(function(&[(2., 0.), (-2., 1.), (2., 2.)]));
    let result = complete(clamp_filter(
        &valley,
        &high,
        &line(100., 0., 2.),
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(result.components().len(), 1);
    assert_eq!(result.components()[0].domain(), (0., 2.));
    for (x, y) in [(0., 2.), (0.5, 1.), (1., 1.), (1.5, 1.), (2., 2.)] {
        assert_eq!(value(&result, x), Some(y));
    }
}
#[test]
fn isolated_feasible_point_is_distinct_from_proven_empty() {
    let valley = SlopeFront::from_prepared(function(&[(2., 0.), (-2., 1.), (2., 2.)]));
    let low = line(-100., 0., 2.);
    let high = line(0., 0., 2.);
    let result = complete(clamp_filter(
        &valley,
        &low,
        &high,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(result.components()[0].pairs(), &[(0., 1.)]);
    let above = SlopeFront::from_prepared(line(2., 0., 2.));
    assert!(matches!(
        clamp_filter(
            &above,
            &low,
            &high,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default()
        ),
        FrontierOutcome::Empty
    ));
}
#[test]
fn cached_function_bounds_apply_independently_to_each_input_component() {
    let input = complete(SlopeFront::from_components(
        vec![line(0., 0., 1.), function(&[(0., 2.), (0., 3.)])],
        FrontierBudget::default(),
    ));
    let prepared = PreparedClampFilter::new(&line(-1., 1., 3.), &line(10., 0., 3.)).unwrap();
    let result = complete(prepared.apply(
        &input,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(
        result
            .components()
            .iter()
            .map(PreparedSlopeFunction::domain)
            .collect::<Vec<_>>(),
        vec![(0., 1.), (2., 3.)]
    );
    assert_eq!(value(&result, 2.), Some(1.));
    assert_eq!(value(&result, 3.), Some(2.));
    assert_eq!(value(&result, 1.5), None);
}
#[test]
fn rounding_ambiguity_near_topology_and_work_exhaustion_are_unknown() {
    let input = SlopeFront::from_prepared(line(0.1, 0.3, 2.));
    let lower = line(-100., 0., 2.);
    let upper = line(0.05, 0.5, 2.);
    assert!(matches!(
        clamp_filter(
            &input,
            &lower,
            &upper,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(_)
    ));
    let near = function(&[(0., 0.00001), (0., 2.)]);
    assert!(matches!(
        clamp_filter(
            &input,
            &near,
            &line(100., 0., 2.),
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(_)
    ));
    let exhausted = FrontierBudget {
        max_operations: 0,
        ..Default::default()
    };
    assert!(matches!(
        clamp_filter(
            &input,
            &lower,
            &upper,
            FrontOperationPolicy::StrictDomain,
            exhausted
        ),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
}

#[test]
fn native_point_inside_constant_bounds_keeps_nearby_domain_endpoints() {
    // Original helpers at ELF 0x723340/0x7200c0 retain (value=2,x=1)
    // for these two constant domains. Their count=1 representation adds
    // a zero-length zero-slope segment; the prepared form is one point.
    let front = SlopeFront::from_prepared(function(&[(2., 1.)]));
    for upper_end in [2., 1.00001] {
        let lower = function(&[(0., 0.99999), (0., upper_end)]);
        let upper = function(&[(10., 0.99999), (0., upper_end)]);
        let result = complete(clamp_filter(
            &front,
            &lower,
            &upper,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(result.components().len(), 1);
        assert_eq!(result.components()[0].pairs(), &[(2., 1.)]);
        assert!(!result.used_tolerant_alignment());
    }
}

#[test]
fn native_point_exception_does_not_widen_domains_or_guess_clamp_values() {
    let front = SlopeFront::from_prepared(function(&[(2., 1.)]));
    let upper = function(&[(10., 0.99999), (0., 2.)]);
    // The native helper accepts this near-disjoint domain by tolerance.
    // Its success must not turn our exact-domain resource contract feasible.
    let disjoint = function(&[(0., 1.00001), (0., 2.)]);
    assert!(matches!(
        clamp_filter(
            &front,
            &disjoint,
            &upper,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported("near-disjoint bound domains"))
    ));
    assert!(matches!(
        clamp_filter(
            &front,
            &disjoint,
            &upper,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default()
        ),
        FrontierOutcome::Empty
    ));
    // Raising the point is our lower-clamp contract, not the native lower
    // filter. Keep that near-endpoint case unresolved in native policy.
    let raises = function(&[(3., 0.99999), (0., 2.)]);
    assert!(matches!(
        clamp_filter(
            &front,
            &raises,
            &upper,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
            "near-nonidentical bound endpoint"
        ))
    ));
    let strict = complete(clamp_filter(
        &front,
        &raises,
        &upper,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(strict.components()[0].pairs(), &[(3., 1.)]);
}
