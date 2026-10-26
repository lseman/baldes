use super::*;
fn function(p: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(p.to_vec()).unwrap()
}
fn arc(waiting: i32) -> PreparedSlopeRelationArc {
    PreparedSlopeRelationArc::new(
        &SlopeRelationArc {
            arrival: function(&[(0., 0.), (1., 10.)]),
            increment: function(&[(1., 0.), (0., 10.)]),
            lower: function(&[(0., 0.), (0., 10.)]),
            upper: function(&[(100., 0.), (0., 10.)]),
            waiting,
        },
        FrontOperationPolicy::StrictDomain,
    )
    .unwrap()
}
#[test]
fn nonconvex_waiting_keeps_record_until_actual_crossing() {
    let front = SlopeFront::from_prepared(function(&[(0., 0.), (2., 1.), (-2., 3.)]));
    let FrontierOutcome::Complete(out) = arc(0).extend(&front, FrontierBudget::default()) else {
        panic!("exact crossing must remain supported")
    };
    assert_eq!(
        out.components()[0]
            .evaluate(1.5, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(1.)
    );
    assert_eq!(
        out.components()[0]
            .evaluate(2.5, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(0.)
    );
}
#[test]
fn fractional_native_underpricing_is_not_a_feasibility_result() {
    let front = SlopeFront::from_prepared(function(&[(2.1, 0.), (0.3, 0.7), (-0.2, 2.)]));
    let out = arc(0).extend(&front, FrontierBudget::default());
    match out {
        FrontierOutcome::Unknown(_) => {}
        FrontierOutcome::Complete(out) => {
            let FrontierOutcome::Complete(w) = out.minimum(FrontierBudget::default()) else {
                panic!("minimum")
            };
            assert!(w.value >= 3.05 - 1e-12);
        }
        FrontierOutcome::Empty => panic!("unsupported arithmetic cannot mean infeasible"),
    }
}
#[test]
fn zero_work_is_unknown_not_empty() {
    let out = arc(1).extend(
        &SlopeFront::zero(0., 10.).unwrap(),
        FrontierBudget {
            max_operations: 0,
            ..FrontierBudget::default()
        },
    );
    assert!(matches!(out, FrontierOutcome::Unknown(_)));
}
#[test]
fn increasing_arrival_and_departure_increment_use_cached_composition() {
    let source = SlopeRelationArc {
        arrival: function(&[(0., 0.), (2., 2.)]),
        increment: function(&[(0., 0.), (1., 2.)]),
        lower: function(&[(0., 0.), (0., 4.)]),
        upper: function(&[(100., 0.), (0., 4.)]),
        waiting: 5,
    };
    let arc = PreparedSlopeRelationArc::new(&source, FrontOperationPolicy::StrictDomain).unwrap();
    let front = SlopeFront::from_prepared(function(&[(2., 0.), (0., 2.)]));
    let FrontierOutcome::Complete(out) = arc.extend(&front, FrontierBudget::default()) else {
        panic!("dyadic composition must remain supported")
    };
    assert_eq!(
        out.components()[0]
            .evaluate(3., EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(3.5)
    );
}
#[test]
fn paired_composition_intersects_each_component_domain_before_waiting() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra,
    ] {
        let source = SlopeRelationArc {
            arrival: function(&[(0., 0.), (2., 4.)]),
            increment: function(&[(1., 0.), (1., 4.)]),
            lower: function(&[(0., 1.), (0., 4.)]),
            upper: function(&[(100., 1.), (0., 4.)]),
            waiting: 1,
        };
        let arc = PreparedSlopeRelationArc::new(&source, policy).unwrap();
        let front = SlopeFront::from_prepared(function(&[(5., 0.), (2., 2.)]));
        let FrontierOutcome::Complete(out) = arc.extend(&front, FrontierBudget::default()) else {
            panic!("partial component preimage remains feasible under {policy:?}");
        };
        assert_eq!(out.components()[0].domain(), (1., 4.));
        assert_eq!(
            out.components()[0]
                .evaluate(1., EvaluationPolicy::StrictDomain)
                .unwrap(),
            Some(7.)
        );
        assert_eq!(
            out.components()[0]
                .evaluate(4., EvaluationPolicy::StrictDomain)
                .unwrap(),
            Some(10.)
        );
    }
}
#[test]
fn signed_waiting_and_function_valued_lower_bound_remain_operational() {
    let front = SlopeFront::from_prepared(function(&[(2., 0.), (0., 2.)]));
    let mut source = SlopeRelationArc {
        arrival: function(&[(0., 0.), (1., 2.)]),
        increment: function(&[(0., 0.), (0., 2.)]),
        lower: function(&[(0., 0.), (0., 2.)]),
        upper: function(&[(100., 0.), (0., 2.)]),
        waiting: -1,
    };
    let prepared =
        PreparedSlopeRelationArc::new(&source, FrontOperationPolicy::StrictDomain).unwrap();
    let FrontierOutcome::Complete(out) = prepared.extend(&front, FrontierBudget::default()) else {
        panic!("signed waiting")
    };
    assert_eq!(
        out.components()[0]
            .evaluate(1., EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(1.)
    );
    source.lower = function(&[(0., 0.), (1., 2.)]);
    source.waiting = 0;
    let prepared =
        PreparedSlopeRelationArc::new(&source, FrontOperationPolicy::StrictDomain).unwrap();
    let FrontierOutcome::Complete(out) = prepared.extend(
        &SlopeFront::zero(0., 2.).unwrap(),
        FrontierBudget::default(),
    ) else {
        panic!("function-valued lower bound")
    };
    assert_eq!(
        out.components()[0]
            .evaluate(1.5, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(1.5)
    );
}
