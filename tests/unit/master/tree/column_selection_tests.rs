use super::*;

fn column(tag: usize, value: f64, cost: f64) -> Column {
    Column {
        tag: Variable::Route(tag),
        value,
        cost,
        priority: 1.,
        reduced: cost / 10.,
        guidance: 0.,
        greedy: cost / 2.,
        terminal_resource: Some(1.),
        down: value.floor() > 0.,
        up: true,
    }
}

#[test]
fn column_selection_native_tokens_validate_and_unimplemented_native_cases_do_not_select() {
    for token in (0..=13).chain(15..=17) {
        assert!(validate(&[token]).is_ok());
    }
    for token in [-1, 14, 18] {
        assert!(validate(&[token]).is_err());
    }
    for token in [13, 17] {
        assert_eq!(select(&[column(0, 0.5, 1.)], &[token]), None);
    }
}

#[test]
fn column_selection_fractionality_and_signed_movement_choose_residual_contributions() {
    let candidates = [column(0, 1.2, 4.), column(1, 1.7, 2.)];
    assert_eq!(select(&candidates, &[3]), Some((Variable::Route(0), 2.)));
    assert_eq!(select(&candidates, &[4]), Some((Variable::Route(0), 1.)));
    assert_eq!(select(&candidates, &[9]), Some((Variable::Route(1), 1.)));
    assert_eq!(select(&candidates, &[10]), Some((Variable::Route(1), 1.)));
    for token in [5, 7, 12] {
        assert_eq!(
            select(&candidates, &[token]),
            Some((Variable::Route(1), 2.))
        );
    }
    // Token 16 shares the weighted comparison body but initializes at zero,
    // unlike tokens 5/7/12, so positive movement scores do not select a column.
    assert_eq!(select(&candidates, &[16]), None);
    assert_eq!(
        select(&[column(0, 0.5, -2.)], &[16]),
        Some((Variable::Route(0), 1.))
    );
    // Closest-round-up's explicit down-suitable tie behavior must not be
    // replaced by a generic closest-distance sort.
    assert_eq!(select(&candidates, &[6]), None);
    assert_eq!(
        select(&[column(0, 0.2, 1.), column(1, 0.7, 1.)], &[6]),
        Some((Variable::Route(1), 1.))
    );
}

#[test]
fn column_selection_lexicographic_criteria_break_priority_ties_and_preserve_first_found() {
    let candidates = [column(0, 0.5, 9.), column(1, 0.5, 2.)];
    assert_eq!(select(&candidates, &[2, 9]), Some((Variable::Route(1), 1.)));
    assert_eq!(select(&candidates, &[0]), Some((Variable::Route(0), 1.)));
    assert_eq!(select(&candidates, &[1]), Some((Variable::Route(0), 1.)));
    assert_eq!(select(&candidates, &[11]), Some((Variable::Route(1), 1.)));
}

#[test]
fn column_selection_guidance_and_terminal_resource_use_incumbent_and_original_profile() {
    let mut first = column(0, 0.4, 1.);
    let mut second = column(1, 0.6, 1.);
    first.guidance = 1.;
    second.guidance = 1.;
    assert_eq!(
        select(&[first.clone(), second.clone()], &[8]),
        Some((Variable::Route(1), 1.))
    );
    first.terminal_resource = Some(10.);
    second.terminal_resource = Some(2.);
    assert_eq!(
        select(&[first, second], &[15]),
        Some((Variable::Route(0), 1.))
    );
}

#[test]
fn column_selection_priority_metadata_orders_distinct_network_and_master_candidates() {
    let mut route = column(0, 0.5, 1.);
    route.priority = 2.;
    let mut master = column(1, 0.5, 1.);
    master.tag = Variable::Master(0);
    master.priority = 3.;
    assert_eq!(
        select(&[route.clone(), master.clone()], &[2]),
        Some((Variable::Master(0), 1.))
    );
    route.priority = 4.;
    assert_eq!(
        select(&[route, master], &[2]),
        Some((Variable::Route(0), 1.))
    );
}
