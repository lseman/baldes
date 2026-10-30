use super::*;
fn route(id: i32, cost: f64) -> crate::Route {
    crate::Route {
        coupled_values: Vec::new(),
        arcs: vec![id],
        nodes: vec![],
        cost,
        original_cost: cost,
        resources: vec![],
        resource_profile: vec![],
    }
}
#[test]
fn numeric_selection_ranks_unique_new_columns_and_handles_short_pools() {
    let pool = vec![
        route(5, 9.),
        route(3, 2.),
        route(1, 1.),
        route(2, 2.),
        route(2, 2.),
    ];
    let existing = vec![route(1, 1.)];
    let top = select_new_columns(pool.clone(), &existing, HgsColumns::Top(2));
    assert_eq!(
        top.iter().map(|r| r.arcs[0]).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(
        select_new_columns(pool.clone(), &existing, HgsColumns::Top(100)).len(),
        3
    );
    assert_eq!(
        select_new_columns(pool, &existing, HgsColumns::All)[0].arcs,
        vec![5]
    );
    assert!(select_new_columns(vec![], &existing, HgsColumns::Top(1)).is_empty());
    assert_eq!(HgsColumns::parse("7").unwrap(), HgsColumns::Top(7));
    for invalid in ["0", "-1", "1.5", "wrong", "184467440737095516160"] {
        assert!(HgsColumns::parse(invalid).is_err());
    }
}
