use super::*;

fn column(tag: i32) -> cg::Column {
    cg::Column {
        cost: 2.,
        reduced_cost: -1.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag,
        identity: format!("route-{tag}"),
    }
}

#[test]
fn child_edits_detach_only_selected_columns() {
    let mut warm = cg::WarmStart::empty(cg::BoundMode::Numerical);
    warm.columns = vec![Some(column(1)), None, Some(column(2))];
    warm.generated_pool = vec![column(3)];
    let parent = RetainedWarmStart::from(warm);
    let mut child = parent.clone();
    let sibling = parent.clone();
    let original = parent.columns[0].as_ref().unwrap().entries.as_ptr();
    assert_eq!(
        child.columns[0].as_ref().unwrap().entries.as_ptr(),
        original
    );
    child.edit_columns(|c| c.tag == 2, |c| c.entries.push((1, 2.)));
    assert_eq!(
        child.columns[0].as_ref().unwrap().entries.as_ptr(),
        original
    );
    assert_eq!(
        child.generated_pool[0].entries.as_ptr(),
        parent.generated_pool[0].entries.as_ptr()
    );
    assert_eq!(parent.columns[2].as_ref().unwrap().entries.len(), 1);
    assert_eq!(sibling.columns[2].as_ref().unwrap().entries.len(), 1);
    let owned = child.into_owned();
    assert!(owned.columns[1].is_none());
    assert_eq!(
        owned.columns[2].as_ref().unwrap().entries,
        vec![(0, 1.), (1, 2.)]
    );
    assert_eq!(owned.generated_pool[0].identity, "route-3");
}

#[test]
fn uniquely_owned_detached_column_moves_payload() {
    let original = column(7);
    let entries = original.entries.as_ptr();
    let retained = RetainedColumn::from(original);
    let owned = retained.into_owned();
    assert_eq!(owned.entries.as_ptr(), entries);
    assert_eq!(owned.tag, 7);
}
