use super::*;
#[test]
fn multiway_parent_waits_for_nested_group_and_last_sibling() {
    let mut history = SubtreeHistory::new();
    let children: Vec<_> = history.split_many(0, 3).collect();
    let grandchildren: Vec<_> = history.split_many(children[1], 4).collect();
    history.close(children[0]);
    for child in grandchildren {
        history.close(child);
    }
    assert_eq!(history.mean(2), Some(1.));
    assert_eq!(history.mean(1), Some(3.));
    assert_eq!(history.mean(0), None);
    history.close(children[2]);
    assert_eq!(history.mean(1), Some(7. / 3.));
    assert_eq!(history.mean(0), Some(8.));
    history.close(children[1]);
    history.close(children[2]);
    assert_eq!(history.mean(0), Some(8.));
}

#[test]
fn averages_wait_for_every_descendant_and_ignore_repeated_closure() {
    let mut history = SubtreeHistory::new();
    let [left, right] = history.split(0);
    let [a, b] = history.split(right);
    history.close(left);
    assert_eq!(history.mean(1), Some(1.));
    assert_eq!(history.mean(0), None);
    history.close(a);
    assert_eq!(history.mean(1), Some(1.));
    assert_eq!(history.mean(0), None);
    history.close(b);
    assert_eq!(history.mean(2), Some(1.));
    assert_eq!(history.mean(1), Some(2.));
    assert_eq!(history.mean(0), Some(5.));
    history.close(b);
    history.close(0);
    assert_eq!(history.mean(1), Some(2.));
    assert_eq!(history.mean(0), Some(5.));
}

#[test]
fn singleton_branch_counts_its_child_before_closing_parent() {
    let mut history = SubtreeHistory::new();
    let child = history.split_many(0, 1).start;
    assert_eq!(history.mean(0), None);
    history.close(0);
    assert_eq!(history.mean(0), None);
    history.close(child);
    assert_eq!(history.mean(0), Some(2.));
    history.close(child);
    assert_eq!(history.mean(0), Some(2.));
}
