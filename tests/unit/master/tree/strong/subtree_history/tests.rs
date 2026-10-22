use super::*;
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
