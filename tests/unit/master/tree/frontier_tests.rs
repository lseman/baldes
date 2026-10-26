use super::*;
struct Min;
impl Priority<(i32, usize)> for Min {
    fn less(&self, left: &(i32, usize), right: &(i32, usize)) -> bool {
        left > right
    }
}

#[test]
fn retained_and_new_nodes_share_stable_priority_after_pruning() {
    let mut queue = Frontier::new(Min);
    for node in [(5, 0), (2, 1), (3, 2), (2, 3), (1, 4), (4, 5)] {
        queue.push(node);
    }
    assert_eq!(queue.pop(), Some((1, 4)));
    queue.retain(|&(bound, _)| bound < 5 && bound != 3);
    for node in [(2, 6), (0, 7)] {
        queue.push(node);
    }
    let order: Vec<_> = std::iter::from_fn(|| queue.pop()).collect();
    assert_eq!(order, vec![(0, 7), (2, 1), (2, 3), (2, 6), (4, 5)]);
    assert!(queue.is_empty());
}

#[test]
fn frontier_handles_every_heap_shape_after_filtering() {
    for count in 0..80 {
        let values: Vec<_> = (0..count)
            .map(|i| (((i * 17 + 3) % 11) as i32, i))
            .collect();
        let mut queue = Frontier::new(Min);
        for node in values.iter().copied() {
            queue.push(node);
        }
        queue.retain(|&(_, id)| id % 3 != 0);
        let actual: Vec<_> = std::iter::from_fn(|| queue.pop()).collect();
        let mut expected: Vec<_> = values.into_iter().filter(|&(_, id)| id % 3 != 0).collect();
        expected.sort();
        assert_eq!(actual, expected);
    }
}
