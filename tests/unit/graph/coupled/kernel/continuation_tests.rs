use super::*;

fn long_tail() -> Continuation {
    (0..30_000).fold(Continuation::default(), |tail, arc| tail.prepend(arc))
}

#[test]
fn dropping_a_branch_preserves_the_shared_tail_and_drains_unique_steps() {
    let tail = long_tail();
    let weak = Arc::downgrade(tail.0.as_ref().unwrap());
    let branch = tail.prepend(30_000);
    drop(branch);
    assert_eq!(tail.len(), 30_000);
    assert!(tail.iter().eq((0..30_000).rev()));
    assert!(weak.upgrade().is_some());
    drop(tail);
    assert!(weak.upgrade().is_none());
}

#[test]
fn concurrent_branch_drops_release_the_complete_shared_tail() {
    let tail = long_tail();
    let weak = Arc::downgrade(tail.0.as_ref().unwrap());
    let barrier = std::sync::Barrier::new(5);
    std::thread::scope(|scope| {
        for arc in 30_000..30_004 {
            let branch = tail.prepend(arc);
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                drop(branch);
            });
        }
        drop(tail);
        barrier.wait();
    });
    assert!(weak.upgrade().is_none());
}
