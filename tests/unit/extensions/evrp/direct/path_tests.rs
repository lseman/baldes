use super::*;

#[test]
fn reconstruction_keeps_inactive_ancestors_and_shared_subpaths() {
    let make = |parent, step, depth, location| Label {
        parent,
        step,
        depth,
        location,
        visited: vec![],
        load: 0.,
        reduced: 0.,
        frontier: vec![],
        active: false,
    };
    let segment: std::sync::Arc<[usize]> = vec![4, 6, 8, 9].into();
    let labels = vec![
        make(None, PathStep::Vertex(9), 1, 9),
        make(Some(0), PathStep::Vertex(4), 2, 4),
        make(Some(1), PathStep::Subpath(segment.clone()), 5, 9),
    ];
    assert_eq!(label_path(&labels, 0), vec![9]);
    assert_eq!(label_path(&labels, 1), vec![9, 4]);
    assert_eq!(label_path(&labels, 2), vec![9, 4, 6, 8, 9]);
    assert_eq!(&*segment, &[4, 6, 8, 9]);
}
