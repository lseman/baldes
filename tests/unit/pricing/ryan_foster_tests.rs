use super::*;

#[test]
fn shared_branch_words_detach_only_on_valid_state_changes() {
    for together in [false, true] {
        let graph = Graph::<2>::new(5, 0).unwrap();
        let resources = RyanFosterResources::prepare(
            &graph,
            &[RyanFosterConstraint {
                first: 1,
                second: 2,
                together,
            }],
        )
        .unwrap();
        let mut parent = None;
        assert!(resources.visit(&mut parent, 1));
        let saved = parent.clone();
        let mut child = parent.clone();
        assert!(resources.visit(&mut child, 3)); // No operations on this vertex.
        assert!(Arc::ptr_eq(
            parent.as_ref().unwrap(),
            child.as_ref().unwrap()
        ));
        assert!(!resources.visit(&mut child, 1)); // Repeated endpoint is forbidden.
        assert!(Arc::ptr_eq(
            parent.as_ref().unwrap(),
            child.as_ref().unwrap()
        ));
        if together {
            assert!(!resources.accepts(&parent));
            assert!(resources.visit(&mut child, 2));
            assert!(!Arc::ptr_eq(
                parent.as_ref().unwrap(),
                child.as_ref().unwrap()
            ));
            assert!(resources.accepts(&child));
            assert!(!resources.accepts(&parent));
            assert!(resources.joins(&parent, &{
                let mut other = None;
                assert!(resources.visit(&mut other, 2));
                other
            }));
        } else {
            assert!(!resources.visit(&mut child, 2));
            assert!(Arc::ptr_eq(
                parent.as_ref().unwrap(),
                child.as_ref().unwrap()
            ));
        }
        assert_eq!(parent, saved);
    }
}
