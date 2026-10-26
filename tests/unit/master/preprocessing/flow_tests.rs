use super::*;
#[test]
fn flow_offsets_and_quantity_links_preserve_self_loop_cancellation() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let rows = conservation(&graph, 4, Some(7));
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].entries, vec![(4, 1.), (7, -1.)]);
    assert_eq!(rows[1].entries, vec![(4, -1.), (6, 1.)]);
    assert_eq!(rows[2].entries, vec![(6, -1.), (7, 1.)]);
    let interior = conservation(&graph, 0, None);
    assert_eq!(interior.len(), 1);
    assert_eq!(interior[0].entries, vec![(0, -1.), (2, 1.)]);
}
