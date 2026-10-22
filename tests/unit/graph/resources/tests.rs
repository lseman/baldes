use super::*;

#[test]
fn arbitrary_const_capacity_uses_only_the_fixed_array() {
    const DIMENSIONS: usize = 37;
    assert_eq!(
        std::mem::size_of::<ResourceValues<DIMENSIONS>>(),
        DIMENSIONS * 8
    );
    let mut graph = crate::Graph::<DIMENSIONS>::new(2, DIMENSIONS).unwrap();
    let arc = graph.add_arc(0, 1, -1.).unwrap();
    graph.arc_mut(arc).unwrap().consumption[DIMENSIONS - 1] = 3.;
    let q = crate::labeling::ScalarResourceKernel::<DIMENSIONS>::extend(
        &graph,
        &graph.arcs[0],
        [0.; DIMENSIONS],
        true,
    )
    .unwrap();
    assert_eq!(q[DIMENSIONS - 1], 3.);
    assert_eq!(graph.initial.len(), DIMENSIONS);
    assert_eq!(graph.arcs[0].bounds.len(), DIMENSIONS);
}
