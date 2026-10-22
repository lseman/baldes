use super::*;
#[test]
fn routing_elementarity_uses_customer_ids_and_omits_depots() {
    let mut instance = Instance::read("instances/R201.txt").unwrap();
    instance.truncate(10).unwrap();
    for graph in [
        instance.graph(12, Rounding::None).unwrap(),
        instance.capacity_graph(12, Rounding::None).unwrap(),
    ] {
        assert_eq!(graph.element_count, 10);
        assert!(graph.vertices[graph.source].elements.is_empty());
        assert!(graph.vertices[graph.sink].elements.is_empty());
        for customer in 1..=10 {
            assert_eq!(graph.vertices[customer].elements, [customer - 1]);
        }
        for vertex in &graph.vertices {
            assert_eq!(vertex.memory, (0..10).collect::<Vec<_>>());
        }
    }
}
