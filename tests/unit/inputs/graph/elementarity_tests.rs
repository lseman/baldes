use super::*;
#[test]
fn routing_elementarity_uses_customer_ids_and_omits_depots() {
    let mut instance = Instance::read("instances/Solomon/R201.txt").unwrap();
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
        assert!(graph.vertices[graph.source].memory.is_empty());
        assert!(graph.vertices[graph.sink].memory.is_empty());
        for customer in 1..=10 {
            assert_eq!(graph.vertices[customer].memory, (0..10).collect::<Vec<_>>());
        }
    }
}

#[test]
fn routing_ng_ranks_customers_only_like_native() {
    let mut instance = Instance::read("instances/Solomon/R201.txt").unwrap();
    instance.truncate(10).unwrap();
    let ng = 3;
    let graph = instance.graph(ng, Rounding::None).unwrap();
    for customer in 1..=10 {
        let mut order: Vec<_> = (1..=10).collect();
        order.sort_by(|&a, &b| {
            instance
                .distance(customer, a, Rounding::None)
                .total_cmp(&instance.distance(customer, b, Rounding::None))
                .then(a.cmp(&b))
        });
        let mut expected: Vec<_> = order[..ng].iter().map(|&c| c - 1).collect();
        expected.sort_unstable();
        assert_eq!(graph.vertices[customer].memory, expected);
        assert!(expected.contains(&(customer - 1)));
    }
}
