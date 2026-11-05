use super::*;
use crate::inputs::Customer;
fn instance(capacity: f64, vehicles: usize) -> Instance {
    Instance {
        name: "savings".into(),
        vehicles,
        capacity,
        rounding: Rounding::Nearest,
        distances: None,
        customers: (0..4)
            .map(|i| Customer {
                x: i as f64,
                y: 0.,
                demand: if i == 0 { 0. } else { 1. },
                ready: 0.,
                due: 100.,
                service: 0.,
            })
            .collect(),
    }
}
#[test]
fn savings_merges_and_replays_capacity_routes() {
    let instance = instance(3., 1);
    let graph = instance.capacity_graph(3, Rounding::Nearest).unwrap();
    let seed = instance.seed_savings(&graph, Rounding::Nearest).unwrap();
    assert_eq!(seed.routes.len(), 1);
    assert_eq!(seed.incumbent, Some(6.));
    assert_eq!(seed.routes[0].nodes.len(), 5);
    assert!(graph
        .elementary_route(&seed.routes[0].arcs)
        .unwrap()
        .is_some());
}
#[test]
fn savings_respects_capacity_and_does_not_claim_an_excess_fleet_incumbent() {
    let instance = instance(1., 1);
    let graph = instance.capacity_graph(3, Rounding::Nearest).unwrap();
    let seed = instance.seed_savings(&graph, Rounding::Nearest).unwrap();
    assert_eq!(seed.routes.len(), 3);
    assert_eq!(seed.incumbent, None);
}
#[test]
fn savings_respects_time_windows_and_waiting() {
    let mut instance = instance(3., 3);
    for customer in &mut instance.customers[1..] {
        customer.ready = 5.;
        customer.due = 5.;
    }
    let graph = instance.graph(3, Rounding::Nearest).unwrap();
    let seed = instance.seed_savings(&graph, Rounding::Nearest).unwrap();
    assert_eq!(seed.routes.len(), 3);
    assert_eq!(seed.incumbent, Some(12.));
    let capacity = instance.capacity_graph(3, Rounding::Nearest).unwrap();
    assert_eq!(
        instance
            .seed_savings(&capacity, Rounding::Nearest)
            .unwrap()
            .routes
            .len(),
        1
    );
}
#[test]
fn savings_omits_unreachable_customers_without_claiming_a_complete_plan() {
    let mut instance = instance(3., 3);
    instance.customers[3].due = 0.;
    let graph = instance.graph(3, Rounding::Nearest).unwrap();
    let seed = instance.seed_savings(&graph, Rounding::Nearest).unwrap();
    assert!(seed.routes.iter().all(|r| !r.nodes.contains(&3)));
    assert_eq!(seed.incumbent, None);
}
