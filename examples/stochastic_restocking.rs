//! Run with `cargo run -p baldes --example stochastic_restocking`.
use baldes::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
use baldes::graph::coupled::{CoupledResource, PluginResource};
use baldes::{price, Controls, Graph, Result};

fn main() -> Result<()> {
    let demand = Demand::new(vec![(0, 0.5), (4, 0.5)])?;
    let problem = Problem::new(
        4,
        vec![demand.clone(), demand],
        vec![vec![0., 3., 7.], vec![3., 0., 4.], vec![7., 4., 0.]],
    )?;
    let mut graph = Graph::<2>::new(4, 0)?;
    let mut endpoints = Vec::new();
    for (tail, head, model_tail, model_head) in [
        (0, 1, 0, 1),
        (0, 2, 0, 2),
        (1, 2, 1, 2),
        (2, 1, 2, 1),
        (1, 3, 1, 0),
        (2, 3, 2, 0),
    ] {
        let arc = graph.add_arc(tail, head, problem.costs()[model_tail][model_head])?;
        // Simulate master customer duals; original costs retain planned travel.
        graph.arc_mut(arc)?.reduced_cost -= if model_head == 0 { 0. } else { 15. };
        endpoints.push((model_tail, model_head));
    }
    let resource = RestockingResource::new(problem, endpoints)?;
    graph.set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(resource))])?;
    let result = price(&graph, &Controls::default())?;
    for column in result.columns {
        println!(
            "{:?}: expected cost {}, reduced cost {}",
            column.nodes, column.original_cost, column.cost
        );
    }
    Ok(())
}
