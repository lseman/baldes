use super::*;
use crate::inputs::{Instance, Rounding};

fn triangle() -> Graph {
    let instance = Instance::parse(
        "triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n",
    )
    .unwrap();
    instance.graph(1, Rounding::None).unwrap()
}

#[test]
fn set_partitioning_covers_customers_and_bounds_the_fleet_by_multiplicity() {
    let graph = triangle();
    let problem = Problem::set_partitioning(graph.clone(), 3.).unwrap();
    assert_eq!(problem.customers(), vec![1, 2, 3]);
    assert_eq!(problem.rows.len(), 3);
    assert!(problem
        .rows
        .iter()
        .all(|r| r.sense == RowSense::Equal && r.rhs == 1.));
    let network = &problem.networks[0];
    assert_eq!(
        (network.multiplicity.lower, network.multiplicity.upper),
        (0., 3.)
    );
    for (arc, entries) in graph.arcs.iter().zip(&network.arc_entries) {
        if arc.head == graph.sink {
            assert!(entries.is_empty());
        } else {
            assert_eq!(entries, &vec![(arc.head - 1, 1.)]);
        }
    }
    assert!(Problem::set_partitioning(graph, 0.).is_err());
}

#[test]
fn arc_flow_variables_register_each_arc_once() {
    let problem = Problem::set_partitioning(triangle(), 3.).unwrap();
    let variables = problem.arc_flow_variables(1.);
    assert_eq!(variables.len(), problem.networks[0].graph.arcs.len());
    for (variable, arc) in variables.iter().zip(&problem.networks[0].graph.arcs) {
        assert_eq!(variable.priority, 1.);
        assert_eq!(variable.networks[0].arcs, vec![(arc.id, 1.)]);
    }
}

#[cfg(feature = "clp")]
#[test]
fn set_partitioning_root_matches_the_fleet_row_formulation() {
    let mut options = Options::default();
    options.cg.stabilize = false;
    let quantity = solve(
        Problem::set_partitioning(triangle(), 3.).unwrap(),
        options.clone(),
    )
    .unwrap();
    // The same model with the fleet as an explicit shared row on sink arcs.
    let mut fleet_row = Problem::set_partitioning(triangle(), 3.).unwrap();
    fleet_row.rows.push(Row {
        sense: RowSense::LessEqual,
        rhs: 3.,
    });
    let network = &mut fleet_row.networks[0];
    network.multiplicity.upper = f64::INFINITY;
    for (arc, entries) in network.graph.arcs.iter().zip(&mut network.arc_entries) {
        if arc.head == network.graph.sink {
            entries.push((3, 1.));
        }
    }
    let explicit = solve(fleet_row, options).unwrap();
    assert!(
        (quantity.master.lp - explicit.master.lp).abs() < 1e-6,
        "{} {}",
        quantity.master.lp,
        explicit.master.lp
    );
}
