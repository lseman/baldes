//! Small fractional CVRP route LP solved by decision-diagram refinement.
#[cfg(feature = "clp")]
fn main() -> baldes::Result<()> {
    use baldes::extensions::ce::{Options, Problem, Status};
    use baldes::master::solvers::ClpMaster;
    let problem = Problem {
        demands: vec![1.0; 3],
        capacity: 2.0,
        costs: vec![vec![Some(1.0); 5]; 5],
        travel_times: vec![vec![1.0; 5]; 5],
        windows: vec![(0.0, f64::INFINITY); 5],
        service_times: vec![0.0; 5],
        max_vehicles: None,
    };
    let result = problem.solve_column_elimination(ClpMaster::new, Options::default())?;
    println!(
        "{:?}: bound {:?}, {} refinements, {} nodes, {} arcs",
        result.status, result.lower_bound, result.refinements, result.nodes, result.arcs
    );
    assert_eq!(result.status, Status::Converged);
    assert!((result.lower_bound.unwrap() - 4.5).abs() < 1e-6);
    for route in result.routes {
        println!("{} * {:?} (cost {})", route.flow, route.path, route.cost);
    }
    Ok(())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("Run with --features clp");
}
