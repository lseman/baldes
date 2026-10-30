use baldes::extensions::stochastic::{Demand, Policy, Problem, TinyOptions};

fn main() -> baldes::Result<()> {
    let problem = Problem::new(
        10,
        vec![
            Demand::new(vec![(4, 0.5), (8, 0.5)])?,
            Demand::new(vec![(4, 1.0)])?,
        ],
        vec![vec![0., 1., 10.], vec![1., 0., 10.], vec![10., 10., 0.]],
    )?;
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let evaluated = problem.evaluate_route(&[1, 2], policy)?;
        println!("{policy:?}, route [1, 2]: {evaluated:?}");
        let solution = problem.solve_tiny(
            policy,
            TinyOptions {
                max_vehicles: 2,
                route_budget: 1000,
            },
        )?;
        println!("Tiny oracle with at most two vehicles: {solution:?}");
        #[cfg(feature = "clp")]
        {
            let solved = problem.solve(policy, 2, Default::default())?;
            println!("BALDES routing with Theta recourse cuts: {solved:?}");
        }
    }
    Ok(())
}
