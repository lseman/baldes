//! Run with: cargo run --release --features clp --example pstep
#[cfg(feature = "clp")]
fn main() -> baldes::Result<()> {
    use baldes::extensions::pstep::{EnumerationOptions, Problem};
    use baldes::master::{clp::ClpMaster, LpMaster, LpSolveOutcome};
    use std::time::Duration;

    let problem = Problem {
        demands: vec![1.0; 3],
        capacity: 2.0,
        // Separate start (0) and end (4) copies of the same depot.
        costs: vec![vec![Some(1.0); 5]; 5],
    };
    for p in 1..=4 {
        let formulation = problem.enumerate(EnumerationOptions {
            p,
            ..Default::default()
        })?;
        let mut lp = formulation.into_master(ClpMaster::new()?)?;
        let outcome = lp.solve_with_time_limit(Duration::from_secs(30))?;
        if outcome == LpSolveOutcome::Optimal {
            println!(
                "p={p}: {} columns, LP bound {:.6}",
                formulation.steps().len(),
                lp.objective()
            );
        } else {
            println!("p={p}: LP stopped: {outcome:?}");
        }
    }
    Ok(())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("Run this example with --features clp");
}
