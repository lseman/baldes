//! Bucket-graph root LP bounds of robust VRPTW over (Γq, Γt).
//! Usage: robust_c201 [instance] [customers] [alpha]. Requires `clp`.
//! Uses the same Solomon conversion and robust resources as CLI `--robust`.
//! Full customer NG memories retain the elementary reference LP's domain.

#[cfg(feature = "clp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use baldes::extensions::robust::{Problem, Uncertainty};
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::cg::{Options, Status};
    use baldes::master::root::{self, RootOptions, RootProblem};
    use std::time::Instant;

    let args = std::env::args().collect::<Vec<_>>();
    let path = args
        .get(1)
        .map_or("instances/Solomon/C201.txt", |s| s.as_str());
    let customers: usize = args.get(2).map_or(Ok(15), |s| s.parse())?;
    let alpha: f64 = args.get(3).map_or(Ok(0.5), |s| s.parse())?;
    let error = |e: baldes::core::error::OrError| format!("{e:?}");
    let mut instance = Instance::read(path).map_err(error)?;
    instance.truncate(customers).map_err(error)?;
    let base = Problem::from_instance(
        &instance,
        Rounding::Tenth,
        Uncertainty::parse(&format!("0,0,{alpha},{alpha}")).map_err(error)?,
    )
    .map_err(error)?;

    println!(
        "customers,alpha,gamma_q,gamma_t,status,lp_bound,artificial,iterations,columns,seconds"
    );
    let budgets = [0, 1, 2, 3, 5, 10];
    for &gq in &budgets {
        for &gt in &budgets {
            let mut problem = base.clone();
            problem.demand_budget = gq;
            problem.time_budget = gt;
            let started = Instant::now();
            let graph = problem
                .native_graph(&vec![0.; problem.nodes.len()], 0., customers)
                .map_err(error)?;
            let options = RootOptions {
                enumerate: false,
                cg: Options {
                    multiplicity: customers as f64,
                    artificial_cost: 1e5,
                    ..Default::default()
                },
                ..Default::default()
            };
            let result = root::solve(
                RootProblem::set_partitioning(graph, customers as f64).map_err(error)?,
                options,
            )
            .map_err(error)?;
            let artificial: f64 = result
                .master
                .artificials
                .iter()
                .map(|&c| result.values[c].max(0.))
                .sum();
            let lp = if result.master.status == Status::Optimal && artificial <= 1e-6 {
                result.master.bound
            } else {
                f64::NAN
            };
            let columns = result
                .master
                .column_tags
                .iter()
                .filter(|&&tag| tag >= 0)
                .count();
            println!(
                "{customers},{alpha},{gq},{gt},{:?},{lp:.4},{artificial:.4},{},{columns},{:.3}",
                result.master.status,
                result.master.iterations,
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("robust_c201 requires --features clp");
}
