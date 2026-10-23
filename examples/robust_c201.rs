//! Root LP bounds of the robust VRPTW on a Solomon prefix, over (Γq, Γt).
//! Usage: robust_c201 [instance] [customers] [alpha]. Requires `clp`.
//! Distances are truncated to one decimal and used as cost and time; the
//! deviations follow Munari et al.: trunc(αq), 0.1·trunc(10αt).

#[cfg(feature = "clp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use baldes::extensions::robust::{
        Node, Problem, RobustColumnGenerationOptions, RobustMaster, Travel,
    };
    use baldes::master::clp::ClpMaster;
    use std::time::Instant;

    let args = std::env::args().collect::<Vec<_>>();
    let path = args
        .get(1)
        .map_or("instances/Solomon/C201.txt", |s| s.as_str());
    let customers: usize = args.get(2).map_or(Ok(15), |s| s.parse())?;
    let alpha: f64 = args.get(3).map_or(Ok(0.5), |s| s.parse())?;

    let text = std::fs::read_to_string(path)?;
    let capacity = text
        .lines()
        .skip_while(|l| !l.contains("CAPACITY"))
        .nth(1)
        .and_then(|l| l.split_whitespace().nth(1))
        .ok_or("missing capacity")?
        .parse::<f64>()?;
    let rows = text
        .lines()
        .filter_map(|l| {
            let v = l
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            (v.len() == 7).then_some(v)
        })
        .take(customers + 1)
        .collect::<Vec<_>>();
    let nodes = rows
        .iter()
        .map(|r| Node {
            demand: r[3],
            demand_deviation: 0.0,
            window: (r[4], r[5]),
            service: r[6],
        })
        .collect();
    let travel = rows
        .iter()
        .enumerate()
        .map(|(i, a)| {
            rows.iter()
                .enumerate()
                .map(|(j, b)| {
                    let d = ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                    let d = (10.0 * d).trunc() / 10.0;
                    (i != j).then_some(Travel {
                        time: d,
                        time_deviation: 0.0,
                        cost: d,
                    })
                })
                .collect()
        })
        .collect();
    let base = Problem {
        nodes,
        travel,
        depot: 0,
        capacity,
        demand_budget: 0,
        time_budget: 0,
    }
    .with_relative_deviations(alpha, alpha)
    .map_err(|e| format!("{e:?}"))?;

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
            let mut master = RobustMaster::new(
                problem,
                ClpMaster::new().map_err(|e| format!("{e:?}"))?,
                1e5,
            )
            .map_err(|e| format!("{e:?}"))?;
            let result = master
                .optimize(RobustColumnGenerationOptions::default())
                .map_err(|e| format!("{e:?}"))?;
            println!(
                "{customers},{alpha},{gq},{gt},{:?},{:.4},{:.4},{},{},{:.3}",
                result.status,
                result.lp_objective.unwrap_or(f64::NAN),
                result.artificial_usage,
                result.iterations.len(),
                master.columns().len(),
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
