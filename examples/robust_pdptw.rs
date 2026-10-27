//! cargo run --release --features clp --example robust_pdptw -- [FILE Γq Γt σq σt SECONDS ROUNDING] [--ng N] [--heuristic [ils|hgs]] [--seed N]
#[cfg(feature = "clp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use baldes::extensions::robust_pd::{
        DistanceRounding, HeuristicMethod, HeuristicOptions, Instance, Parameters, Problem,
    };
    use baldes::master::tree::ModelTreeOptions;
    use std::time::{Duration, Instant};

    fn invalid(message: &str) -> baldes::core::error::OrError {
        baldes::core::error::OrError {
            kind: baldes::core::error::ErrorKind::InvalidArgument,
            message: message.to_owned(),
        }
    }
    fn solve() -> baldes::Result<()> {
        let mut args: Vec<_> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            println!("robust_pdptw [FILE Γq Γt σq σt SECONDS ROUNDING] [--ng N] [--heuristic [ils|hgs]] [--seed N]\nDefaults: instances/RobustPDPTW/AA05.txt 1 1 0.1 0.1 60 hundredth\nROUNDING: hundredth or none. Paper §6.1 demand adjustment enabled.\n--ng N: relax elementarity to NG routes with N requests per neighbourhood.\nWithout it, pricing is elementary.\n--heuristic [ils|hgs]: run only a robust heuristic for SECONDS: iterated\nlocal search (default) or a hybrid genetic search. --seed N: heuristic seed (1).");
            return Ok(());
        }
        let ng_size = match args.iter().position(|a| a == "--ng") {
            Some(i) if i + 1 < args.len() => {
                let size: usize = args[i + 1]
                    .parse()
                    .map_err(|_| invalid("--ng needs a positive request count"))?;
                if size == 0 {
                    return Err(invalid("--ng needs a positive request count"));
                }
                args.drain(i..=i + 1);
                Some(size)
            }
            Some(_) => return Err(invalid("--ng needs a positive request count")),
            None => None,
        };
        let seed = match args.iter().position(|a| a == "--seed") {
            Some(i) if i + 1 < args.len() => {
                let seed: u64 = args[i + 1]
                    .parse()
                    .map_err(|_| invalid("--seed needs an integer"))?;
                args.drain(i..=i + 1);
                seed
            }
            Some(_) => return Err(invalid("--seed needs an integer")),
            None => 1,
        };
        let heuristic = match args.iter().position(|a| a == "--heuristic") {
            Some(i) => {
                let method = match args.get(i + 1).map(String::as_str) {
                    Some("hgs") => Some(HeuristicMethod::Genetic),
                    Some("ils") => Some(HeuristicMethod::IteratedLocalSearch),
                    _ => None,
                };
                args.drain(i..i + 1 + usize::from(method.is_some()));
                Some(method.unwrap_or_default())
            }
            None => None,
        };
        if args.len() > 7 || args.iter().any(|a| a.starts_with("--")) {
            return Err(invalid(
                "expected FILE Γq Γt σq σt SECONDS ROUNDING [--ng N] [--heuristic [ils|hgs]] [--seed N]",
            ));
        }
        let number = |i: usize, default: &str| args.get(i).map_or(default.to_owned(), Clone::clone);
        let parse_error = || invalid("invalid numeric argument");
        let parameters = Parameters {
            demand_budget: number(1, "1").parse().map_err(|_| parse_error())?,
            time_budget: number(2, "1").parse().map_err(|_| parse_error())?,
            demand_deviation: number(3, "0.1").parse().map_err(|_| parse_error())?,
            time_deviation: number(4, "0.1").parse().map_err(|_| parse_error())?,
            distance_rounding: match number(6, "hundredth").as_str() {
                "hundredth" => DistanceRounding::NearestHundredth,
                "none" => DistanceRounding::FullPrecision,
                _ => return Err(invalid("rounding must be hundredth or none")),
            },
            ng_size,
            ..Default::default()
        };
        let seconds: f64 = number(5, "60").parse().map_err(|_| parse_error())?;
        if !seconds.is_finite() || !(0.001..=86400.).contains(&seconds) {
            return Err(invalid("seconds must be in [0.001,86400]"));
        }
        let path = args
            .first()
            .map_or("instances/RobustPDPTW/AA05.txt", String::as_str);
        let problem = Problem::new(Instance::read(path)?, parameters)?;
        let started = Instant::now();
        // (status, bound, incumbent, nodes, pricing calls, routes with quantities)
        let (status, bound, incumbent, nodes, pricing_calls, routes) =
            if let Some(method) = heuristic {
                let solution = problem.heuristic(HeuristicOptions {
                    time_limit: Duration::from_secs_f64(seconds),
                    iterations: usize::MAX,
                    seed,
                    method,
                });
                let incumbent = solution.as_ref().map_or(f64::INFINITY, |s| s.cost);
                let routes = solution.map_or_else(Vec::new, |s| {
                    s.routes.into_iter().map(|r| (1., r)).collect::<Vec<_>>()
                });
                (
                    "Heuristic".to_owned(),
                    f64::NEG_INFINITY,
                    incumbent,
                    0,
                    0,
                    routes,
                )
            } else {
                let mut options = ModelTreeOptions {
                    time_limit: Duration::from_secs_f64(seconds),
                    ..Default::default()
                };
                options.node.cg.time_limit = Duration::from_secs_f64(seconds);
                options.node.cg.artificial_cost =
                    (parameters.vehicle_cost + 100_000.) * problem.instance().requests() as f64;
                let output = baldes::master::tree::solve(problem.master_problem()?, options)?;
                let routes = output
                    .routes
                    .iter()
                    .map(|fixed| {
                        let route = fixed
                            .route
                            .route
                            .nodes
                            .iter()
                            .map(|v| *v as usize)
                            .collect();
                        (fixed.quantity, route)
                    })
                    .collect();
                (
                    format!("{:?}", output.status),
                    output.bound,
                    output.incumbent,
                    output.nodes,
                    output.pricing_calls,
                    routes,
                )
            };
        println!(
            "instance={path} gamma_q={} gamma_t={} sigma_q={} sigma_t={} ng={}",
            parameters.demand_budget,
            parameters.time_budget,
            parameters.demand_deviation,
            parameters.time_deviation,
            parameters
                .ng_size
                .map_or_else(|| "elementary".to_owned(), |n| n.to_string())
        );
        println!(
            "status={status} bound={bound:.6} incumbent={incumbent:.6} nodes={nodes} pricing_calls={pricing_calls} seconds={:.3}",
            started.elapsed().as_secs_f64()
        );
        let mut coverage = vec![0.; problem.instance().requests()];
        let mut cost = 0.;
        for (quantity, route) in &routes {
            let schedule = problem.replay(route)?;
            for &v in route {
                if (1..=coverage.len()).contains(&v) {
                    coverage[v - 1] += quantity;
                }
            }
            cost += quantity * schedule.cost;
            println!(
                "quantity={quantity} route={route:?} cost={:.6} worst_load={:.3} worst_return={:.3}",
                schedule.cost,
                schedule.loads.iter().copied().fold(0., f64::max),
                schedule.starts.last().unwrap().last().unwrap()
            );
        }
        if incumbent.is_finite()
            && (coverage.iter().any(|x| (*x - 1.).abs() > 1e-6) || (cost - incumbent).abs() > 1e-5)
        {
            return Err(invalid("incumbent replay/coverage mismatch"));
        }
        Ok(())
    }
    solve().map_err(|e| e.to_string().into())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("robust_pdptw requires --features clp");
    std::process::exit(1);
}
