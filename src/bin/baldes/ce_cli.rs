use baldes::extensions::ce::{Options, Problem, Status};
use baldes::inputs::{Instance, Rounding};
use baldes::master::solvers::ClpMaster;
use std::time::{Duration, Instant};

const HELP: &str = "Usage: baldes ce INSTANCE [OPTIONS]

Solve the CVRP/VRPTW route LP using column elimination.
Respects capacity, time windows, service times and the instance fleet size.
Returns a numerical LP lower bound; route flows may be fractional.

Options:
  --solver NAME         LP backend: clp (default) or highs (requires highs build)
  --master NAME         Alias for --solver
  --threads N           HiGHS worker threads; 0 = all available CPUs (default)
  --time SECONDS        Overall solve time limit (default: 300)
  --iterations N        LP solve limit (default: 1000)
  --nodes N             Decision-diagram node limit (default: 100000)
  --arcs N              Decision-diagram arc limit (default: 1000000)
  --customers N         Keep the first N customers
  --rounding MODE       none, nearest, tenth or ceil (default: tenth)
  --unlimited-fleet     Ignore the instance fleet limit
  -v, --verbose, --log  Show progress on stderr every 10 LP iterations
  --log-interval N      Show progress every N LP iterations (enables logging)
  --json FILE           Write bound, route flows and statistics as JSON
  -h, --help            Print help

Capacity constraints are always enforced. Rank-1 and capacity cuts are not
implemented in CE; --rank1 and --capacity belong to the main solver.";

pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let file = args.next().ok_or(HELP)?;
    if file == "--help" || file == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if file.starts_with('-') {
        return Err(format!("expected instance path\n{HELP}"));
    }
    let mut options = Options::default();
    let mut customers = None;
    let mut rounding = Some(Rounding::Tenth);
    let mut json_path = None;
    let mut unlimited_fleet = false;
    let mut solver = String::from("clp");
    let mut threads = None;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        if arg == "--unlimited-fleet" {
            unlimited_fleet = true;
            continue;
        }
        if matches!(arg.as_str(), "--rank1" | "--capacity") {
            return Err(format!("{arg} cuts are not implemented in column elimination; capacity constraints are already enforced. Remove this option to run CE, or use the main solver for cuts."));
        }
        if matches!(arg.as_str(), "-v" | "--verbose" | "--log") {
            options.log_interval.get_or_insert(10);
            continue;
        }
        let value = match arg.as_str() {
            "--time" | "--iterations" | "--nodes" | "--arcs" | "--customers" | "--rounding"
            | "--json" | "--log-interval" | "--solver" | "--master" | "--threads" => args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?,
            _ => return Err(format!("unknown ce option: {arg}\n{HELP}")),
        };
        let integer = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid {arg}: {value}"))
        };
        match arg.as_str() {
            "--solver" | "--master" => {
                if !matches!(value.as_str(), "clp" | "highs") {
                    return Err(format!("unknown CE solver: {value}; expected clp or highs"));
                }
                solver = value;
            }
            "--threads" => threads = Some(integer()?),
            "--time" => {
                let seconds = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid time: {value}"))?;
                options.time_limit = Duration::try_from_secs_f64(seconds)
                    .map_err(|_| format!("invalid time: {value}"))?;
            }
            "--iterations" => options.max_iterations = integer()?,
            "--nodes" => options.max_nodes = integer()?,
            "--arcs" => options.max_arcs = integer()?,
            "--log-interval" => {
                let interval = integer()?;
                if interval == 0 {
                    return Err("--log-interval must be positive".into());
                }
                options.log_interval = Some(interval);
            }
            "--customers" => customers = Some(integer()?),
            "--rounding" => rounding = Some(Rounding::parse(&value).map_err(|e| e.to_string())?),
            "--json" => json_path = Some(value),
            _ => unreachable!(),
        }
    }
    if threads.is_some() && solver != "highs" {
        return Err("--threads requires --solver highs".into());
    }
    #[cfg(not(feature = "highs"))]
    if solver == "highs" {
        return Err("CE HiGHS backend requires a build with --features clp,highs".into());
    }
    let mut instance = Instance::read(&file).map_err(|e| e.to_string())?;
    if let Some(count) = customers {
        instance.truncate(count).map_err(|e| e.to_string())?;
    }
    if let Some(mode) = rounding {
        instance.rounding = mode;
    }
    let mut problem = Problem::from_instance(&instance).map_err(|e| e.to_string())?;
    if unlimited_fleet {
        problem.max_vehicles = None;
    }
    let started = Instant::now();
    let (result, worker_threads, simplex_concurrency) = if solver == "highs" {
        #[cfg(feature = "highs")]
        {
            use baldes::master::solvers::HighsMaster;
            let configured =
                HighsMaster::with_threads(threads.unwrap_or(0)).map_err(|e| e.to_string())?;
            let workers = configured.configured_threads().unwrap();
            let concurrency = configured.simplex_concurrency().unwrap();
            if options.log_interval.is_some() {
                eprintln!(
                    "  HiGHS | {workers} worker threads | simplex concurrency limit {concurrency}"
                );
            }
            drop(configured);
            let result = problem
                .solve_column_elimination(|| HighsMaster::with_threads(workers), options)
                .map_err(|e| e.to_string())?;
            (result, workers, Some(concurrency))
        }
        #[cfg(not(feature = "highs"))]
        unreachable!("HiGHS availability was checked before reading the instance")
    } else {
        (
            problem
                .solve_column_elimination(ClpMaster::new, options)
                .map_err(|e| e.to_string())?,
            1usize,
            None::<usize>,
        )
    };
    let elapsed = started.elapsed().as_secs_f64();
    println!("instance={} status={:?} lower_bound={} refinements={} nodes={} arcs={} iterations={} elapsed={elapsed:.3}s",
        instance.name, result.status,
        result.lower_bound.map_or("none".into(), |v| format!("{v:.6}")),
        result.refinements, result.nodes, result.arcs, result.iterations);
    if let Some(path) = json_path {
        let routes: Vec<_> = result
            .routes
            .iter()
            .map(|r| {
                serde_json::json!({
                    "path": r.path, "flow": r.flow, "cost": r.cost
                })
            })
            .collect();
        let report = serde_json::json!({
            "instance": instance.name, "customers": problem.demands.len(),
            "method": "column_elimination", "relaxation": "route_lp",
            "solver": solver, "threads": worker_threads,
            "simplex_concurrency": simplex_concurrency,
            "status": format!("{:?}", result.status),
            "lower_bound": result.lower_bound,
            "lp_bound": if result.status == Status::Converged { result.lower_bound } else { None },
            "max_vehicles": problem.max_vehicles, "bound_mode": "numerical",
            "iterations": result.iterations, "refinements": result.refinements,
            "nodes": result.nodes, "arcs": result.arcs,
            "elapsed_seconds": elapsed, "routes": routes,
        });
        std::fs::write(
            path,
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/bin/baldes/ce_cli/tests.rs"]
mod tests;
