use baldes::extensions::pstep::{ColumnGenerationOptions, EnumerationOptions, Problem};
use baldes::inputs::{Instance, Rounding};
use baldes::master::{clp::ClpMaster, LpMaster};
use std::time::Duration;

const HELP: &str = "Usage: baldes pstep INSTANCE [OPTIONS]

Solve the unlimited-fleet CVRP p-step LP using parallel bucket-graph pricing.
This command computes a numerical LP bound, not an integer routing solution.

Options:
  --p N                 Arcs per step (default: 2; depot starts may be shorter)
  --threads N           Parallel endpoint/extreme pricing workers (default: up to 8)
  --time SECONDS        Overall solve time limit (default: 300)
  --iterations N        LP solve limit (default: 1000)
  --labels N            Label limit per pricing job (default: 1000000)
  --customers N         Keep the first N customers
  --rounding MODE       none, nearest, tenth or ceil (default: file convention)
  --enumerate           Cross-check against the full column pool (small cases)
  --json FILE           Write result and pricing statistics as JSON
  -h, --help            Print help";

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
    let mut options = ColumnGenerationOptions::default();
    let mut json_path = None;
    let mut customers = None;
    let mut rounding = None;
    let mut enumerate = false;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        if arg == "--enumerate" {
            enumerate = true;
            continue;
        }
        let value = match arg.as_str() {
            "--p" | "--threads" | "--time" | "--iterations" | "--labels" | "--customers"
            | "--rounding" | "--json" => args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?,
            _ => return Err(format!("unknown pstep option: {arg}\n{HELP}")),
        };
        let integer = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid {arg}: {value}"))
        };
        match arg.as_str() {
            "--p" => options.pricing.p = integer()?,
            "--threads" => options.pricing.threads = integer()?,
            "--iterations" => options.max_iterations = integer()?,
            "--labels" => options.pricing.max_labels_per_job = integer()?,
            "--customers" => customers = Some(integer()?),
            "--time" => {
                let seconds = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid time: {value}"))?;
                options.time_limit =
                    Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())?;
                options.pricing.time_limit = options.time_limit;
            }
            "--rounding" => rounding = Some(Rounding::parse(&value).map_err(|e| e.to_string())?),
            "--json" => json_path = Some(value),
            _ => unreachable!(),
        }
    }
    let mut instance = Instance::parse(&std::fs::read_to_string(&file).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if let Some(count) = customers {
        instance.truncate(count).map_err(|e| e.to_string())?;
    }
    if let Some(mode) = rounding {
        instance.rounding = mode;
    }
    let problem = Problem::from_instance(&instance).map_err(|e| e.to_string())?;
    let p = options.pricing.p;
    let threads = options.pricing.threads;
    let result = problem
        .solve_column_generation(ClpMaster::new().map_err(|e| e.to_string())?, options)
        .map_err(|e| e.to_string())?;
    let mut enumerated_bound = None;
    if enumerate {
        let pool = problem
            .enumerate(EnumerationOptions {
                p,
                ..Default::default()
            })
            .map_err(|e| e.to_string())?;
        let mut lp = pool
            .into_master(ClpMaster::new().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if !lp.solve() {
            return Err("enumerated p-step LP did not solve to optimality".into());
        }
        enumerated_bound = Some(lp.objective());
        if result
            .lp_bound
            .is_some_and(|v| (v - lp.objective()).abs() > 1e-5)
        {
            return Err("p-step CG/enumeration bound mismatch".into());
        }
    }
    let report = serde_json::json!({
        "instance": instance.name, "customers": problem.demands.len(), "p": p,
        "requested_threads": threads, "pricing_workers": result.pricing_workers,
        "status": format!("{:?}", result.status), "lp_bound": result.lp_bound,
        "restricted_objective": result.restricted_objective, "enumerated_bound": enumerated_bound,
        "iterations": result.iterations, "columns": result.columns,
        "pricing_jobs": result.pricing_jobs, "generated_labels": result.generated_labels,
        "elapsed_seconds": result.elapsed_seconds, "pricing_seconds": result.pricing_seconds,
        "lp_seconds": result.lp_seconds, "fleet": "unlimited", "bound_mode": "numerical"
    });
    println!("instance={} p={p} status={:?} lp_bound={} workers={} columns={} iterations={} elapsed={:.3}s",
        instance.name, result.status, result.lp_bound.map_or("none".into(), |v| format!("{v:.6}")),
        result.pricing_workers, result.columns, result.iterations, result.elapsed_seconds);
    if let Some(path) = json_path {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
