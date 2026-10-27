//! Load synthetic paper-style JSON; validate by default, solve only with --solve.
use baldes::extensions::stochastic::{Demand, Policy, Problem, SolveOptions};
use serde::Deserialize;
use std::{
    error::Error,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct InputDemand {
    customer: usize,
    support: Vec<usize>,
    probabilities: Vec<f64>,
}
#[derive(Deserialize)]
struct Provenance {
    original_benchmark_instance: bool,
    failure_penalty_applied: f64,
}
#[derive(Deserialize)]
struct Input {
    schema: String,
    instance_id: String,
    provenance: Provenance,
    nodes_including_depot: usize,
    depot: usize,
    vehicles: usize,
    capacity: usize,
    demands: Vec<InputDemand>,
    costs: Vec<Vec<f64>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_started = Instant::now();
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(
        "usage: stochastic_instance INSTANCE.json [--solve] [--seconds N] [--policy p1|p3] [--json] [--max-labels N] [--bidirectional] [--ng-size N]",
    )?;
    let mut bidirectional = false;
    let mut solve = false;
    let mut json = false;
    let mut ng_size = 8;
    let mut max_labels = 1_000_000;
    let mut seconds = 30.0_f64;
    let mut policy = Policy::OptimalRestocking;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--solve" => solve = true,
            "--bidirectional" => bidirectional = true,
            "--json" => json = true,
            "--ng-size" => ng_size = args.next().ok_or("missing NG size")?.parse()?,
            "--max-labels" => max_labels = args.next().ok_or("missing max labels")?.parse()?,
            "--seconds" => seconds = args.next().ok_or("missing seconds")?.parse()?,
            "--policy" => {
                policy = match args.next().as_deref() {
                    Some("p1") => Policy::Classical,
                    Some("p3") => Policy::OptimalRestocking,
                    _ => return Err("policy must be p1 or p3".into()),
                }
            }
            _ => return Err(format!("unknown option: {option}").into()),
        }
    }
    let time_limit = Duration::try_from_secs_f64(seconds)?;
    let input: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    if input.schema != "baldes.stochastic.generated.v1"
        || input.provenance.original_benchmark_instance
        || input.provenance.failure_penalty_applied != 0.
        || input.depot != 0
        || input.vehicles == 0
        || input.nodes_including_depot != input.demands.len() + 1
    {
        return Err("unsupported schema, metadata, depot, fleet, or failure-penalty model".into());
    }
    let demands = input
        .demands
        .into_iter()
        .enumerate()
        .map(|(index, demand)| {
            if demand.customer != index + 1 || demand.support.len() != demand.probabilities.len() {
                return Err("invalid customer numbering or demand probability shape".into());
            }
            Demand::new(
                demand
                    .support
                    .into_iter()
                    .zip(demand.probabilities)
                    .collect(),
            )
            .map_err(|error| Box::<dyn Error>::from(error.message))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let problem =
        Problem::new(input.capacity, demands, input.costs).map_err(|error| error.message)?;
    let build_seconds = total_started.elapsed().as_secs_f64();
    if solve {
        let mut options = SolveOptions {
            time_limit,
            ..Default::default()
        };
        options.column_generation.ng_size = ng_size;
        options.column_generation.pricing.max_labels = max_labels;
        options.column_generation.pricing.bidirectional = bidirectional;
        let solve_started = Instant::now();
        let answer = problem
            .solve(policy, input.vehicles, options)
            .map_err(|error| error.message)?;
        let solve_seconds = solve_started.elapsed().as_secs_f64();
        if json {
            let objective = answer.solution.as_ref().map(|s| s.total_cost);
            let lower_bound = answer.lower_bound.is_finite().then_some(answer.lower_bound);
            let gap = objective.and_then(|upper| {
                lower_bound.map(|lower| (upper - lower).max(0.) / upper.abs().max(1e-10))
            });
            println!(
                "{}",
                serde_json::json!({
                    "method": "baldes_stochastic_branch_price", "backend": "Clp",
                    "instance_id": input.instance_id, "policy": format!("{policy:?}"),
                    "status": format!("{:?}", answer.status), "objective": objective,
                    "lower_bound": lower_bound, "gap": gap,
                    "routes": answer.solution.as_ref().map(|s| &s.routes),
                    "nodes": answer.nodes, "pricing_calls": answer.pricing_calls,
                    "generated_labels": answer.generated_labels, "max_labels": max_labels,
                    "bidirectional": bidirectional, "ng_size": ng_size,
                    "ng_memory_capacity": baldes::graph::ng::NG_MEMORY_CAPACITY,
                    "time_limit_seconds": seconds, "build_seconds": build_seconds,
                    "solve_seconds": solve_seconds, "total_seconds": total_started.elapsed().as_secs_f64(),
                })
            );
        } else {
            println!("{answer:#?}");
        }
    } else if json {
        println!(
            "{}",
            serde_json::json!({"validation": "ok", "instance_id": input.instance_id,
            "customers": problem.demands().len(), "ng_memory_capacity": baldes::graph::ng::NG_MEMORY_CAPACITY})
        );
    } else {
        println!(
            "Validated {}: {} customers, {} vehicles, capacity {}, NEW synthetic data",
            input.instance_id,
            problem.demands().len(),
            input.vehicles,
            input.capacity
        );
    }
    Ok(())
}
