//! Load synthetic paper-style JSON; validate by default, solve only with --solve.
use baldes::extensions::stochastic::{Demand, HgsWarmStart, Policy, Problem, SolveOptions};
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
        "usage: stochastic_instance INSTANCE.json [--solve] [--seconds N] [--policy p1|p3] [--json] [--ng-size N] [--dynamic-ng-mode -1|0|1] [--hgs-seconds N] [--hgs-seed N] [--recourse-cuts] [--rank-one-cuts] [--rank-one-cut-limit N] [--log-cg] [--no-load-completion-bound] [--completion-subset-size 0..6]",
    )?;
    let mut solve = false;
    let mut json = false;
    let mut ng_size = 8;
    let mut dynamic_ng_mode = None;
    let mut seconds = 30.0_f64;
    let mut hgs_seconds = None;
    let mut hgs_seed = 0;
    let mut rank_one_cut_limit = None;
    let mut log_cg = false;
    let mut load_completion_bound = true;
    let mut completion_subset_size = 4;
    let mut policy = Policy::OptimalRestocking;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--solve" => solve = true,
            "--json" => json = true,
            "--ng-size" => ng_size = args.next().ok_or("missing NG size")?.parse()?,
            "--dynamic-ng-mode" => {
                let mode = args
                    .next()
                    .ok_or("missing dynamic NG mode")?
                    .parse::<i32>()?;
                if !(-1..=1).contains(&mode) {
                    return Err("dynamic NG mode must be -1, 0 or 1".into());
                }
                dynamic_ng_mode = Some(mode);
            }
            "--seconds" => seconds = args.next().ok_or("missing seconds")?.parse()?,
            "--hgs-seconds" => {
                hgs_seconds = Some(args.next().ok_or("missing HGS seconds")?.parse::<f64>()?)
            }
            "--hgs-seed" => hgs_seed = args.next().ok_or("missing HGS seed")?.parse()?,
            // Compatibility alias: every solve now uses Theta recourse cuts.
            "--recourse-cuts" => {}
            "--rank-one-cuts" => {}
            "--rank-one-cut-limit" => {
                let limit = args
                    .next()
                    .ok_or("missing rank-one cut limit")?
                    .parse::<usize>()?;
                if limit == 0 {
                    return Err("rank-one cut limit must be positive".into());
                }
                rank_one_cut_limit = Some(limit);
            }
            "--log-cg" => log_cg = true,
            "--no-load-completion-bound" => load_completion_bound = false,
            "--completion-subset-size" => {
                completion_subset_size = args
                    .next()
                    .ok_or("missing completion subset size")?
                    .parse::<usize>()?;
                if completion_subset_size > 6 {
                    return Err("completion subset size must be between 0 and 6".into());
                }
            }
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
    let hgs_warm_start = hgs_seconds
        .map(|seconds| -> Result<_, Box<dyn Error>> {
            let time_limit = Duration::try_from_secs_f64(seconds)?;
            if time_limit.is_zero() {
                return Err("HGS seconds must be positive".into());
            }
            Ok(HgsWarmStart {
                time_limit,
                seed: hgs_seed,
            })
        })
        .transpose()?;
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
        // Stochastic routing enables rank-one and mean-capacity cuts by default.
        let rank_one_cuts = true;
        rank_one_cut_limit = Some(rank_one_cut_limit.unwrap_or(8));
        // Classic routing branch-cut-and-price; recourse enters by Theta cuts.
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        let mut tree = baldes::master::tree::TreeOptions {
            threads,
            time_limit,
            ..Default::default()
        };
        tree.root.cg.time_limit = time_limit;
        let dynamic_ng_mode = dynamic_ng_mode.unwrap_or(-1);
        tree.root
            .pricing_options
            .push(("BALDESdynamicNGmode".into(), dynamic_ng_mode as f64));
        if !load_completion_bound {
            tree.root
                .pricing_options
                .push(("BALDESloadCompletionResource".into(), -1.));
        }
        tree.root.pricing_options.push((
            "BALDESloadCompletionSubsetSize".into(),
            completion_subset_size as f64,
        ));
        if rank_one_cuts {
            tree.root = tree.root.with_rank_one_cuts();
            if let Some(limit) = rank_one_cut_limit {
                tree.root
                    .pricing_options
                    .push(("BALDESrankOneCutsMaxNumPerRound".into(), limit as f64));
            }
        }
        if log_cg {
            tree.root.cg.log_interval = Some(1);
        }
        let solve_started = Instant::now();
        let result = problem
            .solve(
                policy,
                input.vehicles,
                SolveOptions {
                    tree,
                    ng_size,
                    hgs_warm_start,
                },
            )
            .map_err(|error| error.message)?;
        let hgs_summary = result.hgs_warm_start;
        let answer = result.tree;
        let solve_seconds = solve_started.elapsed().as_secs_f64();
        let objective = answer.incumbent.is_finite().then_some(answer.incumbent);
        let lower_bound = answer.bound.is_finite().then_some(answer.bound);
        let gap = objective.and_then(|upper| {
            lower_bound.map(|lower| (upper - lower).max(0.) / upper.abs().max(1e-10))
        });
        let routes: Vec<Vec<i32>> = answer
            .routes
            .iter()
            .map(|r| r.nodes[1..r.nodes.len() - 1].to_vec())
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "method": "baldes_theta_cut_branch_cut_price", "backend": "Clp",
                "instance_id": input.instance_id, "policy": format!("{policy:?}"),
                "status": format!("{:?}", answer.status), "objective": objective,
                "lower_bound": lower_bound, "gap": gap, "routes": routes,
                "nodes": answer.nodes, "pricing_calls": answer.pricing_calls,
                "fixing_calls": answer.fixing_calls, "enumerated_routes": answer.enumerated_routes,
                "mip_nodes": answer.mip_nodes, "strong_branching_trials": answer.strong_branching_trials,
                "diving_nodes": answer.diving_nodes,
                "threads": threads, "ng_size": ng_size, "dynamic_ng_mode": dynamic_ng_mode,
                "rank_one_cuts": rank_one_cuts,
                "rank_one_cut_limit": rank_one_cut_limit,
                "capacity_cuts": true,
                "load_completion_bound": load_completion_bound,
                "completion_subset_size": completion_subset_size,
                "hgs_warm_start": hgs_summary.map(|s| serde_json::json!({
                    "accepted_columns": s.accepted_columns, "rejected_columns": s.rejected_columns,
                    "accepted_plan": s.accepted_plan, "initial_objective": s.initial_objective,
                    "proposal_objective": s.proposal_objective, "reversed_routes": s.reversed_routes,
                    "elapsed_seconds": s.elapsed_seconds,
                })),
                "time_limit_seconds": seconds, "build_seconds": build_seconds,
                "solve_seconds": solve_seconds, "total_seconds": total_started.elapsed().as_secs_f64(),
            })
        );
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
