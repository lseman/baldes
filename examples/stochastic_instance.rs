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
        "usage: stochastic_instance INSTANCE.json [--solve] [--seconds N] [--policy p1|p3] [--json] [--max-labels N] [--bidirectional] [--ng-size N] [--dynamic-ng-mode -1|0|1] [--hgs-seconds N] [--hgs-seed N] [--capacity-cut-rounds N] [--pricing-parts N] [--no-stabilize] [--completion-subset-size N] [--label-memory-gb N] [--no-knapsack-bound] [--recourse-cuts] [--rank-one-cuts] [--rank-one-cut-limit N] [--log-cg] [--no-load-completion-bound]",
    )?;
    let mut bidirectional = false;
    let mut solve = false;
    let mut json = false;
    let mut ng_size = 8;
    let mut dynamic_ng_mode = None;
    let mut max_labels = 1_000_000;
    let mut seconds = 30.0_f64;
    let mut hgs_seconds = None;
    let mut hgs_seed = 0;
    let mut capacity_cut_rounds = None;
    let mut pricing_parts = 0;
    let mut stabilize = true;
    let mut label_memory_gb = None;
    let mut knapsack_bound = true;
    let mut recourse_cuts = false;
    let mut rank_one_cuts = false;
    let mut rank_one_cut_limit = None;
    let mut log_cg = false;
    let mut load_completion_bound = true;
    let mut completion_subset_size = None;
    let mut policy = Policy::OptimalRestocking;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--solve" => solve = true,
            "--bidirectional" => bidirectional = true,
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
            "--max-labels" => max_labels = args.next().ok_or("missing max labels")?.parse()?,
            "--seconds" => seconds = args.next().ok_or("missing seconds")?.parse()?,
            "--hgs-seconds" => {
                hgs_seconds = Some(args.next().ok_or("missing HGS seconds")?.parse::<f64>()?)
            }
            "--hgs-seed" => hgs_seed = args.next().ok_or("missing HGS seed")?.parse()?,
            "--capacity-cut-rounds" => {
                capacity_cut_rounds = Some(args.next().ok_or("missing cut rounds")?.parse()?)
            }
            "--no-stabilize" => stabilize = false,
            "--no-knapsack-bound" => knapsack_bound = false,
            "--recourse-cuts" => recourse_cuts = true,
            "--rank-one-cuts" => rank_one_cuts = true,
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
            "--label-memory-gb" => {
                label_memory_gb = Some(args.next().ok_or("missing label memory")?.parse::<f64>()?)
            }
            "--completion-subset-size" => {
                completion_subset_size = Some(
                    args.next()
                        .ok_or("missing completion subset size")?
                        .parse()?,
                )
            }
            "--pricing-parts" => {
                pricing_parts = args.next().ok_or("missing pricing parts")?.parse()?
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
    if dynamic_ng_mode.is_some() && !recourse_cuts {
        return Err("--dynamic-ng-mode requires --recourse-cuts".into());
    }
    if rank_one_cut_limit.is_some() && !recourse_cuts {
        return Err("--rank-one-cut-limit requires --recourse-cuts".into());
    }
    if (rank_one_cuts || log_cg || !load_completion_bound) && !recourse_cuts {
        return Err(
            "--rank-one-cuts, --log-cg and --no-load-completion-bound require --recourse-cuts"
                .into(),
        );
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
    if solve && recourse_cuts {
        // Stochastic routing enables rank-one and mean-capacity cuts by default.
        rank_one_cuts = true;
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
        let answer = problem
            .solve_theta_cuts(policy, input.vehicles, ng_size.max(1), tree)
            .map_err(|error| error.message)?;
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
                "time_limit_seconds": seconds, "build_seconds": build_seconds,
                "solve_seconds": solve_seconds, "total_seconds": total_started.elapsed().as_secs_f64(),
            })
        );
        return Ok(());
    }
    if solve {
        let mut options = SolveOptions {
            time_limit,
            ..Default::default()
        };
        options.column_generation.ng_size = ng_size;
        options.column_generation.pricing.max_labels = max_labels;
        options.column_generation.pricing.bidirectional = bidirectional;
        options.hgs_warm_start = hgs_warm_start;
        if let Some(rounds) = capacity_cut_rounds {
            options.column_generation.capacity_cut_rounds = rounds;
        }
        options.column_generation.pricing_parts = pricing_parts;
        options.column_generation.stabilize = stabilize;
        options.column_generation.pricing.completion_knapsack = knapsack_bound;
        if let Some(gb) = label_memory_gb {
            options.column_generation.pricing_label_memory = (gb * (1u64 << 30) as f64) as usize;
        }
        if let Some(size) = completion_subset_size {
            options.column_generation.pricing.completion_subset_size = size;
        }
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
                    "capacity_cut_rounds": options.column_generation.capacity_cut_rounds,
                    "pricing_parts": pricing_parts, "stabilize": stabilize, "knapsack_bound": knapsack_bound, "recourse_cuts": recourse_cuts,
                    "pricing_label_memory": options.column_generation.pricing_label_memory,
                    "completion_subset_size": options.column_generation.pricing.completion_subset_size,
                    "ng_memory_capacity": baldes::graph::ng::NG_MEMORY_CAPACITY,
                    "hgs_warm_start": answer.hgs_warm_start.map(|s| serde_json::json!({
                        "accepted_columns": s.accepted_columns, "rejected_columns": s.rejected_columns,
                        "accepted_plan": s.accepted_plan, "elapsed_seconds": s.elapsed_seconds,
                    })),
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
