use super::*;

pub(super) const NETWORK_HELP: &str = "Usage: baldes solve MODEL.json [OPTIONS]

Solve shared master rows with multiple pricing networks (Clp).
  --tree                 Run the assignment branch-and-price tree
  --network-pricing-schedule MODE  full-sweep (default), cyclic, sorted,
                            promising-first or unvisited-first
  --network-promising-limit N  positive-status budget (default: 1000000)
  --stabilization-max-tree-depth N  Disable stabilization at depth N (default: 10000)
  --network-unpromising-limit N  Per-batch queue budget (default: 1)
  --network-branch-priorities FILE  Family priorities for phased JSON
  --network-branch-directions FILE  Generator-name to upper-first/lower-first JSON (--tree)
  --mapped-variables FILE  Registered original aggregate branch variables JSON
  --packing FILE  packing-assignment registration JSON (--tree)
  --original-variable-boxes FILE  Original identity lower/upper envelopes JSON
  --variable-boxes FILE  Verified variable-box envelopes JSON
  --no-mapped-strong-branching  Rank registered aggregates without trial solves
  --proximal FILE         Proximal configuration JSON (replaces smoothing)
  --proximal-finite-box-step  finite-cap box-step initialization/schedule
  --proximal-explicit-three-piece  explicit three-piece initialization/schedule
  --proximal-explicit-five-piece  explicit five-piece initialization/schedule
  --proximal-explicit-multipoint  explicit multi-point initialization/schedule
  --automatic-curvature   Automatic three-piece curvature stabilization
  --automatic-curvature-five-piece  Automatic five-piece curvature stabilization
  --proximal-width X      One proximal band, in dual row units
  --proximal-penalty X    Finite band penalty; omitted means hard box-step
  --network-branching MODE  multiplicity-first (default), resource-first or phased
  --network-tree-search MODE  default (default), best-bound, bound-priority or depth-bound-priority
  --network-strategy2-rounding MODE  unrounded (default), guarded-tolerance or exact-ceil
  --network-open-node-limit N  primary frontier size (default: 1000)
  --network-scoring-gap-tolerance X  branch-score absolute gap (default: 1e-6)
  --network-search-inactive-columns  Scan retained inactive columns before pricing
  --network-share-inactive-columns  Reprice shared tree/trial paths (requires --tree)
  --synchronized-services FILE  JSON array of common-start service definitions
  --resource-branches FILE  Apply a JSON array of grouped-resource restrictions
  --enumerate            Attempt bounded full-domain all-network transfer
  --enumerated-mip       Solve transferred nodes with HiGHS (--tree; highs feature)
  --enumeration-route-limit N  Global enumeration candidate limit (default: 10000)
  --enumeration-time SECONDS   Enumeration budget per attempt (default: 1)
  --mip-time SECONDS     MIP budget per node (default: 1)
  --mip-nodes N          MIP search-node budget per call (default: 100000)
  --mip-cutoff X         Search for objectives below X; no incumbent is implied
  --time SECONDS         Total time budget (default: 120)
  --iterations N         Column-generation iteration limit per node
  --nodes N              Assignment tree node limit
  --bound-mode MODE      numerical (default) or certified
  --no-stabilization     Disable dual stabilization
  --artificial-cost X    Initial positive artificial penalty
  --option NAME=VALUE    Pricing option applied to every network (repeatable)
  --log-interval N       Progress logging interval
  -v, --verbose          Log every ten CG iterations
  --json FILE            Write structured results; '-' writes JSON to stdout
  -h, --help             Print help

Model schema and examples: docs/extensions/mp.md";

struct NetworkOptions {
    options: SolveOptions,
    json: Option<String>,
    resource_branches: Option<String>,
}

fn parse_options(args: Vec<String>) -> Result<Option<NetworkOptions>, String> {
    let mut args = args.into_iter();
    let mut common = CommonOptions::default();
    let mut options = SolveOptions::default();
    let mut resource_branches = None;
    let mut proximal_width = None;
    let mut proximal_penalty = None;
    while let Some(arg) = args.next() {
        if common.parse(&arg, &mut args)? {
            continue;
        }
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing {}", arg.trim_start_matches('-').replace('-', " ")))
        };
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{NETWORK_HELP}");
                return Ok(None);
            }
            "--proximal-finite-box-step" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::finite_box_step());
            }
            "--proximal-explicit-three-piece" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::explicit_three_piece());
            }
            "--proximal-explicit-five-piece" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::explicit_five_piece());
            }
            "--proximal-explicit-multipoint" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::explicit_multipoint());
            }
            "--automatic-curvature" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::automatic_three_piece())
            }
            "--automatic-curvature-five-piece" => {
                options.proximal =
                    Some(baldes::master::stabilization::ProximalParameters::automatic_five_piece())
            }
            "--network-promising-limit" => {
                options.network_promising_limit =
                    value()?.parse::<usize>().map_err(|e| e.to_string())?;
            }
            "--network-unpromising-limit" => {
                options.network_unpromising_limit =
                    value()?.parse::<usize>().map_err(|e| e.to_string())?;
            }
            "--network-branch-priorities" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.network_branch_priorities = serde_json::from_str(&text)
                    .map_err(|e| format!("network branch priorities JSON: {e}"))?;
            }
            "--network-branch-directions" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.network_branch_directions = serde_json::from_str(&text)
                    .map_err(|e| format!("network branch directions JSON: {e}"))?;
            }
            "--original-variable-boxes" => {
                options.original_variable_boxes = serde_json::from_str(
                    &std::fs::read_to_string(value()?).map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("original variable boxes JSON: {e}"))?;
            }
            "--variable-boxes" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.variable_boxes =
                    serde_json::from_str(&text).map_err(|e| format!("variable boxes JSON: {e}"))?;
            }
            "--packing" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.packing =
                    Some(serde_json::from_str(&text).map_err(|e| format!("packing JSON: {e}"))?);
            }
            "--mapped-variables" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.mapped_variables = serde_json::from_str(&text)
                    .map_err(|e| format!("mapped variables JSON: {e}"))?;
            }
            "--no-mapped-strong-branching" => options.mapped_strong_branching = false,
            "--proximal" => {
                let text = std::fs::read_to_string(value()?).map_err(|e| e.to_string())?;
                options.proximal =
                    Some(serde_json::from_str(&text).map_err(|e| format!("proximal JSON: {e}"))?);
            }
            "--proximal-width" => {
                proximal_width = Some(value()?.parse::<f64>().map_err(|e| e.to_string())?)
            }
            "--proximal-penalty" => {
                proximal_penalty = Some(value()?.parse::<f64>().map_err(|e| e.to_string())?)
            }
            "--network-pricing-schedule" => {
                options.network_pricing_schedule =
                    model::NetworkPricingSchedule::parse(&value()?).map_err(|e| e.to_string())?
            }
            "--network-branching" => {
                options.network_branching =
                    baldes::master::tree::NetworkBranchingPolicy::parse(&value()?)
                        .map_err(|e| e.to_string())?
            }
            "--network-tree-search" => {
                options.network_tree_search =
                    baldes::master::tree::TreeSearchPolicy::parse(&value()?)
                        .map_err(|e| e.to_string())?
            }
            "--network-strategy2-rounding" => {
                options.network_strategy2_bound_rounding = match value()?.as_str() {
                    "unrounded" => baldes::master::tree::IntegerBoundRounding::Unrounded,
                    "guarded-tolerance" => {
                        baldes::master::tree::IntegerBoundRounding::GuardedTolerance
                    }
                    "exact-ceil" => baldes::master::tree::IntegerBoundRounding::ExactCeil,
                    _ => return Err(
                        "strategy-2 rounding must be unrounded, guarded-tolerance or exact-ceil"
                            .into(),
                    ),
                }
            }
            "--network-search-inactive-columns" => options.network_search_inactive_columns = true,
            "--network-share-inactive-columns" => options.network_share_inactive_columns = true,
            "--network-scoring-gap-tolerance" => {
                options.network_scoring_gap_tolerance =
                    value()?.parse::<f64>().map_err(|e| e.to_string())?;
            }
            "--network-open-node-limit" => {
                options.network_open_node_limit =
                    value()?.parse::<usize>().map_err(|e| e.to_string())?;
            }
            "--resource-branches" => resource_branches = Some(value()?),
            "--enumerate" => options.enumerate = true,
            "--enumerated-mip" => options.enumerated_mip = true,
            "--enumeration-route-limit" => {
                options.enumeration_route_limit =
                    value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            "--enumeration-time" => {
                options.enumeration_time_limit =
                    value()?.parse::<f64>().map_err(|e| e.to_string())?
            }
            "--mip-time" => {
                options.mip_time_limit = value()?.parse::<f64>().map_err(|e| e.to_string())?
            }
            "--mip-cutoff" => {
                options.mip_objective_cutoff =
                    Some(value()?.parse::<f64>().map_err(|e| e.to_string())?);
            }
            "--mip-nodes" => {
                options.mip_node_limit = value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            "--artificial-cost" => {
                options.artificial_cost = value()?.parse::<f64>().map_err(|e| e.to_string())?
            }
            "--synchronized-services" => {
                let path = value()?;
                let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
                options.synchronized_services = serde_json::from_str(&text)
                    .map_err(|e| format!("synchronized services JSON: {e}"))?;
            }
            _ => return Err(format!("unknown networks option: {arg}\n{NETWORK_HELP}")),
        }
    }
    common.apply_networks(&mut options);
    let json = common.json;
    if proximal_penalty.is_some() && proximal_width.is_none() {
        return Err("--proximal-penalty requires --proximal-width".into());
    }
    if let Some(width) = proximal_width {
        if options.proximal.is_some() {
            return Err("choose --proximal or --proximal-width".into());
        }
        options.proximal = Some(
            baldes::master::stabilization::ProximalParameters::three_piece(
                width,
                proximal_penalty.unwrap_or(f64::INFINITY),
            ),
        );
    }
    Ok(Some(NetworkOptions {
        options,
        json,
        resource_branches,
    }))
}

pub(super) fn solve_networks(file: String, args: Vec<String>) -> Result<(), String> {
    let Some(NetworkOptions {
        options,
        json,
        resource_branches,
    }) = parse_options(args)?
    else {
        return Ok(());
    };
    let text = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let (problem, registry) = Model::from_json(&text)
        .and_then(Model::compile::<{ baldes::DEFAULT_RESOURCE_DIMENSIONS }>)
        .map_err(|e| e.to_string())?;
    let node_options = options
        .node_options_with_registry(&registry)
        .map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let result = if let Some(path) = resource_branches {
        let requests: Vec<ResourceGroupRestriction> =
            serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("resource branches JSON: {e}"))?;
        // Construct the master without spending a preliminary LP solve. Each
        // restriction validates and forks it before the requested root/tree.
        let mut initial = node_options.clone();
        initial.cg.time_limit = std::time::Duration::ZERO;
        initial.enumeration = None;
        let mut node = model::solve(problem, initial)
            .map_err(|e| e.to_string())?
            .snapshot;
        for request in requests {
            node = request.apply(&node).map_err(|e| e.to_string())?;
        }
        if options.tree {
            let mut configured = options
                .tree_options_with_registry(&registry)
                .map_err(|e| e.to_string())?;
            configured.time_limit = configured.time_limit.saturating_sub(started.elapsed());
            let output = node.solve_tree(configured).map_err(|e| e.to_string())?;
            SolveResult::tree(&output, started.elapsed().as_secs_f64())
        } else {
            let mut configured = node_options;
            configured.cg.time_limit = configured.cg.time_limit.saturating_sub(started.elapsed());
            let output = node.solve(configured).map_err(|e| e.to_string())?;
            SolveResult::root(&output, started.elapsed().as_secs_f64())
        }
    } else if options.tree {
        let configured = options
            .tree_options_with_registry(&registry)
            .map_err(|e| e.to_string())?;
        let output = baldes::master::tree::solve(problem, configured).map_err(|e| e.to_string())?;
        SolveResult::tree(&output, started.elapsed().as_secs_f64())
    } else {
        let output = model::solve(problem, node_options).map_err(|e| e.to_string())?;
        SolveResult::root(&output, started.elapsed().as_secs_f64())
    };
    let result = result
        .with_original_variables(&registry)
        .map_err(|e| e.to_string())?;
    let data = result.to_json().map_err(|e| e.to_string())?;
    if let Some(path) = json {
        if write_json(&path, &data)? {
            return Ok(());
        }
    }
    {
        println!(
            "mode={} status={} lp={} incumbent={} bound={} bound_mode={} nodes={} tree_processed_nodes={} open_nodes={}",
            result.mode,
            result.status,
            result.lp.map_or("unknown".into(), |v| v.to_string()),
            result.incumbent.map_or("unknown".into(), |v| v.to_string()),
            result.bound.map_or("unknown".into(), |v| v.to_string()),
            result.bound_mode,
            result.nodes,
            result.tree_processed_nodes,
            result.open_nodes
        );
    }
    Ok(())
}
