use baldes::master::mp::{
    self,
    format::{Model, ResourceGroupRestriction, SolveOptions, SolveResult},
};

const HELP: &str = "Usage: baldes multi-pricing MODEL.json [OPTIONS]

Solve shared master rows with multiple pricing networks (Clp).
  --tree                 Run the assignment branch-and-price tree
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

pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let file = args.next().ok_or(HELP)?;
    if file == "--help" || file == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if file.starts_with('-') {
        return Err(format!("expected model path\n{HELP}"));
    }
    let mut options = SolveOptions::default();
    let mut json = None;
    let mut resource_branches = None;
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--tree" => options.tree = true,
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
            "--no-stabilization" => options.stabilize = false,
            "--verbose" | "-v" => options.log_interval = Some(10),
            "--time" => options.time_limit = value()?.parse::<f64>().map_err(|e| e.to_string())?,
            "--iterations" => {
                options.max_iterations = value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            "--nodes" => {
                options.node_limit = value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            "--artificial-cost" => {
                options.artificial_cost = value()?.parse::<f64>().map_err(|e| e.to_string())?
            }
            "--bound-mode" => options.bound_mode = value()?,
            "--log-interval" => {
                options.log_interval = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--json" => json = Some(value()?),
            "--option" => {
                let assignment = value()?;
                let (name, val) = assignment
                    .split_once('=')
                    .ok_or("pricing option requires NAME=VALUE")?;
                options
                    .pricing_options
                    .insert(name.into(), val.parse::<f64>().map_err(|e| e.to_string())?);
            }
            _ => return Err(format!("unknown networks option: {arg}\n{HELP}")),
        }
    }
    let node_options = options.node_options().map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let problem = Model::from_json(&text)
        .and_then(Model::problem::<{ baldes::DEFAULT_RESOURCE_DIMENSIONS }>)
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
        let mut node = mp::solve(problem, initial)
            .map_err(|e| e.to_string())?
            .snapshot;
        for request in requests {
            node = request.apply(&node).map_err(|e| e.to_string())?;
        }
        if options.tree {
            let mut configured = options.tree_options().map_err(|e| e.to_string())?;
            configured.time_limit = configured.time_limit.saturating_sub(started.elapsed());
            let output = node
                .solve_assignment_tree(configured)
                .map_err(|e| e.to_string())?;
            SolveResult::tree(&output, started.elapsed().as_secs_f64())
        } else {
            let mut configured = node_options;
            configured.cg.time_limit = configured.cg.time_limit.saturating_sub(started.elapsed());
            let output = node.solve(configured).map_err(|e| e.to_string())?;
            SolveResult::root(&output, started.elapsed().as_secs_f64())
        }
    } else if options.tree {
        let output =
            mp::solve_assignment_tree(problem, options.tree_options().map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        SolveResult::tree(&output, started.elapsed().as_secs_f64())
    } else {
        let output = mp::solve(problem, node_options).map_err(|e| e.to_string())?;
        SolveResult::root(&output, started.elapsed().as_secs_f64())
    };
    let data = result.to_json().map_err(|e| e.to_string())?;
    if json.as_deref() == Some("-") {
        println!("{data}");
    } else {
        if let Some(path) = json {
            std::fs::write(path, data).map_err(|e| e.to_string())?;
        }
        println!(
            "mode={} status={} lp={} incumbent={} bound={} bound_mode={} nodes={} open_nodes={}",
            result.mode,
            result.status,
            result.lp.map_or("unknown".into(), |v| v.to_string()),
            result.incumbent.map_or("unknown".into(), |v| v.to_string()),
            result.bound.map_or("unknown".into(), |v| v.to_string()),
            result.bound_mode,
            result.nodes,
            result.open_nodes
        );
    }
    Ok(())
}
