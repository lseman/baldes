//! One command frontend for routing and collections of pricing networks.
use super::{ce_cli, pstep_cli};
use baldes::inputs::{Instance, Rounding};
use baldes::master::model::{
    self,
    format::{Model, ResourceGroupRestriction, SolveOptions, SolveResult},
};
use baldes::master::routing::RootOptions;

pub(super) fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(ROUTING_HELP)?;
    match command.as_str() {
        "-h" | "--help" => {
            println!(
                "{ROUTING_HELP}\n\n{}\n\n{NETWORK_HELP}",
                super::hgs_cli::HELP
            );
            Ok(())
        }
        "pstep" => pstep_cli::run(args.collect()),
        "ce" => ce_cli::run(args.collect()),
        "solve" | "hgs" => {
            let file = args.next().ok_or(ROUTING_HELP)?;
            if file == "--help" || file == "-h" {
                println!(
                    "{ROUTING_HELP}\n\n{}\n\n{NETWORK_HELP}",
                    super::hgs_cli::HELP
                );
                return Ok(());
            }
            if file.starts_with('-') {
                return Err(format!("expected model or instance path, got {file}"));
            }
            if command == "solve"
                && std::path::Path::new(&file)
                    .extension()
                    .is_some_and(|e| e == "json")
            {
                solve_networks(file, args.collect())
            } else {
                solve_routing(command, file, args.collect())
            }
        }
        _ => Err(format!("unknown command: {command}\n{ROUTING_HELP}")),
    }
}

/// The same result sink is used for routing and network models.
fn write_json(path: &str, data: &str) -> Result<bool, String> {
    if path == "-" {
        println!("{}", data.trim_end());
        Ok(true)
    } else {
        std::fs::write(path, data).map_err(|e| e.to_string())?;
        Ok(false)
    }
}

#[derive(Default)]
struct CommonOptions {
    time: Option<std::time::Duration>,
    iterations: Option<usize>,
    nodes: Option<usize>,
    bound_mode: Option<baldes::master::cg::BoundMode>,
    stabilize: Option<bool>,
    depth: Option<i64>,
    log_interval: Option<usize>,
    pricing: Vec<(String, f64)>,
    tree: bool,
    json: Option<String>,
}
impl CommonOptions {
    fn parse(&mut self, arg: &str, args: &mut std::vec::IntoIter<String>) -> Result<bool, String> {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing {}", arg.trim_start_matches('-').replace('-', " ")))
        };
        match arg {
            "--tree" => self.tree = true,
            "--json" => self.json = Some(value()?),
            "--no-stabilization" => self.stabilize = Some(false),
            "--verbose" | "--log" | "-v" => {
                self.log_interval.get_or_insert(10);
            }
            "--time" => {
                self.time = Some(
                    std::time::Duration::try_from_secs_f64(
                        value()?.parse::<f64>().map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| e.to_string())?,
                )
            }
            "--iterations" => {
                self.iterations = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--nodes" => self.nodes = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?),
            "--stabilization-max-tree-depth" => {
                self.depth = Some(value()?.parse::<i64>().map_err(|e| e.to_string())?)
            }
            "--bound-mode" => {
                self.bound_mode = Some(
                    baldes::master::cg::BoundMode::parse(&value()?).map_err(|e| e.to_string())?,
                )
            }
            "--log-interval" => {
                let n = value()?.parse::<usize>().map_err(|e| e.to_string())?;
                if n == 0 {
                    return Err("log interval must be positive".into());
                }
                self.log_interval = Some(n);
            }
            "--option" => {
                let assignment = value()?;
                let (name, v) = assignment
                    .split_once('=')
                    .ok_or("option must be NAME=VALUE")?;
                self.pricing
                    .push((name.into(), v.parse::<f64>().map_err(|e| e.to_string())?));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
    fn apply_routing(&self, options: &mut RootOptions) {
        if let Some(v) = self.time {
            options.cg.time_limit = v;
        }
        if let Some(v) = self.iterations {
            options.cg.max_iterations = v;
        }
        if let Some(v) = self.bound_mode {
            options.bound_mode = v;
        }
        if let Some(v) = self.stabilize {
            options.cg.stabilize = v;
        }
        if let Some(v) = self.depth {
            options.cg.stabilization_max_tree_depth = v;
        }
        if let Some(v) = self.log_interval {
            options.cg.log_interval = Some(v);
        }
        options.pricing_options.extend(self.pricing.iter().cloned());
    }
    fn apply_networks(&self, options: &mut SolveOptions) {
        options.tree = self.tree;
        if let Some(v) = self.nodes {
            options.node_limit = v;
        }
        if let Some(v) = self.time {
            options.time_limit = v.as_secs_f64();
        }
        if let Some(v) = self.iterations {
            options.max_iterations = v;
        }
        if let Some(v) = self.bound_mode {
            options.bound_mode = v.as_str().into();
        }
        if let Some(v) = self.stabilize {
            options.stabilize = v;
        }
        if let Some(v) = self.depth {
            options.stabilization_max_tree_depth = v;
        }
        if let Some(v) = self.log_interval {
            options.log_interval = Some(v);
        }
        options.pricing_options.extend(self.pricing.iter().cloned());
    }
}

const ROUTING_HELP: &str = "Usage: baldes <solve|hgs|ce> INSTANCE [OPTIONS]

JSON pricing models use the same solve command, with one network or an array.
Additional command: baldes pstep INSTANCE --p P --threads N (see pstep --help)
Additional command: baldes ce INSTANCE [OPTIONS] (see ce --help)

Commands (Solomon-format or CVRPLIB .vrp input):
  solve                 Solve routing with constraints detected from the input
  hgs                   Hybrid Genetic Search heuristic
  ce                    Column elimination CVRP/VRPTW route LP

Options:
  --ignore-time-windows Ignore time windows and service times (solve only)
  --customers N         Keep the first N customers
  --ng N                NG neighborhood size (default: 8)
  --steps S|TIME,CAPACITY Initial bucket widths in graph resource units; one value
                        applies to both resources (0 = automatic; default: 0)
  --reference-ng        Use reference-compatible customer neighborhoods
  --robust GQ,GT,AQ,AT  Robust VRPTW (Munari et al., 2019): at most GQ demands and
                        GT travel times deviate by trunc(AQ*q) and 0.1*trunc(10*AT*t)
                        (solve/vrptw with time windows and tenth rounding)
  --rounding MODE       Distances: none, nearest, tenth or ceil (default: tenth)
  --cuts                Enable all applicable cuts (2-path requires time windows)
  --rank1               Enable rank-one cuts
  --fenchel-cuts        Enable Fenchel cuts on the gcd-scaled integer-demand row
  --capacity-cuts        Enable capacity cuts (alias: --capacity)
  --two-path-cuts        Enable capacity and Kohl 2-path cuts using time windows
  --tree                Run branch-and-price with a restricted-master MIP heuristic
  --tree-threads N      Concurrent tree node evaluations (default 1)
  --nodes N             Tree node limit (default: 10000)
  --branching MODE      default, vehicles, ryan-foster, arc, cluster, resource, hybrid, edge
  --branch-history      Enable branch history (default; observed native rules)
  --no-branch-history   Disable history-based candidate ranking and recording
  --no-strong-branching  Skip strong trials; history may still rank candidates
  --no-diving           Disable strong diving in the tree
  --no-mip-heuristic    Disable restricted-master integer search
  --mip-nodes N         Restricted-master integer-search node budget per call
  --master MODE         clp (default), highs, gurobi, or ipm (disables stabilization)
  --master-threads N    HiGHS/Gurobi worker budget; 0 = all available CPUs (default)
  --clp-scaling MODE    equilibrium (default), auto, geometric, or off
  --clp-work-areas MODE retain (default) or discard after each LP solve
  --bound-mode MODE     numerical (default) or certified (independent bound validation)
  --hgs-time SECONDS    Run HGS before root/tree solve for this search budget
  --hgs-columns MODE    best (default), all, or N cheapest feasible HGS routes
  --no-stabilization    Disable dual stabilization
  --stabilization-max-tree-depth N  Activate only below depth N (default: 10000)
  --enumerated-mip      Solve complete enumerated node masters with HiGHS (--tree; highs feature)
  --deluxing            Filter enumerated MIP columns (enables --enumerated-mip; requires --tree)
  --no-enumeration      Disable reduced-cost fixing and enumeration
  --upper-bound X       Initial incumbent
  --dynamic-ng P        Strengthen NG memories from master cycles at separator
                        priority P (capacity cuts 3, rank-one 1); see
                        BALDESdynamicNGmode and BALDESmaxNGneighbourhoodSize
  --option NAME=VALUE   Set a supported BALDES pricing option (repeatable)
  -v, --verbose, --log  Show solver progress (every 10 CG iterations)
  --log-interval N      Show progress every N CG iterations (enables logging)
  --iterations N        Master iteration limit
  --time SECONDS        Time limit (solve and hgs)
  --json FILE           Write results as JSON (solve and hgs)
  --population-size N   HGS minimum population size (default: 25)
  --nb-iter N           HGS iterations without improvement before stopping, or
                        restarting under a time limit (default: 20000)
  --seed N              HGS random seed (default: 0)
  -h, --help            Print help";

fn solve_routing(kind: String, file: String, args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let mut common = CommonOptions::default();
    let mut ignore_time_windows = false;
    let mut customers = None;
    let mut reference_ng = false;
    let mut ng = 8;
    let mut options = RootOptions::default();
    let mut all_cuts = false;
    let mut rank1 = false;
    let mut fenchel = false;
    let mut capacity = false;
    let mut two_path = false;
    let mut tree_options = baldes::master::tree::TreeOptions::default();
    let mut rounding = Some(Rounding::Tenth);
    let mut rounding_specified = false;
    let mut hgs_extension = super::hgs_cli::ExtensionOptions::default();
    let mut hgs_pop_size: Option<u32> = None;
    let mut hgs_nb_iter: Option<u32> = None;
    let mut hgs_seed = 0;
    let mut hgs_time = None;
    let mut hgs_columns = baldes::inputs::HgsColumns::Best;
    let mut robust = None;
    while let Some(arg) = args.next() {
        if common.parse(&arg, &mut args)? {
            continue;
        }
        if hgs_extension.parse(&arg, &mut args)? {
            continue;
        }
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{ROUTING_HELP}");
                println!("\n{}", super::hgs_cli::HELP);
                return Ok(());
            }
            "--ignore-time-windows" => {
                if kind != "solve" {
                    return Err("--ignore-time-windows requires solve".into());
                }
                ignore_time_windows = true;
            }
            "--reference-ng" => reference_ng = true,
            "--robust" => {
                let text = args.next().ok_or("missing robust uncertainty")?;
                robust = Some(
                    baldes::extensions::robust::Uncertainty::parse(&text)
                        .map_err(|e| e.to_string())?,
                );
            }
            "--customers" => {
                customers = Some(
                    args.next()
                        .ok_or("missing customer count")?
                        .parse::<usize>()
                        .map_err(|e| e.to_string())?,
                )
            }
            "--steps" => {
                let value = args.next().ok_or("missing bucket steps")?;
                let steps = value
                    .split(',')
                    .map(|v| v.parse::<f64>().map_err(|_| "invalid bucket steps"))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                if !(1..=2).contains(&steps.len())
                    || steps.iter().any(|s| !s.is_finite() || *s < 0.)
                {
                    return Err("bucket steps require one or two finite nonnegative widths".into());
                }
                options.bucket_steps = if steps.len() == 1 {
                    vec![steps[0]; 2]
                } else {
                    steps
                };
            }
            "--ng" => {
                ng = args
                    .next()
                    .ok_or("missing NG size")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--clp-scaling" => {
                options.clp_options.scaling = args
                    .next()
                    .ok_or("missing Clp scaling mode")?
                    .parse::<baldes::master::routing::ClpScaling>()
                    .map_err(|e| e.to_string())?
            }
            "--clp-work-areas" => {
                options.clp_options.keep_work_areas =
                    match args.next().ok_or("missing Clp work-area mode")?.as_str() {
                        "retain" => true,
                        "discard" => false,
                        mode => return Err(format!("unknown Clp work-area mode: {mode}")),
                    }
            }
            "--master-threads" => {
                options.master_threads = args
                    .next()
                    .ok_or("missing master thread count")?
                    .parse::<usize>()
                    .map_err(|_| "master threads must be a nonnegative integer")?;
                if options.master_threads > i32::MAX as usize {
                    return Err("master thread count exceeds native integer range".into());
                }
            }
            "--master" => {
                options.master_solver = match args.next().ok_or("missing master mode")?.as_str() {
                    "clp" => baldes::master::routing::MasterSolver::Clp,
                    "ipm" => baldes::master::routing::MasterSolver::Ipm,
                    "highs" => {
                        let solver = baldes::master::routing::MasterSolver::Highs;
                        solver.ensure_available().map_err(|e| e.to_string())?;
                        solver
                    }
                    "gurobi" => {
                        let solver = baldes::master::routing::MasterSolver::Gurobi;
                        solver.ensure_available().map_err(|e| e.to_string())?;
                        solver
                    }
                    mode => return Err(format!("unknown master mode: {mode}")),
                }
            }
            "--hgs-time" => {
                let seconds = args
                    .next()
                    .ok_or("missing HGS time")?
                    .parse::<f64>()
                    .map_err(|e| e.to_string())?;
                let duration =
                    std::time::Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())?;
                if duration.is_zero() {
                    return Err("HGS time must be positive".into());
                }
                hgs_time = Some(duration);
            }
            "--hgs-columns" => {
                hgs_columns = baldes::inputs::HgsColumns::parse(
                    &args.next().ok_or("missing HGS columns mode")?,
                )
                .map_err(|e| e.to_string())?
            }
            "--dynamic-ng" => {
                options.relaxation_improvement_priority = args
                    .next()
                    .ok_or("missing dynamic NG priority")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--no-enumeration" => options.enumerate = false,
            "--enumerated-mip" => tree_options.enumerated_mip = Some(Default::default()),
            "--deluxing" => {
                tree_options.deluxing = Some(Default::default());
                tree_options
                    .enumerated_mip
                    .get_or_insert_with(Default::default);
            }
            "--rounding" => {
                rounding_specified = true;
                rounding = Some(
                    Rounding::parse(&args.next().ok_or("missing rounding")?)
                        .map_err(|e| e.to_string())?,
                )
            }
            "--cuts" => {
                all_cuts = true;
                rank1 = true;
                fenchel = true;
                capacity = true;
            }
            "--rank1" => rank1 = true,
            "--fenchel-cuts" => fenchel = true,
            "--capacity" | "--capacity-cuts" => capacity = true,
            "--two-path-cuts" => {
                capacity = true;
                two_path = true;
            }
            "--tree-threads" => {
                tree_options.threads = args
                    .next()
                    .ok_or("missing tree thread count")?
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?;
                if tree_options.threads == 0 {
                    return Err("tree threads must be positive".into());
                }
            }
            "--branching" => {
                use baldes::master::branching::BranchingOptions;
                tree_options.branching = match args.next().ok_or("missing branching mode")?.as_str()
                {
                    "default" => BranchingOptions::default(),
                    "arc" => BranchingOptions::arc_flow_only(),
                    "cluster" => BranchingOptions {
                        clusters: Some(Default::default()),
                        ..BranchingOptions::arc_flow_only()
                    },
                    "resource" => BranchingOptions {
                        accumulated_resource: true,
                        ..BranchingOptions::arc_flow_only()
                    },
                    "hybrid" => BranchingOptions {
                        clusters: Some(Default::default()),
                        accumulated_resource: true,
                        ..BranchingOptions::default()
                    },
                    "ryan-foster" => BranchingOptions {
                        ryan_foster: true,
                        ..BranchingOptions::arc_flow_only()
                    },
                    "vehicles" => BranchingOptions {
                        number_of_vehicles: true,
                        ..BranchingOptions::arc_flow_only()
                    },
                    "edge" => BranchingOptions {
                        edge_flow: true,
                        ..BranchingOptions::arc_flow_only()
                    },
                    mode => return Err(format!("unknown branching mode: {mode}")),
                };
            }
            "--branch-history" => tree_options.branch_history = Some(Default::default()),
            "--no-branch-history" => tree_options.branch_history = None,
            "--no-strong-branching" => tree_options.strong_branching = None,
            "--no-diving" => tree_options.diving = None,
            "--no-mip-heuristic" => tree_options.mip_node_limit = 0,
            "--mip-nodes" => {
                tree_options.mip_node_limit = args
                    .next()
                    .ok_or("missing MIP node budget")?
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?
            }
            "--upper-bound" => {
                options.incumbent = args
                    .next()
                    .ok_or("missing upper bound")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--population-size" => {
                hgs_pop_size = Some(
                    args.next()
                        .ok_or("missing population size")?
                        .parse::<u32>()
                        .map_err(|e| e.to_string())?,
                )
            }
            "--nb-iter" => {
                hgs_nb_iter = Some(
                    args.next()
                        .ok_or("missing nb-iter value")?
                        .parse::<u32>()
                        .map_err(|e| e.to_string())?,
                )
            }
            "--seed" => {
                hgs_seed = args
                    .next()
                    .ok_or("missing seed")?
                    .parse::<i32>()
                    .map_err(|e| e.to_string())?
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    common.apply_routing(&mut options);
    hgs_extension.validate()?;
    if hgs_extension.active() && kind != "hgs" {
        return Err("BKS seeding and intensification options require hgs".into());
    }
    if (hgs_extension.get_bks
        || hgs_extension.initial_solution.is_some()
        || hgs_extension.intensify)
        && !rounding_specified
    {
        rounding = None;
    }
    let tree = common.tree;
    let node_limit = common.nodes.unwrap_or(10000);
    let json_path = common.json;
    if rank1 {
        options = options.with_rank_one_cuts();
    }
    let mut instance = Instance::read(file).map_err(|e| e.to_string())?;
    if let Some(n) = customers {
        instance.truncate(n).map_err(|e| e.to_string())?;
    }
    // HGS heuristic path
    if tree_options.deluxing.is_some() {
        if !tree {
            return Err("--deluxing requires --tree".into());
        }
        if !options.enumerate {
            return Err("--deluxing conflicts with --no-enumeration".into());
        }
        if kind == "hgs" {
            return Err("--deluxing requires a routing solve, not hgs".into());
        }
    }
    if tree_options.threads > 1 && !tree {
        return Err("--tree-threads requires --tree".into());
    }
    if tree_options.enumerated_mip.is_some() && !tree {
        return Err("--enumerated-mip requires --tree".into());
    }
    if kind == "hgs" {
        if all_cuts {
            return Err("--cuts requires a routing solve, not hgs".into());
        }
        if fenchel {
            return Err("--fenchel-cuts requires solve".into());
        }
        use baldes::extensions::hgs::HgsOptions;
        if let Some(rounding) = rounding {
            instance.rounding = rounding;
        } else if hgs_extension.get_bks && instance.has_time_windows() {
            // Current CVRPLib Solomon/Homberger BKSs use truncated tenths.
            instance.rounding = Rounding::Tenth;
        }
        let time_limit = if options.cg.time_limit.as_secs_f64() > 0.0 {
            options.cg.time_limit.as_secs_f64()
        } else {
            60.0
        };
        let opts = HgsOptions {
            time_limit,
            nb_iter: hgs_nb_iter.unwrap_or(20000),
            population_size: hgs_pop_size.unwrap_or(25),
            seed: hgs_seed,
            verbose: options.cg.log_interval.is_some(),
        };
        let started = std::time::Instant::now();
        let result = hgs_extension.run(&instance, &opts)?;
        let elapsed = started.elapsed().as_secs_f64();
        if let Some(path) = json_path {
            fn number(v: f64) -> String {
                if v.is_finite() {
                    v.to_string()
                } else {
                    "null".into()
                }
            }
            let routes_json: Vec<_> = result
                .routes
                .iter()
                .enumerate()
                .map(|(i, r)| format!("{{\"route\":{},\"customers\":{:?}}}", i, r))
                .collect();
            let json = format!("{{\"instance\":{:?},\"status\":\"{}\",\"cost\":{},\"generations\":{},\"seconds\":{},\"routes\":[{}]}}\n",
                instance.name,
                if result.best_cost.is_some() { "Found" } else { "None" },
                result.best_cost.map(number).unwrap_or("null".into()),
                result.generations,
                elapsed,
                routes_json.join(","),
            );
            if write_json(&path, &json)? {
                return Ok(());
            }
        }
        println!(
            "instance={} status={} cost={} generations={} seconds={}",
            instance.name,
            if result.best_cost.is_some() {
                "Found"
            } else {
                "None"
            },
            result
                .best_cost
                .map_or("null".into(), |c| format!("{:.2}", c)),
            result.generations,
            elapsed
        );
        for (i, route) in result.routes.iter().enumerate() {
            if !route.is_empty() {
                println!("route={} customers={:?}", i, route);
            }
        }
        return Ok(());
    }

    let rounding = rounding.unwrap_or(instance.rounding);
    let capacity_only = kind == "solve" && (ignore_time_windows || !instance.has_time_windows());
    two_path |= all_cuts && !capacity_only;
    if two_path && capacity_only {
        return Err("Kohl 2-path cuts require vrptw time windows".into());
    }
    if fenchel {
        let mut config = instance.fenchel_cut_config().map_err(|e| e.to_string())?;
        if all_cuts {
            config.parameters.use_rounding = true;
            config.parameters.max_one_k = 1;
        }
        options = options.with_knapsack_cuts(config);
    }
    if capacity {
        let config = instance.capacity_cut_config(rounding, two_path);
        if all_cuts {
            let mut kpath = baldes::master::routing::KPathCutConfig::new(
                config.demands.clone(),
                config.capacity,
            );
            kpath.time_windows = config.time_windows.clone();
            options = options.with_kpath_cuts(kpath);
        }
        options = options.with_capacity_cuts(config);
    }
    let started = std::time::Instant::now();
    let mut graph = if capacity_only {
        instance.capacity_graph(ng, rounding)
    } else {
        instance.graph(ng, rounding)
    }
    .map_err(|e| e.to_string())?;
    if let Some(uncertainty) = robust {
        if capacity_only {
            return Err("--robust requires time windows".into());
        }
        // HGS routes are only nominally feasible.
        if hgs_time.is_some() {
            return Err("--robust conflicts with --hgs-time".into());
        }
        let problem =
            baldes::extensions::robust::Problem::from_instance(&instance, rounding, uncertainty)
                .map_err(|e| e.to_string())?;
        // Keep the nominal graph's NG memories so Γ = 0 matches it exactly.
        let memory: Vec<_> = graph.vertices.iter().map(|v| v.memory.clone()).collect();
        graph = problem
            .pricing_graph_with_memory(&vec![0.; problem.nodes.len()], 0., &memory)
            .map_err(|e| e.to_string())?;
    }
    let n = instance.customers.len() - 1;
    if reference_ng {
        let sink = n + 1;
        for v in 0..=sink {
            let mut neighbours: Vec<_> = if v == 0 || v == sink {
                vec![v]
            } else {
                let mut ids: Vec<_> = (1..=n).collect();
                ids.sort_by(|&a, &b| {
                    instance
                        .distance(v, a, rounding)
                        .total_cmp(&instance.distance(v, b, rounding))
                        .then(a.cmp(&b))
                });
                ids.truncate(ng);
                ids
            };
            neighbours.sort_unstable();
            let elements: Vec<_> = neighbours
                .iter()
                .flat_map(|&vertex| graph.vertices[vertex].elements.iter().copied())
                .collect();
            graph.set_ng(v, &elements).map_err(|e| e.to_string())?;
        }
    }
    options.cg.multiplicity = n as f64;
    let fleet = if tree {
        instance.vehicles as f64
    } else {
        n as f64
    };
    let problem = baldes::master::routing::RoutingContext::set_partitioning(graph, fleet)
        .map_err(|e| e.to_string())?;
    if let Some(time_limit) = hgs_time {
        let summary = instance
            .seed_hgs(
                &problem,
                rounding,
                &mut options,
                &baldes::inputs::HgsWarmStart {
                    time_limit,
                    columns: hgs_columns,
                    seed: hgs_seed,
                },
            )
            .map_err(|e| e.to_string())?;
        eprintln!(
            "HGS seed: columns={} rejected={} incumbent={:?} seconds={:.3}",
            summary.inserted_columns,
            summary.rejected_columns,
            summary.incumbent,
            summary.elapsed_secs
        );
    }
    if tree {
        tree_options.node_limit = node_limit;
        tree_options.time_limit = options.cg.time_limit;
        tree_options.root = options;
        let result =
            baldes::master::tree::solve(problem, tree_options).map_err(|e| e.to_string())?;
        if let Some(path) = json_path {
            if write_json(&path, &result.to_json(&instance.name))? {
                return Ok(());
            }
        }
        println!(
            "instance={} status={:?} incumbent={:.2} bound={:.2} nodes={} pricing_calls={} bound_mode={}",
            instance.name,
            result.status,
            result.incumbent,
            result.bound,
            result.nodes,
            result.pricing_calls,
            result.bound_mode.as_str()
        );
        for route in &result.routes {
            println!("cost={} nodes={:?}", route.original_cost, route.nodes);
        }
        return Ok(());
    }
    let output = baldes::master::routing::solve(problem, options).map_err(|e| e.to_string())?;
    let seconds = started.elapsed().as_secs_f64();
    if let Some(path) = json_path {
        fn number(v: f64) -> String {
            if v.is_finite() {
                v.to_string()
            } else {
                "null".into()
            }
        }
        let feasibility: Vec<_> = output
            .master
            .feasibility
            .iter()
            .map(|round| {
                serde_json::json!({
                    "status": round.status.as_str(),
                    "iterations": round.iterations,
                    "pricing_calls": round.pricing_calls,
                    "simplex_iterations": round.simplex_iterations,
                    "ipm_iterations": round.ipm_iterations,
                    "seconds": round.seconds,
                    "artificial_activity": round.artificial_activity,
                    "lower_bound": round.lower_bound,
                    "certified_infeasible": round.certified_infeasible,
                })
                .to_string()
            })
            .collect();
        let history:Vec<_>=output.master.history.iter().map(|h|format!("{{\"iteration\":{},\"lp\":{},\"min_reduced_cost\":{},\"new_columns\":{},\"columns\":{},\"phase\":{},\"pricing_calls\":{},\"alpha\":{},\"misprices\":{}}}",h.iteration,number(h.lp),number(h.minimum),h.new_columns,h.columns,h.phase,h.pricing_calls,number(h.alpha),h.misprices)).collect();
        let routes: Vec<_> = output
            .master
            .column_tags
            .iter()
            .zip(&output.values)
            .filter_map(|(&tag, &value)| {
                if tag >= 0 && value > 1e-9 {
                    let r = &output.routes[tag as usize];
                    Some(format!(
                        "{{\"value\":{},\"nodes\":{:?},\"arcs\":{:?},\"cost\":{}}}",
                        number(value),
                        r.nodes,
                        r.arcs,
                        number(r.original_cost)
                    ))
                } else {
                    None
                }
            })
            .collect();
        // A bound strictly above the incumbent prunes the root: the C++ driver
        // reports the LP sentinel 1e12 and status cutoff.
        let bound = output.master.bound;
        let cut_off = output.master.pruned_by_bound
            && output.incumbent.is_finite()
            && bound > output.incumbent + (bound.abs().max(output.incumbent.abs()) * 1e-10 + 1e-6);
        let lp_value = if cut_off { 1e12 } else { output.master.lp };
        // "columns" counts generated columns inserted over the
        // run, including any later removed by the clean-ups. "master_columns" is
        // the final master size, artificials included; "route_columns" the route
        // columns still in it.
        let route_columns = output
            .master
            .column_tags
            .iter()
            .filter(|&&t| t >= 0)
            .count();
        let json=format!("{{\"instance\":{:?},\"status\":\"{}\",\"lp\":{},\"cuts\":{},\"cut_rounds\":{},\"lp_before_cuts\":{},\"route_columns\":{},\"fixing_calls\":{},\"enumerated_routes\":{},\"incumbent\":{},\"bound\":{},\"bound_mode\":\"{}\",\"numerical_bound\":{},\"certified_bound\":{},\"iterations\":{},\"simplex_iterations\":{},\"ipm_iterations\":{},\"pricing_calls\":{},\"oracle_calls\":{},\"columns\":{},\"master_columns\":{},\"seconds\":{},\"pricing_seconds\":{},\"master_seconds\":{},\"feasibility\":[{}],\"history\":[{}],\"routes\":[{}]}}\n",instance.name,if cut_off {"Cutoff".to_string()} else {format!("{:?}",output.master.status)},number(lp_value),output.cuts,output.master.cut_rounds,number(output.lp_before_cuts),route_columns,output.fixing_calls,output.enumerated_routes,number(output.incumbent),number(output.master.bound),output.master.bound_mode.as_str(),number(output.master.numerical_bound),number(output.master.certified_bound),output.master.iterations,output.simplex_iterations,output.ipm_iterations,output.master.pricing_calls,output.oracle_calls,output.master.inserted_tags.len(),output.master.columns,seconds,output.pricing_seconds,output.solver_seconds-output.pricing_seconds,feasibility.join(","),history.join(","),routes.join(","));
        if write_json(&path, &json)? {
            return Ok(());
        }
    }
    println!(
        "instance={} status={:?} lp={:.2} bound={:.2} solves={} pricing_calls={} columns={} master_columns={} cuts={} cut_rounds={} bound_mode={}",
        instance.name,
        output.master.status,
        output.master.lp,
        output.master.bound,
        output.master.iterations,
        output.master.pricing_calls,
        output.master.inserted_tags.len(),
        output.master.columns,
        output.cuts,
        output.master.cut_rounds,
        output.master.bound_mode.as_str()
    );
    for (&tag, &value) in output.master.column_tags.iter().zip(&output.values) {
        if tag >= 0 && value > 1e-6 {
            let route = &output.routes[tag as usize];
            println!(
                "value={:.2} cost={:.2} nodes={:?}",
                value, route.original_cost, route.nodes
            );
        }
    }
    Ok(())
}

const NETWORK_HELP: &str = "Usage: baldes solve MODEL.json [OPTIONS]

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

fn solve_networks(file: String, args: Vec<String>) -> Result<(), String> {
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
                return Ok(());
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
