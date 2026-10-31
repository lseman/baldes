use super::*;

pub(super) const ROUTING_HELP: &str = "Usage: baldes <solve|hgs|ails-ii|ce> INSTANCE [OPTIONS]

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
  --robust GQ,GT,AQ,AT  Robust VRPTW (Munari et al., 2019): at most GQ demands and
                        GT travel times deviate by trunc(AQ*q) and 0.1*trunc(10*AT*t)
                        (solve/vrptw with time windows and tenth rounding)
  --rounding MODE       Distances: none, nearest, tenth or ceil (default: tenth)
  --cuts                Enable all applicable cuts (2-path requires time windows)
  --rank1               Enable rank-one cuts
  --fenchel-cuts        Enable Fenchel cuts on the gcd-scaled integer-demand row
  --capacity-cuts        Enable capacity cuts (alias: --capacity)
  --two-path-cuts        Enable capacity and Kohl 2-path cuts using time windows
  --tree                Run branch-and-price
  --nodes N             Tree node limit (default: 10000)
  --branching MODE      default (route count, then arc flow), arc, edge,
                        resource, cluster (route count, MST cluster boundaries/pairs,
                        then arc flow), ryan-foster (route count, customer-pair
                        Ryan–Foster, then arc flow) or hybrid (phased: route count,
                        resource, clusters, assignment, Ryan–Foster, arc flow)
  --no-strong-branching  Skip strong trials on mapped arc/edge variables
  --diving              Strong limited-discrepancy diving every 10 nodes (off by default)
  --no-diving           Disable diving (default)
  --restricted-mip      Restricted-master integer heuristic at the root (HiGHS)
  --mip-nodes N         Node budget of that heuristic (enables it)
  --no-mip-heuristic    Disable it (default)
  --tree-threads N      Evaluate up to N open nodes concurrently (default: 1)
  --absolute-gap X      Close nodes within this absolute gap (default: 1e-6)
  --relative-gap X      Close nodes within this relative gap (default: 1e-9)
  --integer-objective   Round node bounds up (requires integer costs)
  --no-branch-history   Do not rank branching candidates by history
  --master MODE         clp (default), highs, gurobi, or ipm (disables stabilization)
  --master-threads N    HiGHS/Gurobi worker budget; 0 = all available CPUs (default)
  --clp-scaling MODE    off (default), equilibrium, auto, geometric, or ruiz
  --clp-work-areas MODE retain (default) or discard after each LP solve
  --bound-mode MODE     numerical (default) or certified (independent bound validation)
  --hgs-time SECONDS    Run HGS before root/tree solve for this search budget
  --hgs-columns MODE    best (default), all, or N cheapest feasible HGS routes
  --no-stabilization    Disable dual stabilization
  --stabilization-max-tree-depth N  Activate only below depth N (default: 10000)
  --enumerated-mip      Solve complete enumerated node masters with HiGHS (--tree; highs feature)
  --no-enumeration      Disable reduced-cost fixing and enumeration
  --upper-bound X       Initial incumbent
  --dynamic-ng P        Strengthen NG memories from master cycles at separator
                        priority P (capacity cuts 3, rank-one 1); see
                        BALDESdynamicNGmode and BALDESmaxNGneighbourhoodSize
  --option NAME=VALUE   Set a supported BALDES pricing option (repeatable)
  -v, --verbose, --log  Show solver progress (every 10 CG iterations)
  --log-interval N      Show progress every N CG iterations (enables logging)
  --iterations N        Master or AILS-II iteration limit
  --time SECONDS        Time limit (solve, hgs and ails-ii)
  --json FILE           Write results as JSON (solve and hgs)
  --population-size N   HGS minimum population size (default: 25)
  --nb-iter N           HGS iterations without improvement before stopping, or
                        restarting under a time limit (default: 20000)
  --seed N              Heuristic random seed (default: 0)
  -h, --help            Print help";

struct RoutingOptions {
    ignore_time_windows: bool,
    customers: Option<usize>,
    ng: usize,
    options: model::Options,
    enumerate: bool,
    all_cuts: bool,
    fenchel: bool,
    capacity: bool,
    two_path: bool,
    branching: String,
    strong_branching: bool,
    diving: bool,
    tree_controls: TreeControls,
    enumerated_mip: bool,
    rounding: Option<Rounding>,
    hgs_extension: crate::hgs_cli::ExtensionOptions,
    ails_iterations: Option<usize>,
    hgs_pop_size: Option<u32>,
    hgs_nb_iter: Option<u32>,
    hgs_seed: i32,
    hgs_time: Option<std::time::Duration>,
    hgs_columns: baldes::inputs::HgsColumns,
    robust: Option<baldes::extensions::robust::Uncertainty>,
    tree: bool,
    node_limit: usize,
    json_path: Option<String>,
}

fn parse_options(kind: &str, args: Vec<String>) -> Result<Option<RoutingOptions>, String> {
    let mut args = args.into_iter();
    let mut common = CommonOptions::default();
    let mut ignore_time_windows = false;
    let mut customers = None;
    let mut ng = 8;
    let mut options = model::Options::default();
    // Master column clean-up: threshold 10000, keep ratio 0.66.
    options.cg.column_cleanup_threshold = 10000.;
    options.cg.column_cleanup_ratio = 0.66;
    let mut enumerate = true;
    let mut all_cuts = false;
    let mut rank1 = false;
    let mut fenchel = false;
    let mut capacity = false;
    let mut two_path = false;
    let mut branching = "default".to_string();
    let mut strong_branching = true;
    let mut diving = false;
    let mut tree_controls = TreeControls::default();
    let mut enumerated_mip = false;
    let mut rounding = Some(Rounding::Tenth);
    let mut rounding_specified = false;
    let mut hgs_extension = crate::hgs_cli::ExtensionOptions::default();
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
                println!("\n{}", crate::hgs_cli::HELP);
                return Ok(None);
            }
            "--ignore-time-windows" => {
                if kind != "solve" {
                    return Err("--ignore-time-windows requires solve".into());
                }
                ignore_time_windows = true;
            }
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
                    .parse::<baldes::master::clp::ClpScaling>()
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
                    "clp" => baldes::master::clp::MasterSolver::Clp,
                    "ipm" => baldes::master::clp::MasterSolver::Ipm,
                    "highs" => {
                        let solver = baldes::master::clp::MasterSolver::Highs;
                        solver.ensure_available().map_err(|e| e.to_string())?;
                        solver
                    }
                    "gurobi" => {
                        let solver = baldes::master::clp::MasterSolver::Gurobi;
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
                if kind == "ails-ii" {
                    return Err("--hgs-columns does not apply to ails-ii".into());
                }
                hgs_columns = baldes::inputs::HgsColumns::parse(
                    &args.next().ok_or("missing HGS columns mode")?,
                )
                .map_err(|e| e.to_string())?
            }
            "--dynamic-ng" => {
                options.separation.relaxation_improvement_priority = args
                    .next()
                    .ok_or("missing dynamic NG priority")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--no-enumeration" => enumerate = false,
            "--enumerated-mip" => enumerated_mip = true,
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
            "--branching" => {
                branching = args.next().ok_or("missing branching mode")?;
                if ![
                    "default",
                    "arc",
                    "edge",
                    "resource",
                    "hybrid",
                    "cluster",
                    "ryan-foster",
                ]
                .contains(&branching.as_str())
                {
                    return Err(format!("unknown branching mode: {branching}"));
                }
            }
            "--no-strong-branching" => strong_branching = false,
            "--diving" => diving = true,
            "--no-diving" => diving = false,
            "--restricted-mip" => tree_controls.restricted_mip = true,
            "--no-mip-heuristic" => tree_controls.restricted_mip = false,
            "--mip-nodes" => {
                tree_controls.restricted_mip = true;
                tree_controls.mip_nodes = Some(
                    args.next()
                        .ok_or("missing MIP node budget")?
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?,
                );
            }
            "--tree-threads" => {
                tree_controls.threads = args
                    .next()
                    .ok_or("missing tree thread count")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                if tree_controls.threads == 0 {
                    return Err("tree threads must be positive".into());
                }
            }
            "--absolute-gap" | "--relative-gap" => {
                let value: f64 = args
                    .next()
                    .ok_or("missing gap value")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                if !value.is_finite() || value < 0. {
                    return Err("gaps must be finite and nonnegative".into());
                }
                if arg == "--absolute-gap" {
                    tree_controls.absolute_gap = value;
                } else {
                    tree_controls.relative_gap = value;
                }
            }
            "--integer-objective" => tree_controls.integer_objective = true,
            "--branch-history" => tree_controls.history = true,
            "--no-branch-history" => tree_controls.history = false,
            "--upper-bound" => {
                options.separation.upper_bound = Some(
                    args.next()
                        .ok_or("missing upper bound")?
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?,
                )
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
    let ails_iterations = common.iterations;
    common.apply_routing(&mut options);
    hgs_extension.validate()?;
    if kind == "ails-ii" && (hgs_extension.intensify || hgs_extension.dri) {
        return Err("ails-ii conflicts with --intensify and --dri".into());
    }
    if kind == "ails-ii" && (hgs_pop_size.is_some() || hgs_nb_iter.is_some() || hgs_time.is_some())
    {
        return Err(
            "HGS population, iteration and warm-start options do not apply to ails-ii".into(),
        );
    }
    if hgs_extension.active() && kind != "hgs" && kind != "ails-ii" {
        return Err("BKS seeding and intensification options require hgs".into());
    }
    if (hgs_extension.get_bks
        || hgs_extension.initial_solution.is_some()
        || hgs_extension.intensify
        || kind == "ails-ii"
        || hgs_extension.dri)
        && !rounding_specified
    {
        rounding = None;
    }
    let tree = common.tree;
    let node_limit = common.nodes.unwrap_or(10000);
    let json_path = common.json;
    options.separation.rank_one.enabled = rank1;
    Ok(Some(RoutingOptions {
        ignore_time_windows,
        customers,
        ng,
        options,
        enumerate,
        all_cuts,
        fenchel,
        capacity,
        two_path,
        branching,
        strong_branching,
        diving,
        tree_controls,
        enumerated_mip,
        rounding,
        hgs_extension,
        ails_iterations,
        hgs_pop_size,
        hgs_nb_iter,
        hgs_seed,
        hgs_time,
        hgs_columns,
        robust,
        tree,
        node_limit,
        json_path,
    }))
}

pub(super) fn solve_routing(kind: String, file: String, args: Vec<String>) -> Result<(), String> {
    let Some(RoutingOptions {
        ignore_time_windows,
        customers,
        ng,
        mut options,
        enumerate,
        all_cuts,
        fenchel,
        capacity,
        mut two_path,
        branching,
        strong_branching,
        diving,
        tree_controls,
        enumerated_mip,
        rounding,
        hgs_extension,
        ails_iterations,
        hgs_pop_size,
        hgs_nb_iter,
        hgs_seed,
        hgs_time,
        hgs_columns,
        robust,
        tree,
        node_limit,
        json_path,
    }) = parse_options(&kind, args)?
    else {
        return Ok(());
    };
    let mut instance = Instance::read(file).map_err(|e| e.to_string())?;
    if let Some(n) = customers {
        instance.truncate(n).map_err(|e| e.to_string())?;
    }
    if enumerated_mip && !tree {
        return Err("--enumerated-mip requires --tree".into());
    }
    if kind == "ails-ii" {
        if let Some(rounding) = rounding {
            instance.rounding = rounding;
        } else if hgs_extension.get_bks && instance.has_time_windows() {
            instance.rounding = Rounding::Tenth;
        }
        let time = if options.cg.time_limit.is_zero() {
            60.0
        } else {
            options.cg.time_limit.as_secs_f64()
        };
        let result = crate::ails_cli::run_command(
            &instance,
            hgs_extension.get_bks,
            hgs_extension.initial_solution.as_deref(),
            hgs_extension.output.as_deref(),
            time,
            hgs_seed,
            ails_iterations,
        )?;
        if let Some(path) = json_path {
            let json = serde_json::json!({"instance": instance.name, "status": "Found", "cost": result.best_cost, "iterations": result.iterations, "seconds": result.elapsed_secs, "routes": result.routes});
            if write_json(&path, &json.to_string())? {
                return Ok(());
            }
        }
        println!(
            "instance={} status=Found cost={:.2} iterations={} seconds={}",
            instance.name, result.best_cost, result.iterations, result.elapsed_secs
        );
        for (i, route) in result.routes.iter().enumerate() {
            if !route.is_empty() {
                println!("route={i} customers={route:?}");
            }
        }
        return Ok(());
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
    let separation = &mut options.separation;
    if fenchel {
        let mut config = instance.fenchel_cut_config().map_err(|e| e.to_string())?;
        if all_cuts {
            config.parameters.use_rounding = true;
            config.parameters.max_one_k = 1;
        }
        separation.knapsack = Some(config);
    }
    if capacity {
        let config = instance.capacity_cut_config(rounding, two_path);
        if all_cuts {
            let mut kpath =
                model::separation::KPathCutConfig::new(config.demands.clone(), config.capacity);
            kpath.time_windows = config.time_windows.clone();
            separation.kpath = Some(kpath);
        }
        separation.capacity = Some(config);
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
    // A root LP may use one route per customer; the tree enforces the fleet.
    let fleet = if tree { instance.vehicles } else { n };
    let mut node = options;
    node.enumeration = enumerate.then(Default::default);
    let mut initial_solution = None;
    if let Some(time_limit) = hgs_time {
        let seed = instance
            .seed_hgs(
                &graph,
                rounding,
                &baldes::inputs::HgsWarmStart {
                    time_limit,
                    columns: hgs_columns,
                    seed: hgs_seed,
                },
                node.cg.log_interval.is_some(),
                &[],
            )
            .map_err(|e| e.to_string())?;
        eprintln!(
            "HGS seed: columns={} rejected={} incumbent={:?} seconds={:.3}",
            seed.summary.inserted_columns,
            seed.summary.rejected_columns,
            seed.incumbent,
            seed.summary.elapsed_secs
        );
        if let Some(upper) = seed.incumbent {
            let bound = node.separation.upper_bound.get_or_insert(upper);
            *bound = bound.min(upper);
            if let (true, Some(solution)) = (tree, &seed.solution) {
                let routes = solution
                    .iter()
                    .map(|route| model::FixedRoute {
                        route: model::NetworkRoute {
                            network: 0,
                            route: route.clone(),
                        },
                        quantity: 1.,
                    })
                    .collect();
                initial_solution = Some((upper, routes));
            }
        }
        node.seed_routes = seed
            .routes
            .into_iter()
            .map(|route| model::NetworkRoute { network: 0, route })
            .collect();
    }
    let problem =
        model::Problem::set_partitioning(graph, fleet as f64).map_err(|e| e.to_string())?;
    let result = if tree {
        use baldes::master::tree::{ModelTreeOptions, NetworkBranchingPolicy};
        let mapped_variables = match branching.as_str() {
            "edge" => problem.edge_flow_variables(1.),
            // Phased collection already proposes every fractional arc flow.
            "hybrid" => vec![],
            _ => problem.arc_flow_variables(1.),
        };
        let configured = ModelTreeOptions {
            branching: match branching.as_str() {
                "resource" => NetworkBranchingPolicy::ResourceFirst,
                "hybrid" => NetworkBranchingPolicy::Phased,
                _ => NetworkBranchingPolicy::MultiplicityFirst,
            },
            mapped_variables,
            clusters: if matches!(branching.as_str(), "cluster" | "hybrid") {
                vec![baldes::master::model::NetworkClusterBranching::new(0)]
            } else {
                vec![]
            },
            ryan_foster: matches!(branching.as_str(), "ryan-foster" | "hybrid")
                .then(|| baldes::master::model::RyanFosterBranching::new(2.)),
            mapped_strong_branching: strong_branching
                .then(baldes::master::tree::StrongBranchingOptions::default),
            restricted_master_mip: tree_controls.restricted_mip.then(|| {
                let mut mip = baldes::master::tree::RestrictedMasterMip::default();
                if let Some(nodes) = tree_controls.mip_nodes {
                    mip.node_limit = nodes;
                }
                mip
            }),
            absolute_gap: tree_controls.absolute_gap,
            relative_gap: tree_controls.relative_gap,
            integer_objective: tree_controls.integer_objective,
            use_branching_history: tree_controls.history,
            threads: tree_controls.threads,
            diving: diving.then(|| baldes::master::tree::DivingOptions {
                depth_limit: usize::MAX,
                frequency: 10,
                lds_depth: 30,
                max_discrepancy: 2,
                strong_candidates: 3,
                stop_after_first_solution: true,
                node_limit: 100,
                time_limit: 2.,
                ..Default::default()
            }),
            node_limit,
            time_limit: node.cg.time_limit,
            enumerated_mip: enumerated_mip.then(Default::default),
            initial_solution,
            node,
            ..ModelTreeOptions::default()
        };
        let output = baldes::master::tree::solve(problem, configured).map_err(|e| e.to_string())?;
        SolveResult::tree(&output, started.elapsed().as_secs_f64())
    } else {
        let output = model::solve(problem, node).map_err(|e| e.to_string())?;
        SolveResult::root(&output, started.elapsed().as_secs_f64())
    };
    let data = result.to_json().map_err(|e| e.to_string())?;
    if let Some(path) = json_path {
        if write_json(&path, &data)? {
            return Ok(());
        }
    }
    println!(
        "instance={} mode={} status={} lp={} incumbent={} bound={} bound_mode={} nodes={} iterations={} pricing_calls={} seconds={:.3}",
        instance.name,
        result.mode,
        result.status,
        result.lp.map_or("unknown".into(), |v| v.to_string()),
        result.incumbent.map_or("unknown".into(), |v| v.to_string()),
        result.bound.map_or("unknown".into(), |v| v.to_string()),
        result.bound_mode,
        result.nodes,
        result.iterations,
        result.pricing_calls,
        result.seconds,
    );
    for route in &result.routes {
        if route.value > 1e-6 {
            println!(
                "value={:.2} cost={:.2} nodes={:?}",
                route.value, route.cost, route.nodes
            );
        }
    }
    Ok(())
}

/// Tree heuristic, gap and parallelism flags.
struct TreeControls {
    restricted_mip: bool,
    mip_nodes: Option<usize>,
    threads: usize,
    absolute_gap: f64,
    relative_gap: f64,
    integer_objective: bool,
    history: bool,
}
impl Default for TreeControls {
    fn default() -> Self {
        Self {
            restricted_mip: false,
            mip_nodes: None,
            threads: 1,
            absolute_gap: 1e-6,
            relative_gap: 1e-9,
            integer_objective: false,
            history: true,
        }
    }
}
