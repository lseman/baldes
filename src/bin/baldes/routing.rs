use super::{ce_cli, mp_cli, pstep_cli};
use baldes::inputs::{Instance, Rounding};
use baldes::master::root::RootOptions;
const HELP: &str = "Usage: baldes <solve|hgs|ce> INSTANCE [OPTIONS]

Additional command: baldes multi-pricing MODEL.json [OPTIONS] (see multi-pricing --help; networks is an alias)
Additional command: baldes pstep INSTANCE --p P --threads N (see pstep --help)
Additional command: baldes ce INSTANCE [OPTIONS] (see ce --help)

Commands (Solomon-format or CVRPLIB .vrp input):
  solve                 Solve routing with constraints detected from the input
  vrptw, cvrp           Compatibility aliases; cvrp ignores time windows and service times
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
  --branching MODE      native (default), vehicles, ryan-foster, arc, cluster, resource, hybrid
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

pub(super) fn run() -> std::result::Result<(), String> {
    let mut args = std::env::args().skip(1);
    let kind = args.next().ok_or(HELP)?;
    if kind == "--help" || kind == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if kind == "multi-pricing" || kind == "networks" {
        return mp_cli::run(args.collect());
    }
    if kind == "pstep" {
        return pstep_cli::run(args.collect());
    }
    if kind == "ce" {
        return ce_cli::run(args.collect());
    }
    if !matches!(kind.as_str(), "solve" | "vrptw" | "cvrp" | "hgs") {
        return Err(format!("unknown instance type: {kind}\n\n{HELP}"));
    }
    let file = args.next().ok_or(HELP)?;
    if file == "--help" || file == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if file.starts_with('-') {
        return Err(format!("expected instance path, got {file}\n\n{HELP}"));
    }
    let mut ignore_time_windows = false;
    let mut customers = None;
    let mut reference_ng = false;
    let mut json_path = None;
    let mut ng = 8;
    let mut options = RootOptions::default();
    let mut all_cuts = false;
    let mut rank1 = false;
    let mut fenchel = false;
    let mut capacity = false;
    let mut two_path = false;
    let mut tree = false;
    let mut node_limit = 10000usize;
    let mut tree_options = baldes::master::tree::TreeOptions::default();
    let mut rounding = Some(Rounding::Tenth);
    let mut hgs_pop_size: Option<u32> = None;
    let mut hgs_nb_iter: Option<u32> = None;
    let mut hgs_seed = 0;
    let mut hgs_time = None;
    let mut hgs_columns = baldes::inputs::HgsColumns::Best;
    let mut robust = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--verbose" | "--log" | "-v" => {
                options.cg.log_interval.get_or_insert(10);
            }
            "--log-interval" => {
                let interval = args
                    .next()
                    .ok_or("missing log interval")?
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?;
                if interval == 0 {
                    return Err("log interval must be positive".into());
                }
                options.cg.log_interval = Some(interval);
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
            "--bound-mode" => {
                options.bound_mode = baldes::master::root::BoundMode::parse(
                    &args.next().ok_or("missing bound mode")?,
                )
                .map_err(|e| e.to_string())?
            }
            "--json" => json_path = Some(args.next().ok_or("missing JSON path")?),
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
            "--iterations" => {
                options.cg.max_iterations = args
                    .next()
                    .ok_or("missing iteration limit")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--time" => {
                options.cg.time_limit = std::time::Duration::try_from_secs_f64(
                    args.next()
                        .ok_or("missing time limit")?
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?,
                )
                .map_err(|e| e.to_string())?
            }
            "--clp-scaling" => {
                options.clp_options.scaling = args
                    .next()
                    .ok_or("missing Clp scaling mode")?
                    .parse::<baldes::master::root::ClpScaling>()
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
                    "clp" => baldes::master::root::MasterSolver::Clp,
                    "ipm" => baldes::master::root::MasterSolver::Ipm,
                    "highs" => {
                        let solver = baldes::master::root::MasterSolver::Highs;
                        solver.ensure_available().map_err(|e| e.to_string())?;
                        solver
                    }
                    "gurobi" => {
                        let solver = baldes::master::root::MasterSolver::Gurobi;
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
            "--no-stabilization" => options.cg.stabilize = false,
            "--dynamic-ng" => {
                options.relaxation_improvement_priority = args
                    .next()
                    .ok_or("missing dynamic NG priority")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--option" => {
                let setting = args.next().ok_or("missing NAME=VALUE")?;
                let (name, value) = setting.split_once('=').ok_or("option must be NAME=VALUE")?;
                let value = value
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                options.pricing_options.push((name.to_string(), value));
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
            "--tree" => tree = true,
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
                    "native" => BranchingOptions::default(),
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
            "--nodes" => {
                node_limit = args
                    .next()
                    .ok_or("missing node limit")?
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
        use baldes::extensions::hgs::{run_hgs, HgsOptions};
        if let Some(rounding) = rounding {
            instance.rounding = rounding;
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
        let result = run_hgs(&instance, &opts).map_err(|e| e.to_string())?;
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
            std::fs::write(path, json).map_err(|e| e.to_string())?;
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
    let capacity_only = kind == "cvrp"
        || (kind == "solve" && (ignore_time_windows || !instance.has_time_windows()));
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
            let mut kpath =
                baldes::master::root::KPathCutConfig::new(config.demands.clone(), config.capacity);
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
            .native_graph_with_memory(&vec![0.; problem.nodes.len()], 0., &memory)
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
    let problem = baldes::master::root::RootProblem::set_partitioning(graph, fleet)
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
            std::fs::write(path, result.to_json(&instance.name)).map_err(|e| e.to_string())?;
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
    let output = baldes::master::root::solve(problem, options).map_err(|e| e.to_string())?;
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
        let json=format!("{{\"instance\":{:?},\"status\":\"{}\",\"lp\":{},\"cuts\":{},\"cut_rounds\":{},\"lp_before_cuts\":{},\"route_columns\":{},\"fixing_calls\":{},\"enumerated_routes\":{},\"incumbent\":{},\"bound\":{},\"bound_mode\":\"{}\",\"numerical_bound\":{},\"certified_bound\":{},\"iterations\":{},\"simplex_iterations\":{},\"ipm_iterations\":{},\"pricing_calls\":{},\"columns\":{},\"master_columns\":{},\"seconds\":{},\"pricing_seconds\":{},\"master_seconds\":{},\"feasibility\":[{}],\"history\":[{}],\"routes\":[{}]}}\n",instance.name,if cut_off {"Cutoff".to_string()} else {format!("{:?}",output.master.status)},number(lp_value),output.cuts,output.master.cut_rounds,number(output.lp_before_cuts),route_columns,output.fixing_calls,output.enumerated_routes,number(output.incumbent),number(output.master.bound),output.master.bound_mode.as_str(),number(output.master.numerical_bound),number(output.master.certified_bound),output.master.iterations,output.simplex_iterations,output.ipm_iterations,output.master.pricing_calls,output.master.inserted_tags.len(),output.master.columns,seconds,output.pricing_seconds,output.solver_seconds-output.pricing_seconds,feasibility.join(","),history.join(","),routes.join(","));
        std::fs::write(path, json).map_err(|e| e.to_string())?;
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
