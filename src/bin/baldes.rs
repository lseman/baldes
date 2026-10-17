use baldes::master::root::RootOptions;
use baldes::vrptw::{Instance, Rounding};
const HELP: &str = "Usage: baldes <vrptw|cvrp|hgs> INSTANCE [OPTIONS]

Instance types (Solomon-format or CVRPLIB .vrp input):
  vrptw                 Vehicle routing with time windows
  cvrp                  Capacitated vehicle routing; ignores time windows and service times
  hgs                   Hybrid Genetic Search heuristic

Options:
  --customers N         Keep the first N customers
  --ng N                NG neighborhood size (default: 8)
  --reference-ng        Use reference-compatible customer neighborhoods
  --rounding MODE       Distances: none, nearest, tenth or ceil (default: the
                        file's convention; none for Solomon, nearest for EUC_2D)
  --rank1               Enable rank-one cuts
  --capacity-cuts        Enable capacity cuts (alias: --capacity)
  --no-stabilization    Disable dual stabilization
  --no-enumeration      Disable reduced-cost fixing and enumeration
  --upper-bound X       Initial incumbent
  --iterations N        Master iteration limit
  --time SECONDS        Time limit (BOTH vrptw/cvrp and hgs)
  --json FILE           Write results as JSON (BOTH vrptw/cvrp and hgs)
  --population-size N   HGS minimum population size (default: 25)
  --nb-iter N           HGS iterations without improvement before stopping, or
                        restarting under a time limit (default: 20000)
  --seed N              HGS random seed (default: 0)
  -h, --help            Print help";

fn run() -> std::result::Result<(), String> {
    let mut args = std::env::args().skip(1);
    let kind = args.next().ok_or(HELP)?;
    if kind == "--help" || kind == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if kind != "vrptw" && kind != "cvrp" && kind != "hgs" {
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
    let mut customers = None;
    let mut reference_ng = false;
    let mut json_path = None;
    let mut ng = 8;
    let mut options = RootOptions::default();
    let mut rank1 = false;
    let mut capacity = false;
    let mut rounding = None;
    let mut hgs_pop_size: Option<u32> = None;
    let mut hgs_nb_iter: Option<u32> = None;
    let mut hgs_seed = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--reference-ng" => reference_ng = true,
            "--json" => json_path = Some(args.next().ok_or("missing JSON path")?),
            "--customers" => {
                customers = Some(
                    args.next()
                        .ok_or("missing customer count")?
                        .parse::<usize>()
                        .map_err(|e| e.to_string())?,
                )
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
            "--no-stabilization" => options.cg.stabilize = false,
            "--no-enumeration" => options.enumerate = false,
            "--rounding" => {
                rounding = Some(
                    Rounding::parse(&args.next().ok_or("missing rounding")?).map_err(|e| e.to_string())?,
                )
            }
            "--rank1" => rank1 = true,
            "--capacity" | "--capacity-cuts" => capacity = true,
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
    if kind == "hgs" {
        use baldes::hgs::{HgsOptions, run_hgs};
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
            verbose: true,
        };
        let started = std::time::Instant::now();
        let result = run_hgs(&instance, &opts).map_err(|e| e.to_string())?;
        let elapsed = started.elapsed().as_secs_f64();
        if let Some(path) = json_path {
            fn number(v: f64) -> String {
                if v.is_finite() { v.to_string() } else { "null".into() }
            }
            let routes_json: Vec<_> = result.routes.iter().enumerate().map(|(i, r)| {
                format!("{{\"route\":{},\"customers\":{:?}}}", i, r)
            }).collect();
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
            if result.best_cost.is_some() { "Found" } else { "None" },
            result.best_cost.map_or("null".into(), |c| format!("{:.2}", c)),
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

    if capacity {
        options = options.with_capacity_cuts(baldes::master::root::CapacityCutConfig::new(
            instance.customers.iter().map(|c| c.demand).collect(),
            instance.capacity,
        ));
    }
    let rounding = rounding.unwrap_or(instance.rounding);
    let started = std::time::Instant::now();
    let mut graph = if kind == "cvrp" {
        instance.capacity_graph(ng, rounding)
    } else {
        instance.graph(ng, rounding)
    }
    .map_err(|e| e.to_string())?;
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
            graph.set_ng(v, &neighbours).map_err(|e| e.to_string())?;
        }
    }
    options.cg.multiplicity = n as f64;
    let problem = baldes::master::root::RootProblem::set_partitioning(graph, n as f64)
        .map_err(|e| e.to_string())?;
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
        // "columns" is BaPCod's bcCountCol: generated columns inserted over the
        // run, including any later removed by the clean-ups. "master_columns" is
        // the final master size, artificials included; "route_columns" the route
        // columns still in it.
        let route_columns = output
            .master
            .column_tags
            .iter()
            .filter(|&&t| t >= 0)
            .count();
        let json=format!("{{\"instance\":{:?},\"status\":\"{}\",\"lp\":{},\"cuts\":{},\"cut_rounds\":{},\"lp_before_cuts\":{},\"route_columns\":{},\"fixing_calls\":{},\"enumerated_routes\":{},\"incumbent\":{},\"bound\":{},\"iterations\":{},\"pricing_calls\":{},\"columns\":{},\"master_columns\":{},\"seconds\":{},\"pricing_seconds\":{},\"master_seconds\":{},\"history\":[{}],\"routes\":[{}]}}\n",instance.name,if cut_off {"Cutoff".to_string()} else {format!("{:?}",output.master.status)},number(lp_value),output.cuts,output.master.cut_rounds,number(output.lp_before_cuts),route_columns,output.fixing_calls,output.enumerated_routes,number(output.incumbent),number(output.master.bound),output.master.iterations,output.master.pricing_calls,output.master.inserted_tags.len(),output.master.columns,seconds,output.pricing_seconds,output.solver_seconds-output.pricing_seconds,history.join(","),routes.join(","));
        std::fs::write(path, json).map_err(|e| e.to_string())?;
    }
    println!(
        "instance={} status={:?} lp={:.12} bound={:.12} solves={} pricing_calls={} columns={} master_columns={} cuts={} cut_rounds={}",
        instance.name,
        output.master.status,
        output.master.lp,
        output.master.bound,
        output.master.iterations,
        output.master.pricing_calls,
        output.master.inserted_tags.len(),
        output.master.columns,
        output.cuts,
        output.master.cut_rounds
    );
    for (&tag, &value) in output.master.column_tags.iter().zip(&output.values) {
        if tag >= 0 && value > 1e-6 {
            let route = &output.routes[tag as usize];
            println!(
                "value={:.12} cost={:.12} nodes={:?}",
                value, route.original_cost, route.nodes
            );
        }
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
