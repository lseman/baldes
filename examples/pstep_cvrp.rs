//! cargo run --features clp --example pstep_cvrp -- instances/cvrplib/P-n16-k8.vrp 2 4 --enumerate
#[cfg(feature = "clp")]
fn main() -> Result<(), String> {
    use baldes::extensions::pstep::{
        ColumnGenerationOptions, EnumerationOptions, PricingOptions, Problem,
    };
    use baldes::master::{clp::ClpMaster, LpMaster};
    use std::time::{Duration, Instant};

    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 || args.len() > 4 || (args.len() == 4 && args[3] != "--enumerate") {
        return Err("usage: pstep_cvrp INSTANCE.vrp P THREADS [--enumerate]".into());
    }
    let instance = baldes::inputs::Instance::parse(
        &std::fs::read_to_string(&args[0]).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let problem = Problem::from_instance(&instance).map_err(|e| e.to_string())?;
    let p = args[1]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let threads = args[2]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let result = problem
        .solve_column_generation(
            ClpMaster::new().map_err(|e| e.to_string())?,
            ColumnGenerationOptions {
                pricing: PricingOptions {
                    p,
                    threads,
                    time_limit: Duration::from_secs(300),
                    ..Default::default()
                },
                time_limit: Duration::from_secs(300),
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
    let mut enumerated = None;
    let mut enumeration_seconds = None;
    if args.len() == 4 {
        let tick = Instant::now();
        let pool = problem
            .enumerate(EnumerationOptions {
                p,
                max_columns: 2_000_000,
                max_extensions: 20_000_000,
            })
            .map_err(|e| e.to_string())?;
        let mut lp = pool
            .into_master(ClpMaster::new().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if lp.solve() {
            enumerated = Some(lp.objective());
        }
        enumeration_seconds = Some(tick.elapsed().as_secs_f64());
        if let (Some(a), Some(b)) = (result.lp_bound, enumerated) {
            if (a - b).abs() > 1e-5 {
                return Err(format!("CG/enumeration bound mismatch: {a} vs {b}").into());
            }
        }
    }
    let number = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.6}"));
    println!("instance,customers,p,requested_threads,workers,status,lp_bound,iterations,columns,pricing_jobs,labels,elapsed_s,pricing_s,lp_s,enumerated_bound,enumeration_s");
    println!(
        "{},{},{},{},{},{:?},{},{},{},{},{},{:.6},{:.6},{:.6},{},{}",
        instance.name,
        problem.demands.len(),
        p,
        threads,
        result.pricing_workers,
        result.status,
        number(result.lp_bound),
        result.iterations,
        result.columns,
        result.pricing_jobs,
        result.generated_labels,
        result.elapsed_seconds,
        result.pricing_seconds,
        result.lp_seconds,
        number(enumerated),
        number(enumeration_seconds)
    );
    Ok(())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("Run with --features clp");
}
