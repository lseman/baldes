//! Fixed-work heuristic benchmark. The last argument is an optional time cap;
//! zero (default) runs the configured search work without a timed cutoff.
use baldes::{
    extensions::{
        ails_ii::{run_ails_ii, AilsIiOptions},
        common::{construct_feasible, validate_plan},
        hgs::driver::{run_hgs, HgsOptions},
    },
    inputs::{Instance, Rounding},
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 6 {
        return Err(
            "usage: heuristic_perf <hgs|ails-ii> INSTANCE CUSTOMERS SEED WORK [SECONDS]".into(),
        );
    }
    let mut instance = Instance::read(&args[2]).map_err(|e| e.to_string())?;
    instance
        .truncate(args[3].parse()?)
        .map_err(|e| e.to_string())?;
    instance.rounding = Rounding::Tenth;
    let seed = args[4].parse()?;
    let work: usize = args[5].parse()?;
    let time_limit: f64 = args.get(6).map_or(Ok(0.), |s| s.parse())?;
    let started = Instant::now();
    let (routes, cost, completed, search_seconds) = match args[1].as_str() {
        "hgs" => {
            let found = run_hgs(
                &instance,
                &HgsOptions {
                    time_limit,
                    nb_iter: work.try_into()?,
                    seed,
                    ..Default::default()
                },
            )?;
            (
                found.routes,
                found.best_cost.ok_or("HGS found no feasible plan")?,
                found.generations as usize,
                found.elapsed_secs,
            )
        }
        "ails-ii" => {
            let initial = construct_feasible(&instance, Instant::now() + Duration::from_secs(120))?;
            let found = run_ails_ii(
                &instance,
                &initial,
                &AilsIiOptions {
                    time_limit,
                    iterations: work,
                    seed,
                    ..Default::default()
                },
            )?;
            (
                found.routes,
                found.best_cost,
                found.iterations,
                found.elapsed_secs,
            )
        }
        _ => return Err("unknown heuristic".into()),
    };
    let seconds = started.elapsed().as_secs_f64();
    let validated = validate_plan(&instance, &routes)?;
    if (cost - validated).abs() > 1e-7 * cost.abs().max(1.) {
        return Err("reported cost disagrees with independent replay".into());
    }
    println!("{{\"algorithm\":\"{}\",\"seed\":{seed},\"work\":{completed},\"cost\":{cost},\"seconds\":{seconds},\"search_seconds\":{search_seconds},\"routes\":{routes:?}}}", args[1]);
    Ok(())
}
