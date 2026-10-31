//! Standalone AILS-II search and initialization for the command frontend.
use baldes::extensions::{
    ails_ii::{run_ails_ii, AilsIiOptions, AilsIiResult},
    common::{construct_feasible, validate_plan},
};
use baldes::inputs::Instance;

pub fn run_command(
    instance: &Instance,
    get_bks: bool,
    input: Option<&str>,
    output: Option<&str>,
    time_limit: f64,
    seed: i32,
    iterations: Option<usize>,
) -> Result<AilsIiResult, String> {
    use baldes::extensions::common::solution::{format_solution, parse_solution};
    let text = if get_bks {
        Some(crate::common_cli::download_bks(&instance.name)?)
    } else {
        input
            .map(std::fs::read_to_string)
            .transpose()
            .map_err(|e| e.to_string())?
    };
    let initial = text.as_deref().map(parse_solution).transpose()?;
    let checkpoint = |routes: &[Vec<usize>], cost: f64| {
        if let Some(path) = output {
            crate::common_cli::write_checkpoint(
                std::path::Path::new(path),
                &format_solution(routes, cost),
            )?;
        }
        Ok(())
    };
    run(instance, initial, time_limit, seed, iterations, checkpoint)
}

pub fn run(
    instance: &Instance,
    seed: Option<Vec<Vec<usize>>>,
    time_limit: f64,
    random_seed: i32,
    iterations: Option<usize>,
    checkpoint: impl Fn(&[Vec<usize>], f64) -> Result<(), String>,
) -> Result<AilsIiResult, String> {
    let start = std::time::Instant::now();
    if iterations == Some(0) {
        return Err("AILS-II iterations must be positive".into());
    }
    if !time_limit.is_finite() || time_limit < 0.0 {
        return Err("AILS-II time limit must be finite and nonnegative".into());
    }
    let deadline = if time_limit > 0.0 {
        let duration = std::time::Duration::try_from_secs_f64(time_limit)
            .map_err(|_| "AILS-II time limit is too large")?;
        start
            .checked_add(duration)
            .ok_or("AILS-II time limit is too large")?
    } else {
        start + std::time::Duration::from_secs(365 * 24 * 3600)
    };
    let seed = match seed {
        Some(routes) => routes,
        None => construct_feasible(instance, deadline)?,
    };
    let initial_cost = validate_plan(instance, &seed)?;
    checkpoint(&seed, initial_cost)?;
    let remaining = if time_limit > 0.0 {
        (time_limit - start.elapsed().as_secs_f64()).max(0.0)
    } else {
        0.0
    };
    if time_limit > 0.0 && remaining == 0.0 {
        return Ok(AilsIiResult {
            routes: seed,
            best_cost: initial_cost,
            iterations: 0,
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }
    let mut result = run_ails_ii(
        instance,
        &seed,
        &AilsIiOptions {
            time_limit: remaining,
            seed: random_seed,
            iterations: iterations.unwrap_or(AilsIiOptions::default().iterations),
            ..Default::default()
        },
    )?;
    let cost = validate_plan(instance, &result.routes)?;
    checkpoint(&result.routes, cost)?;
    result.elapsed_secs = start.elapsed().as_secs_f64();
    eprintln!(
        "AILS-II: initial={initial_cost:.6} best={cost:.6} iterations={} elapsed={:.2}s",
        result.iterations, result.elapsed_secs
    );
    Ok(result)
}
