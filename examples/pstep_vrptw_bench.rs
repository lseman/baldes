//! Matched Solomon VRPTW pricing rounds; all service windows are enforced.
//! cargo run --release --features cuda --example pstep_vrptw_bench -- instances/Solomon/C201.txt instances/Solomon/C202.txt
#[cfg(feature = "cuda")]
fn main() -> baldes::Result<()> {
    use baldes::extensions::pstep::{PricingMethod, PricingOptions, TimeWindowProblem};
    use baldes::inputs::{Instance, Rounding};
    use std::time::Instant;
    let files: Vec<_> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "provide Solomon instance paths");
    println!("instance,customers,p,method,threads,repetition,seconds,extensions,minimum_rc");
    let max_p = std::env::var("PSTEP_MAX_P")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);
    for file in files {
        let text = std::fs::read_to_string(file).expect("read Solomon instance");
        let mut instance = Instance::parse(&text)?;
        instance.rounding = Rounding::None;
        let problem = TimeWindowProblem::from_instance(&instance)?;
        let n = problem.routing.demands.len();
        let duals: Vec<_> = (0..4 * n)
            .map(|i| ((i * 13 % 47) as f64 - 20.0) * 0.037)
            .collect();
        for p in 2..=max_p {
            let mut reference = None;
            for repetition in 0..=3 {
                let mut configs = vec![(PricingMethod::Enumeration, 16), (PricingMethod::Cuda, 1)];
                if repetition % 2 == 1 {
                    configs.reverse();
                }
                for (method, threads) in configs {
                    let options = PricingOptions {
                        p,
                        method,
                        threads,
                        max_labels_per_job: 5_000_000,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let round = problem.price_steps(&duals, 1.0, &options)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let costs: Vec<_> = round
                        .steps
                        .iter()
                        .map(|s| {
                            s.step.cost
                                - problem
                                    .entries(s)
                                    .iter()
                                    .map(|&(r, a)| a * duals[r])
                                    .sum::<f64>()
                        })
                        .collect();
                    if let Some(old) = &reference {
                        let old: &Vec<f64> = old;
                        assert_eq!(old.len(), costs.len());
                        for (a, b) in old.iter().zip(&costs) {
                            assert!(
                                (a - b).abs() < 1e-7 * (1.0 + a.abs()),
                                "CPU/GPU job mismatch: {a} != {b}"
                            );
                        }
                    } else {
                        reference = Some(costs);
                    }
                    if repetition > 0 {
                        println!(
                            "{},{n},{p},{method:?},{threads},{repetition},{elapsed:.9},{},{:.12}",
                            instance.name,
                            round.enumerated_extensions,
                            round.minimum_reduced_cost.unwrap()
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
}
