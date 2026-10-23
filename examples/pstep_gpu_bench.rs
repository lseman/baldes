//! Matched exact pricing rounds, including allocation, transfers and CPU replay.
//! cargo run --release --features cuda --example pstep_gpu_bench
#[cfg(feature = "cuda")]
fn main() -> baldes::Result<()> {
    use baldes::extensions::pstep::{PricingMethod, PricingOptions, Problem};
    use std::time::Instant;
    println!("customers,p,method,threads,repetition,seconds,extensions,minimum_rc");
    for n in [16, 32, 64] {
        let problem = Problem {
            demands: (0..n).map(|i| 1.0 + (i % 5) as f64).collect(),
            capacity: 30.0,
            costs: (0..n + 2)
                .map(|i| {
                    (0..n + 2)
                        .map(|j| {
                            let (x, y) = ((i * 17 % 101) as f64, (i * 31 % 97) as f64);
                            let (a, b) = ((j * 17 % 101) as f64, (j * 31 % 97) as f64);
                            Some((x - a).hypot(y - b))
                        })
                        .collect()
                })
                .collect(),
        };
        let duals: Vec<_> = (0..3 * n)
            .map(|i| ((i * 13 % 47) as f64 - 20.0) * 0.37)
            .collect();
        for p in [2, 3, 4] {
            let mut reference = None;
            // Warm each configuration once, then alternate measured order.
            for repetition in 0..=5 {
                let mut configs = vec![
                    (PricingMethod::Enumeration, 1),
                    (PricingMethod::Enumeration, 8),
                    (PricingMethod::Enumeration, 16),
                    (PricingMethod::Cuda, 1),
                ];
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
                    let signature: Vec<_> = round
                        .steps
                        .iter()
                        .map(|s| (s.path.clone(), s.cost, s.demand, s.prior_demand))
                        .collect();
                    if let Some(old) = &reference {
                        assert_eq!(old, &signature);
                    } else {
                        reference = Some(signature);
                    }
                    if repetition > 0 {
                        println!(
                            "{n},{p},{method:?},{threads},{repetition},{elapsed:.9},{},{:.12}",
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
