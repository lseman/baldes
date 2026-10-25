use super::*;
#[test]
fn cuda_matches_cpu_with_missing_arcs_capacity_and_both_objectives() -> Result<()> {
    let mut problem = Problem {
        demands: vec![1.0, 2.0, 1.0, 3.0, 2.0],
        capacity: 6.0,
        costs: (0..7)
            .map(|i| {
                (0..7)
                    .map(|j| Some(((i * 19 + j * 7) % 23) as f64))
                    .collect()
            })
            .collect(),
    };
    problem.costs[1][3] = None;
    let duals: Vec<_> = (0..15).map(|i| (i as f64 - 6.0) * 0.37).collect();
    for p in 1..=6 {
        for weight in [0.0, 1.0] {
            let mut options = PricingOptions {
                p,
                method: PricingMethod::Enumeration,
                ..Default::default()
            };
            let cpu = problem.price_steps(&duals, weight, &options)?;
            options.method = PricingMethod::Cuda;
            let gpu = problem.price_steps(&duals, weight, &options)?;
            assert_eq!(cpu.jobs, gpu.jobs);
            assert_eq!(cpu.enumerated_extensions, gpu.enumerated_extensions);
            assert_eq!(cpu.steps.len(), gpu.steps.len());
            for (a, b) in cpu.steps.iter().zip(&gpu.steps) {
                assert_eq!(a.path, b.path);
                assert_eq!(a.cost, b.cost);
                assert_eq!(a.demand, b.demand);
                assert_eq!(a.prior_demand, b.prior_demand);
            }
            assert_eq!(cpu.minimum_reduced_cost, gpu.minimum_reduced_cost);
        }
    }
    let options = PricingOptions {
        p: 3,
        method: PricingMethod::Cuda,
        max_labels_per_job: 1,
        ..Default::default()
    };
    assert!(problem.price_steps(&duals, 1.0, &options).is_err());
    Ok(())
}
