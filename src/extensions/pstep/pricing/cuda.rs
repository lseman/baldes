//! Optional exact CUDA small-p pricing. All transfers and synchronization are
//! included in the round deadline. A running kernel cannot be preempted here;
//! an over-budget round is discarded after synchronization.
use super::*;

#[repr(C)]
struct DeviceJob {
    start: i32,
    end: i32,
    maximum: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Output {
    cost: f64,
    load: f64,
    reduced: f64,
    start_time: f64,
    end_time: f64,
    extensions: u64,
    path: [i32; 7],
    length: i32,
    status: i32,
}
extern "C" {
    fn baldes_pstep_cuda(
        n: i32,
        p: i32,
        capacity: f64,
        costs: *const f64,
        demands: *const f64,
        duals: *const f64,
        weight: f64,
        jobs: *const DeviceJob,
        count: i32,
        limit: u64,
        output: *mut Output,
        ready: *const f64,
        due: *const f64,
        service: *const f64,
    ) -> i32;
}

fn launch(
    problem: &Problem,
    timed: Option<&crate::extensions::pstep::TimeWindowProblem>,
    jobs: &[Job],
    p: usize,
    duals: &[f64],
    weight: f64,
    options: &PricingOptions,
) -> Result<Vec<Output>> {
    if p > 6 || problem.demands.len() > i32::MAX as usize - 2 || jobs.len() > i32::MAX as usize {
        return Err(invalid(
            "CUDA p-step requires p <= 6 and 32-bit vertex/job counts",
        ));
    }
    if options.time_limit.is_zero() {
        return Err(incomplete("CUDA p-step time limit"));
    }
    let costs: Vec<_> = problem
        .costs
        .iter()
        .flatten()
        .map(|c| c.unwrap_or(f64::NAN))
        .collect();
    let device_jobs: Vec<_> = jobs
        .iter()
        .map(|j| DeviceJob {
            start: j.start as i32,
            end: j.end as i32,
            maximum: i32::from(j.maximum),
        })
        .collect();
    let ready: Vec<_> = timed.map_or_else(Vec::new, |t| t.windows.iter().map(|w| w.0).collect());
    let due: Vec<_> = timed.map_or_else(Vec::new, |t| t.windows.iter().map(|w| w.1).collect());
    let mut output = vec![Output::default(); jobs.len()];
    // SAFETY: repr(C) matches cuda.cu; validated input dimensions and one output
    // slot per job. The native call synchronizes and retains no host pointers.
    let status = unsafe {
        baldes_pstep_cuda(
            problem.demands.len() as i32,
            p as i32,
            problem.capacity,
            costs.as_ptr(),
            problem.demands.as_ptr(),
            duals.as_ptr(),
            weight,
            device_jobs.as_ptr(),
            jobs.len() as i32,
            options.max_labels_per_job as u64,
            output.as_mut_ptr(),
            if timed.is_some() {
                ready.as_ptr()
            } else {
                std::ptr::null()
            },
            if timed.is_some() {
                due.as_ptr()
            } else {
                std::ptr::null()
            },
            timed.map_or(std::ptr::null(), |t| t.service.as_ptr()),
        )
    };
    if status != 0 {
        return Err(incomplete(format!("CUDA p-step runtime error {status}")));
    }
    if output.iter().any(|o| o.status != 0) {
        return Err(incomplete("CUDA p-step extension limit or cost overflow"));
    }
    Ok(output)
}
fn decode(problem: &Problem, job: &Job, out: &Output) -> Result<Option<Step>> {
    if out.length == 0 {
        return Ok(None);
    }
    if !(2..=7).contains(&out.length) {
        return Err(incomplete("CUDA invalid path length"));
    }
    Ok(Some(Step {
        path: out.path[..out.length as usize]
            .iter()
            .map(|&v| v as usize)
            .collect(),
        cost: out.cost,
        demand: out.load,
        prior_demand: if job.maximum {
            problem.capacity - out.load
        } else {
            0.0
        },
    }))
}
fn empty_round<S>(jobs: usize) -> PricingRound<S> {
    PricingRound {
        steps: vec![],
        minimum_reduced_cost: None,
        jobs,
        workers: 0,
        generated_labels: 0,
        enumerated_extensions: 0,
        method: PricingMethod::Cuda,
    }
}
fn add<S>(round: &mut PricingRound<S>, step: S, rc: f64, reduced: f64) -> Result<()> {
    if !rc.is_finite() || (rc - reduced).abs() > 1e-7 * (1.0 + rc.abs()) {
        return Err(incomplete("CUDA p-step reduced cost replay mismatch"));
    }
    round.minimum_reduced_cost = Some(round.minimum_reduced_cost.map_or(rc, |v| v.min(rc)));
    round.steps.push(step);
    Ok(())
}
pub(super) fn price(
    problem: &Problem,
    jobs: &[Job],
    p: usize,
    duals: &[f64],
    weight: f64,
    options: &PricingOptions,
) -> Result<PricingRound> {
    let started = Instant::now();
    let output = launch(problem, None, jobs, p, duals, weight, options)?;
    let mut round = empty_round(jobs.len());
    for (out, job) in output.iter().zip(jobs) {
        round.enumerated_extensions = round
            .enumerated_extensions
            .checked_add(out.extensions as usize)
            .ok_or_else(|| incomplete("CUDA extension count overflow"))?;
        if let Some(step) = decode(problem, job, out)? {
            let rc = weight * step.cost
                - step_entries(&problem.demands, &step)
                    .iter()
                    .map(|&(r, a)| a * duals[r])
                    .sum::<f64>();
            add(&mut round, step, rc, out.reduced)?;
        }
    }
    if started.elapsed() >= options.time_limit {
        return Err(incomplete("CUDA p-step time limit"));
    }
    Ok(round)
}
pub(super) fn price_timed(
    problem: &crate::extensions::pstep::TimeWindowProblem,
    jobs: &[Job],
    p: usize,
    duals: &[f64],
    weight: f64,
    options: &PricingOptions,
) -> Result<PricingRound<crate::extensions::pstep::TimedStep>> {
    let started = Instant::now();
    let output = launch(
        &problem.routing,
        Some(problem),
        jobs,
        p,
        duals,
        weight,
        options,
    )?;
    let mut round = empty_round(jobs.len());
    for (out, job) in output.iter().zip(jobs) {
        round.enumerated_extensions = round
            .enumerated_extensions
            .checked_add(out.extensions as usize)
            .ok_or_else(|| incomplete("CUDA extension count overflow"))?;
        if let Some(step) = decode(&problem.routing, job, out)? {
            let step = crate::extensions::pstep::TimedStep {
                step,
                start_time: out.start_time,
                end_time: out.end_time,
            };
            let rc = weight * step.step.cost
                - problem
                    .entries(&step)
                    .iter()
                    .map(|&(r, a)| a * duals[r])
                    .sum::<f64>();
            add(&mut round, step, rc, out.reduced)?;
        }
    }
    if started.elapsed() >= options.time_limit {
        return Err(incomplete("CUDA VRPTW p-step time limit"));
    }
    Ok(round)
}
#[cfg(test)]
mod tests {
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
}
