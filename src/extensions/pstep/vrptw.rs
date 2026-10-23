//! Endpoint service-start flow extends the capacity p-step LP to VRPTW.
//! For a path, earliest end time is max(start + transit, barrier); the
//! nondominated scheduling frontier has two endpoints after waiting is removed.
use super::master::{self, Model};
use super::pricing::{endpoint_jobs, Job};
use super::*;
use crate::incomplete;
use std::time::Instant;

/// CVRP data plus service-start windows and service durations, including both
/// depot copies. Travel duration equals routing cost (same units as windows).
#[derive(Clone, Debug)]
pub struct TimeWindowProblem {
    pub routing: Problem,
    pub windows: Vec<(f64, f64)>,
    pub service: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct TimedStep {
    pub step: Step,
    pub start_time: f64,
    pub end_time: f64,
}
#[derive(Clone, Debug)]
pub struct TimeWindowFormulation {
    problem: TimeWindowProblem,
    steps: Vec<TimedStep>,
}

impl TimeWindowProblem {
    pub fn from_instance(instance: &crate::inputs::Instance) -> Result<Self> {
        let mut windows: Vec<_> = instance
            .customers
            .iter()
            .map(|c| (c.ready, c.due))
            .collect();
        let mut service: Vec<_> = instance.customers.iter().map(|c| c.service).collect();
        if windows.is_empty() {
            return Err(invalid("VRPTW requires a depot"));
        }
        windows.push(windows[0]);
        service.push(0.0);
        let problem = Self {
            routing: Problem::from_instance(instance)?,
            windows,
            service,
        };
        problem.validate()?;
        Ok(problem)
    }
    pub fn validate(&self) -> Result<()> {
        self.routing.validate()?;
        let size = self.routing.demands.len() + 2;
        if self.windows.len() != size
            || self.service.len() != size
            || self
                .windows
                .iter()
                .any(|&(a, b)| !a.is_finite() || !b.is_finite() || a < 0.0 || a > b)
            || self.service.iter().any(|&s| !s.is_finite() || s < 0.0)
        {
            return Err(invalid("p-step VRPTW requires finite nonnegative service times and ordered finite windows for all vertices"));
        }
        Ok(())
    }
    /// Rows: degree, endpoint, cumulative-load, service-start flow (n each).
    pub fn entries(&self, column: &TimedStep) -> Vec<(usize, f64)> {
        let n = self.routing.demands.len();
        let mut entries = step_entries(&self.routing.demands, &column.step);
        let first = column.step.path[0];
        let last = *column.step.path.last().unwrap();
        if first != 0 && column.start_time != 0.0 {
            entries.push((3 * n + first - 1, column.start_time));
        }
        if last != n + 1 && column.end_time != 0.0 {
            entries.push((3 * n + last - 1, -column.end_time));
        }
        entries
    }
    pub fn solve_column_generation<M: LpMaster>(
        &self,
        lp: M,
        options: ColumnGenerationOptions,
    ) -> Result<ColumnGenerationResult> {
        master::solve(self, lp, options)
    }

    pub fn price_steps(
        &self,
        duals: &[f64],
        weight: f64,
        options: &PricingOptions,
    ) -> Result<PricingRound<TimedStep>> {
        self.validate()?;
        let n = self.routing.demands.len();
        if duals.len() != 4 * n
            || duals.iter().any(|v| !v.is_finite())
            || !matches!(weight, 0.0 | 1.0)
            || options.p == 0
            || options.threads == 0
            || options.max_labels_per_job == 0
            || options.max_subproblems_per_job == 0
        {
            return Err(invalid(
                "invalid VRPTW p-step pricing options or dual dimensions",
            ));
        }
        let p = options.p.min(n + 1);
        let jobs = endpoint_jobs(&self.routing, p);
        #[cfg(feature = "cuda")]
        if options.method == PricingMethod::Cuda && !jobs.is_empty() {
            return super::pricing::price_timed_cuda(self, &jobs, p, duals, weight, options);
        }
        if options.method == PricingMethod::Bucket {
            return Err(invalid(
                "VRPTW p-step currently supports enumeration or cuda pricing",
            ));
        }
        let started = Instant::now();
        let workers = options.threads.min(jobs.len());
        let mut round = PricingRound {
            steps: vec![],
            minimum_reduced_cost: None,
            jobs: jobs.len(),
            workers,
            generated_labels: 0,
            enumerated_extensions: 0,
            method: PricingMethod::Enumeration,
        };
        if jobs.is_empty() {
            return Ok(round);
        }
        let chunk = (jobs.len() + workers - 1) / workers;
        round.workers = jobs.chunks(chunk).len();
        let batches = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|jobs| {
                    scope.spawn(move || {
                        let mut result = Vec::new();
                        for &job in jobs {
                            let mut search =
                                Search::new(self, job, p, duals, weight, options, started, false);
                            search.run()?;
                            result.push((search.best, search.extensions));
                        }
                        Ok::<_, crate::core::error::OrError>(result)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(incomplete("VRPTW p-step worker panicked")))
                })
                .collect::<Vec<_>>()
        });
        for batch in batches {
            for (step, extensions) in batch? {
                round.enumerated_extensions += extensions;
                if let Some(step) = step {
                    let rc = weight * step.step.cost
                        - self
                            .entries(&step)
                            .iter()
                            .map(|&(r, a)| a * duals[r])
                            .sum::<f64>();
                    if !rc.is_finite() {
                        return Err(invalid("VRPTW reduced cost overflow"));
                    }
                    round.minimum_reduced_cost =
                        Some(round.minimum_reduced_cost.map_or(rc, |old| old.min(rc)));
                    round.steps.push(step);
                }
            }
        }
        if started.elapsed() >= options.time_limit {
            return Err(incomplete("VRPTW p-step time limit"));
        }
        Ok(round)
    }
    pub fn enumerate(&self, options: EnumerationOptions) -> Result<TimeWindowFormulation> {
        self.validate()?;
        if options.p == 0 || options.max_columns == 0 || options.max_extensions == 0 {
            return Err(invalid("VRPTW p-step enumeration limits must be positive"));
        }
        let p = options.p.min(self.routing.demands.len() + 1);
        let pricing = PricingOptions {
            p,
            max_labels_per_job: options.max_extensions,
            ..Default::default()
        };
        let duals = vec![0.0; 4 * self.routing.demands.len()];
        let started = Instant::now();
        let mut steps = Vec::new();
        let mut extensions = 0usize;
        for job in endpoint_jobs(&self.routing, p) {
            let remaining = options.max_extensions.saturating_sub(extensions);
            if remaining == 0 {
                return Err(incomplete("VRPTW p-step full-pool extension limit"));
            }
            let job_options = PricingOptions {
                max_labels_per_job: remaining,
                ..pricing.clone()
            };
            let mut search = Search::new(self, job, p, &duals, 0.0, &job_options, started, true);
            search.pool_limit = options.max_columns.saturating_sub(steps.len());
            search.run()?;
            extensions += search.extensions;
            if extensions > options.max_extensions
                || steps.len() + search.pool.len() > options.max_columns
            {
                return Err(incomplete("VRPTW p-step full-pool enumeration limit"));
            }
            steps.extend(search.pool);
        }
        Ok(TimeWindowFormulation {
            problem: self.clone(),
            steps,
        })
    }
}
impl TimeWindowFormulation {
    pub fn steps(&self) -> &[TimedStep] {
        &self.steps
    }
    pub fn into_master<M: LpMaster>(&self, lp: M) -> Result<M> {
        let mut lp = self.problem.empty_master(lp)?;
        let entries: Vec<_> = self.steps.iter().map(|s| self.problem.entries(s)).collect();
        let columns: Vec<_> = self
            .steps
            .iter()
            .zip(&entries)
            .map(|(s, a)| (s.step.cost, a.as_slice()))
            .collect();
        let added: Vec<_> = lp.add_columns(&columns)?.collect();
        for c in added {
            if lp.upper()[c] < 1.0 {
                lp.set_upper(c, 1.0)?;
            }
        }
        Ok(lp)
    }
}
impl Model for TimeWindowProblem {
    type Column = TimedStep;
    fn validate(&self) -> Result<()> {
        Self::validate(self)
    }
    fn customers(&self) -> usize {
        self.routing.demands.len()
    }
    fn row_count(&self) -> usize {
        4 * self.customers()
    }
    fn empty_master<M: LpMaster>(&self, lp: M) -> Result<M> {
        let mut lp = Formulation {
            customers: self.customers(),
            demands: self.routing.demands.clone(),
            steps: vec![],
        }
        .into_master(lp)?;
        for _ in 0..self.customers() {
            lp.add_row(RowSense::GreaterEqual, 0.0, &[])?;
        }
        Ok(lp)
    }
    fn price(
        &self,
        duals: &[f64],
        weight: f64,
        options: &PricingOptions,
    ) -> Result<PricingRound<TimedStep>> {
        self.price_steps(duals, weight, options)
    }
    fn entries(&self, column: &TimedStep) -> Vec<(usize, f64)> {
        Self::entries(self, column)
    }
    fn cost(&self, column: &TimedStep) -> f64 {
        column.step.cost
    }
    fn identity(&self, column: &TimedStep) -> (Vec<usize>, Vec<u64>) {
        (
            column.step.path.clone(),
            vec![
                column.step.prior_demand.to_bits(),
                column.start_time.to_bits(),
                column.end_time.to_bits(),
            ],
        )
    }
}

struct Search<'a> {
    problem: &'a TimeWindowProblem,
    job: Job,
    p: usize,
    duals: &'a [f64],
    weight: f64,
    options: &'a PricingOptions,
    started: Instant,
    path: Vec<usize>,
    best: Option<TimedStep>,
    best_rc: f64,
    extensions: usize,
    collect: bool,
    pool_limit: usize,
    pool: Vec<TimedStep>,
}
impl<'a> Search<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        problem: &'a TimeWindowProblem,
        job: Job,
        p: usize,
        duals: &'a [f64],
        weight: f64,
        options: &'a PricingOptions,
        started: Instant,
        collect: bool,
    ) -> Self {
        Self {
            problem,
            job,
            p,
            duals,
            weight,
            options,
            started,
            path: vec![job.start],
            best: None,
            best_rc: f64::INFINITY,
            extensions: 0,
            collect,
            pool_limit: usize::MAX,
            pool: vec![],
        }
    }
    fn run(&mut self) -> Result<()> {
        let load = if self.job.start == 0 {
            0.0
        } else {
            self.problem.routing.demands[self.job.start - 1]
        };
        let (a, b) = self.problem.windows[self.job.start];
        let n = self.problem.routing.demands.len();
        let d = |row: usize, v: usize| {
            if v == 0 || v == n + 1 {
                0.0
            } else {
                self.duals[row * n + v - 1]
            }
        };
        let gi = d(2, self.job.start);
        let gj = d(2, self.job.end);
        let constant = -d(1, self.job.start)
            + d(1, self.job.end)
            + if self.job.maximum {
                self.problem.routing.capacity * (gj - gi)
            } else {
                (gj - gi) * load
            };
        self.visit(load, 0.0, constant, 0.0, a, b)
    }
    fn visit(
        &mut self,
        load: f64,
        cost: f64,
        reduced: f64,
        transit: f64,
        barrier: f64,
        latest: f64,
    ) -> Result<()> {
        let n = self.problem.routing.demands.len();
        let tail = *self.path.last().unwrap();
        let hops = self.path.len();
        let heads = if hops == self.p {
            self.job.end..=self.job.end
        } else {
            1..=n + 1
        };
        for head in heads {
            let terminal = head == self.job.end;
            if self.path.contains(&head)
                || (head == n + 1 && !terminal)
                || (tail == 0 && head == n + 1)
                || (terminal && self.job.start != 0 && hops != self.p)
                || (!terminal && hops >= self.p)
            {
                continue;
            }
            self.extensions += 1;
            if self.extensions > self.options.max_labels_per_job {
                return Err(incomplete("VRPTW p-step extension limit"));
            }
            if (self.extensions == 1 || self.extensions % 256 == 0)
                && self.started.elapsed() >= self.options.time_limit
            {
                return Err(incomplete("VRPTW p-step time limit"));
            }
            let Some(arc) = self.problem.routing.costs[tail][head] else {
                continue;
            };
            let next_load = load
                + if head <= n {
                    self.problem.routing.demands[head - 1]
                } else {
                    0.0
                };
            if next_load > self.problem.routing.capacity {
                continue;
            }
            let duration = self.problem.service[tail] + arc;
            let next_transit = transit + duration;
            let next_barrier = (barrier + duration).max(self.problem.windows[head].0);
            let next_latest = latest.min(self.problem.windows[head].1 - next_transit);
            if !next_transit.is_finite() || !next_barrier.is_finite() || !(cost + arc).is_finite() {
                return Err(invalid("VRPTW p-step time or cost overflow"));
            }
            if next_barrier > self.problem.windows[head].1
                || next_latest < self.problem.windows[self.job.start].0
            {
                continue;
            }
            let dual = |row: usize, v: usize| {
                if v == 0 || v == n + 1 {
                    0.0
                } else {
                    self.duals[row * n + v - 1]
                }
            };
            let price = if self.job.maximum {
                dual(2, self.job.start)
            } else {
                dual(2, self.job.end)
            };
            let demand = if head <= n {
                self.problem.routing.demands[head - 1]
            } else {
                0.0
            };
            let next_reduced =
                reduced + self.weight * arc - dual(0, tail) - dual(0, head) + price * demand;
            if !next_reduced.is_finite() {
                return Err(invalid("VRPTW p-step cost overflow"));
            }
            self.path.push(head);
            if terminal {
                let early = (next_barrier - next_transit)
                    .max(self.problem.windows[self.job.start].0)
                    .min(next_latest);
                let times: &[f64] = if self.job.start == 0 {
                    std::slice::from_ref(&early)
                } else if head == n + 1 {
                    std::slice::from_ref(&next_latest)
                } else {
                    &[early, next_latest]
                };
                for &start_time in times {
                    let end_time = (start_time + next_transit).max(next_barrier);
                    let rc = next_reduced - dual(3, self.job.start) * start_time
                        + dual(3, head) * end_time;
                    if !rc.is_finite() {
                        return Err(invalid("VRPTW p-step reduced cost overflow"));
                    }
                    if !self.collect && rc >= self.best_rc {
                        continue;
                    }
                    let step = TimedStep {
                        step: Step {
                            path: self.path.clone(),
                            cost: cost + arc,
                            demand: next_load,
                            prior_demand: if self.job.maximum {
                                self.problem.routing.capacity - next_load
                            } else {
                                0.0
                            },
                        },
                        start_time,
                        end_time,
                    };
                    if self.collect {
                        if self.pool.len() >= self.pool_limit {
                            return Err(incomplete("VRPTW p-step full-pool column limit"));
                        }
                        self.pool.push(step);
                    } else {
                        if rc < self.best_rc {
                            self.best_rc = rc;
                            self.best = Some(step);
                        }
                    }
                }
            } else {
                self.visit(
                    next_load,
                    cost + arc,
                    next_reduced,
                    next_transit,
                    next_barrier,
                    next_latest,
                )?;
            }
            self.path.pop();
        }
        Ok(())
    }
}
