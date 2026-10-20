//! Independent endpoint/extreme jobs, solved by the core bucket graph engine.
use super::{step_entries, Problem, Step};
use crate::graph::{Graph, ResourceKind};
use crate::labeling::engine::{EngineControls, LabelingEngine};
use crate::labeling::Controls;
use crate::pricing::phases::Phase;
use crate::{incomplete, invalid, Result};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct PricingOptions {
    pub p: usize,
    /// CPU workers; each owns its graph, labels and result buffer.
    pub threads: usize,
    pub max_labels_per_job: usize,
    /// Bucket solves per job, including exact cycle-exclusion branches.
    pub max_subproblems_per_job: usize,
    pub time_limit: Duration,
}

impl Default for PricingOptions {
    fn default() -> Self {
        Self {
            p: 2,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(8)),
            max_labels_per_job: 1_000_000,
            max_subproblems_per_job: 10_000,
            time_limit: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PricingRound {
    /// Best path from every feasible endpoint/extreme job, in deterministic
    /// job order. Positive reduced cost steps may be included.
    pub steps: Vec<Step>,
    pub minimum_reduced_cost: Option<f64>,
    pub jobs: usize,
    pub workers: usize,
    pub generated_labels: usize,
}

#[derive(Clone, Copy)]
struct Job {
    start: usize,
    end: usize,
    maximum: bool,
}

impl Problem {
    /// Exact elementary p-step pricing at the supplied master row duals.
    /// Objective weight is zero for phase I and one for travel-cost pricing.
    /// Limits/errors abort the round; an incomplete round cannot certify CG.
    pub fn price_steps(
        &self,
        duals: &[f64],
        objective_weight: f64,
        options: &PricingOptions,
    ) -> Result<PricingRound> {
        self.validate()?;
        let n = self.demands.len();
        if duals.len() != 3 * n
            || duals.iter().any(|d| !d.is_finite())
            || !matches!(objective_weight, 0.0 | 1.0)
            || options.p == 0
            || options.threads == 0
            || options.max_labels_per_job == 0
            || options.max_subproblems_per_job == 0
        {
            return Err(invalid("invalid p-step pricing options or dual dimensions"));
        }
        // An elementary path has at most n+1 arcs. Clamping also keeps the
        // resource windows small when the caller supplies a very large p.
        let p = options.p.min(n + 1);
        let mut jobs = Vec::new();
        for start in 0..=n {
            for end in 1..=n + 1 {
                if start == end
                    || (start == 0 && end == n + 1 && p == 1)
                    || (start != 0 && p > n)
                    || (start != 0 && end <= n && p >= n)
                    || (start != 0
                        && end <= n
                        && self.demands[start - 1] + self.demands[end - 1] > self.capacity)
                {
                    continue;
                }
                if start == 0 {
                    jobs.push(Job {
                        start,
                        end,
                        maximum: false,
                    });
                } else {
                    if end <= n {
                        jobs.push(Job {
                            start,
                            end,
                            maximum: false,
                        });
                    }
                    jobs.push(Job {
                        start,
                        end,
                        maximum: true,
                    });
                }
            }
        }
        let started = Instant::now();
        let workers = options.threads.min(jobs.len());
        if jobs.is_empty() {
            return Ok(PricingRound {
                steps: vec![],
                minimum_reduced_cost: None,
                jobs: 0,
                workers: 0,
                generated_labels: 0,
            });
        }
        let chunk_size = (jobs.len() + workers - 1) / workers;
        let workers = jobs.chunks(chunk_size).len();
        // Scoped threads borrow immutable costs and duals. No LP backend or
        // mutable labeling storage is shared. Join every worker before return.
        let batches = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut output = Vec::new();
                        for &job in chunk {
                            let remaining = options.time_limit.saturating_sub(started.elapsed());
                            if remaining.is_zero() {
                                return Err(incomplete("p-step pricing time limit"));
                            }
                            output.push(self.price_job(
                                job,
                                p,
                                duals,
                                objective_weight,
                                options,
                                remaining,
                            )?);
                        }
                        Ok(output)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(incomplete("p-step pricing worker panicked")))
                })
                .collect::<Vec<Result<Vec<_>>>>()
        });
        if started.elapsed() >= options.time_limit {
            return Err(incomplete("p-step pricing time limit"));
        }
        let mut result = PricingRound {
            steps: vec![],
            minimum_reduced_cost: None,
            jobs: jobs.len(),
            workers,
            generated_labels: 0,
        };
        for batch in batches {
            for (step, labels) in batch? {
                result.generated_labels += labels;
                if let Some(step) = step {
                    let rc = objective_weight * step.cost
                        - step_entries(&self.demands, &step)
                            .iter()
                            .map(|&(i, a)| a * duals[i])
                            .sum::<f64>();
                    result.minimum_reduced_cost =
                        Some(result.minimum_reduced_cost.map_or(rc, |v| v.min(rc)));
                    result.steps.push(step);
                }
            }
        }
        Ok(result)
    }

    fn price_job(
        &self,
        job: Job,
        p: usize,
        duals: &[f64],
        weight: f64,
        options: &PricingOptions,
        remaining: Duration,
    ) -> Result<(Option<Step>, usize)> {
        let n = self.demands.len();
        let demand = |v: usize| {
            if v == 0 || v == n + 1 {
                0.0
            } else {
                self.demands[v - 1]
            }
        };
        let dual = |offset: usize, v: usize| {
            if v == 0 || v == n + 1 {
                0.0
            } else {
                duals[offset * n + v - 1]
            }
        };
        let (i, j) = (job.start, job.end);
        let (gi, gj) = (dual(2, i), dual(2, j));
        // c - degree*alpha - endpoint*beta - load*gamma.
        // With fixed endpoints, choosing either prior-load extreme leaves
        // an affine function of path demand. Put its constant on first arcs.
        let load_price = if job.maximum { gi } else { gj };
        let constant = -dual(1, i)
            + dual(1, j)
            + if job.maximum {
                self.capacity * (gj - gi)
            } else {
                (gj - gi) * demand(i)
            };
        let mut g = Graph::<2>::new(n + 2, 2)?;
        g.source = i;
        g.sink = j;
        // p<=3 paths are already elementary (distinct terminals, no loops,
        // no arcs into source or out of sink). Longer steps use exact cycle
        // exclusion below instead of the core's 15-element local NG mask.
        g.elementary = false;
        // Hop equality prevents waiting from fabricating extra arcs. Load
        // windows all have lower bound zero: waiting cannot improve a path,
        // so a cheaper label using less capacity safely dominates. Arc costs
        // use physical demands, independent of any resource waiting.
        g.resources = vec![ResourceKind::Equality, ResourceKind::Disposable];
        g.initial[0] = Some(0.0);
        g.initial[1] = Some(demand(i));
        for v in 0..n + 2 {
            g.set_bounds(v, 0, 0.0, p as f64)?;
            g.set_bounds(v, 1, 0.0, self.capacity)?;
        }
        g.set_bounds(i, 0, 0.0, 0.0)?;
        g.set_bounds(j, 0, if i == 0 { 1.0 } else { p as f64 }, p as f64)?;
        for tail in 0..=n {
            if tail == j || (tail == 0 && i != 0) {
                continue;
            }
            for head in 1..=n + 1 {
                if head == tail
                    || head == i
                    || (head == n + 1 && j != n + 1)
                    || (tail == 0 && head == n + 1)
                    || (p == 1 && (tail != i || head != j))
                    || (p == 2 && tail != i && head != j)
                {
                    continue;
                }
                let Some(cost) = self.costs[tail][head] else {
                    continue;
                };
                let id = g.add_arc(tail, head, cost)?;
                let arc = g.arc_mut(id)?;
                arc.consumption[0] = 1.0;
                arc.consumption[1] = demand(head);
                arc.reduced_cost = weight * cost - dual(0, tail) - dual(0, head)
                    + load_price * demand(head)
                    + if tail == i { constant } else { 0.0 };
                if !arc.reduced_cost.is_finite() {
                    return Err(invalid("p-step reduced cost overflow"));
                }
            }
        }
        if g.arcs
            .iter()
            .any(|a| a.reduced_cost.abs() * p as f64 >= crate::labeling::NO_ROUTE_COST * 0.5)
        {
            return Err(invalid(
                "p-step reduced costs exceed the bucket solver cost range",
            ));
        }
        let (route, labels) = elementary_minimum(&g, p, options, remaining)?;
        let Some(route) = route else {
            return Ok((None, labels));
        };
        let path: Vec<_> = route.nodes.iter().map(|&v| v as usize).collect();
        let load: f64 = path.iter().map(|&v| demand(v)).sum();
        if load > self.capacity
            || path.first() != Some(&i)
            || path.last() != Some(&j)
            || path.len() < 2
            || path.len() - 1 > p
            || (i != 0 && path.len() - 1 != p)
        {
            return Err(incomplete("p-step bucket resource replay mismatch"));
        }
        let step = Step {
            prior_demand: if job.maximum {
                self.capacity - load
            } else {
                0.0
            },
            demand: load,
            cost: route.original_cost,
            path,
        };
        let reduced = weight * step.cost
            - step_entries(&self.demands, &step)
                .iter()
                .map(|&(r, a)| a * duals[r])
                .sum::<f64>();
        if (reduced - route.cost).abs() > 1e-7 * (1.0 + reduced.abs()) {
            return Err(incomplete("p-step reduced cost replay mismatch"));
        }
        Ok((Some(step), labels))
    }
}

/// Each relaxed optimum is a lower bound for its arc-restricted subproblem.
/// A directed cycle cannot be contained in an elementary path, so forbidding
/// at least one of its arcs gives a covering disjunction. Branches may overlap;
/// canonical forbidden sets deduplicate them. Exhaustion/pruning proves the
/// best elementary cost, without changing the core label representation.
fn elementary_minimum(
    g: &Graph<2>,
    p: usize,
    options: &PricingOptions,
    budget: Duration,
) -> Result<(Option<crate::Route>, usize)> {
    if p != 4 {
        return cycle_minimum(g, p, options, budget);
    }
    // A four-arc path i,a,b,c,j can only repeat an internal vertex if a=c.
    // Partition by its first arc and forbid later returns to a. This covers
    // every elementary path, including shorter depot-start paths, and avoids
    // the potentially overlapping cycle-exclusion tree for this common p.
    let started = Instant::now();
    let (relaxed, mut labels) = relaxed_minimum(g, p, options.max_labels_per_job, budget)?;
    let Some(relaxed) = relaxed else {
        return Ok((None, labels));
    };
    if !relaxed
        .nodes
        .iter()
        .enumerate()
        .any(|(i, v)| relaxed.nodes[..i].contains(v))
    {
        return Ok((Some(relaxed), labels));
    }
    let first: Vec<_> = g
        .arcs
        .iter()
        .filter(|a| a.tail == g.source)
        .map(|a| (a.id, a.head))
        .collect();
    let mut best: Option<crate::Route> = None;
    for (solves, (id, head)) in first.into_iter().enumerate() {
        if solves + 1 >= options.max_subproblems_per_job || labels >= options.max_labels_per_job {
            return Err(incomplete("p-step prefix pricing work limit"));
        }
        let mut graph = g.clone();
        graph
            .arcs
            .retain(|a| (a.tail != g.source || a.id == id) && (a.head != head || a.id == id));
        let mut limits = options.clone();
        limits.max_labels_per_job -= labels;
        limits.max_subproblems_per_job = 1;
        let (route, generated) =
            cycle_minimum(&graph, p, &limits, budget.saturating_sub(started.elapsed()))?;
        labels += generated;
        if let Some(route) = route {
            if best.as_ref().map_or(true, |old| route.cost < old.cost) {
                best = Some(route);
            }
        }
        if best
            .as_ref()
            .is_some_and(|route| route.cost <= relaxed.cost)
        {
            break;
        }
    }
    Ok((best, labels))
}

fn cycle_minimum(
    g: &Graph<2>,
    p: usize,
    options: &PricingOptions,
    budget: Duration,
) -> Result<(Option<crate::Route>, usize)> {
    let started = Instant::now();
    let mut pending = vec![(Vec::<i32>::new(), f64::NEG_INFINITY)];
    let mut seen = BTreeSet::new();
    seen.insert(Vec::<i32>::new());
    let mut best: Option<crate::Route> = None;
    let mut labels = 0usize;
    let mut solves = 0;
    while let Some((forbidden, bound)) = pending.pop() {
        if best.as_ref().is_some_and(|r| bound >= r.cost) {
            continue;
        }
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero()
            || solves >= options.max_subproblems_per_job
            || labels >= options.max_labels_per_job
        {
            return Err(incomplete("p-step cycle pricing work/time limit"));
        }
        let mut graph = g.clone();
        graph
            .arcs
            .retain(|a| forbidden.binary_search(&a.id).is_err());
        let (route, generated) = relaxed_minimum(
            &graph,
            p,
            options.max_labels_per_job - labels,
            budget.saturating_sub(started.elapsed()),
        )?;
        solves += 1;
        labels += generated;
        let Some(route) = route else {
            continue;
        };
        if best.as_ref().is_some_and(|r| route.cost >= r.cost) {
            continue;
        }
        let mut cycle = None;
        let mut positions = vec![None; graph.vertices.len()];
        for (index, &v) in route.nodes.iter().enumerate() {
            if let Some(first) = positions[v as usize] {
                cycle = Some((first, index));
                break;
            }
            positions[v as usize] = Some(index);
        }
        if let Some((first, last)) = cycle {
            for &arc in &route.arcs[first..last] {
                let mut next = forbidden.clone();
                next.push(arc);
                next.sort_unstable();
                next.dedup();
                if seen.insert(next.clone()) {
                    pending.push((next, route.cost));
                }
            }
            // Pop the smallest available relaxed lower bound first.
            pending.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        } else {
            best = Some(route);
        }
    }
    Ok((best, labels))
}

fn relaxed_minimum(
    g: &Graph<2>,
    p: usize,
    max_labels: usize,
    budget: Duration,
) -> Result<(Option<crate::Route>, usize)> {
    let started = Instant::now();
    if budget.is_zero() || max_labels == 0 {
        return Err(incomplete("p-step bucket pricing work/time limit"));
    }
    let mut engine = LabelingEngine::prepare(
        g,
        EngineControls {
            bidirectional: false,
            target_buckets: 8,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )?;
    let result = engine.solve(
        g,
        &Controls {
            max_labels,
            max_columns: 1,
            max_route_arcs: Some(p),
            time_limit: budget.saturating_sub(started.elapsed()),
            negative_only: false,
            zero_threshold: 0.0,
        },
    )?;
    if !result.exact {
        return Err(incomplete("p-step bucket pricing incomplete"));
    }
    Ok((
        result.columns.into_iter().next(),
        result.statistics.generated,
    ))
}
