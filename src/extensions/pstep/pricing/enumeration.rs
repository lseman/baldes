//! Streaming elementary enumeration for small endpoint jobs.
use super::{Job, PricingMethod, PricingOptions};
use crate::extensions::pstep::{step_entries, Problem, Step};
use crate::{incomplete, invalid, Result};
use std::time::{Duration, Instant};

pub(super) fn select_method(n: usize, p: usize, method: PricingMethod) -> PricingMethod {
    if method != PricingMethod::Auto {
        return method;
    }
    // Dense upper estimate of internal prefixes per endpoint. Saturation
    // prevents large p from overflowing; keep both depth and work bounded.
    let mut width = 1usize;
    let mut work = 1usize;
    for _ in 1..p.min(7) {
        width = width.saturating_mul(n);
        work = work.saturating_add(width);
    }
    if p <= 6 && work <= 100_000 {
        PricingMethod::Enumeration
    } else {
        PricingMethod::Bucket
    }
}

pub(super) fn price_job(
    problem: &Problem,
    job: Job,
    p: usize,
    duals: &[f64],
    weight: f64,
    options: &PricingOptions,
    budget: Duration,
) -> Result<(Option<Step>, usize)> {
    let n = problem.demands.len();
    let dual = |offset: usize, v: usize| {
        if v == 0 || v == n + 1 {
            0.0
        } else {
            duals[offset * n + v - 1]
        }
    };
    let initial_load = if job.start == 0 {
        0.0
    } else {
        problem.demands[job.start - 1]
    };
    let gi = dual(2, job.start);
    let gj = dual(2, job.end);
    let constant = -dual(1, job.start)
        + dual(1, job.end)
        + if job.maximum {
            problem.capacity * (gj - gi)
        } else {
            (gj - gi) * initial_load
        };
    let mut search = Search {
        problem,
        job,
        p,
        duals,
        weight,
        load_price: if job.maximum { gi } else { gj },
        options,
        started: Instant::now(),
        budget,
        path: vec![job.start],
        visited: vec![false; n + 2],
        best: None,
        best_cost: f64::INFINITY,
        extensions: 0,
    };
    search.visited[job.start] = true;
    search.visit(initial_load, 0.0, constant)?;
    if search.started.elapsed() >= budget {
        return Err(incomplete("p-step enumeration time limit"));
    }
    if let Some(step) = &search.best {
        let replay = weight * step.cost
            - step_entries(&problem.demands, step)
                .iter()
                .map(|&(r, a)| a * duals[r])
                .sum::<f64>();
        if !replay.is_finite() || (replay - search.best_cost).abs() > 1e-7 * (1.0 + replay.abs()) {
            return Err(incomplete(
                "p-step enumeration reduced cost replay mismatch",
            ));
        }
    }
    Ok((search.best, search.extensions))
}

struct Search<'a> {
    problem: &'a Problem,
    job: Job,
    p: usize,
    duals: &'a [f64],
    weight: f64,
    load_price: f64,
    options: &'a PricingOptions,
    started: Instant,
    budget: Duration,
    path: Vec<usize>,
    visited: Vec<bool>,
    best: Option<Step>,
    best_cost: f64,
    extensions: usize,
}

impl Search<'_> {
    fn visit(&mut self, load: f64, cost: f64, reduced: f64) -> Result<()> {
        let n = self.problem.demands.len();
        let tail = *self.path.last().unwrap();
        // path.len() is the number of arcs after adding the next vertex.
        let hops = self.path.len();
        let heads = if hops == self.p {
            self.job.end..=self.job.end
        } else {
            1..=n + 1
        };
        for head in heads {
            let terminal = head == self.job.end;
            if self.visited[head]
                || (head == n + 1 && !terminal)
                || (tail == 0 && head == n + 1)
                || (terminal && self.job.start != 0 && hops != self.p)
                || (!terminal && hops >= self.p)
            {
                continue;
            }
            self.extensions += 1;
            if self.extensions > self.options.max_labels_per_job {
                return Err(incomplete("p-step enumeration extension limit"));
            }
            if (self.extensions == 1 || self.extensions % 256 == 0)
                && self.started.elapsed() >= self.budget
            {
                return Err(incomplete("p-step enumeration time limit"));
            }
            let Some(arc_cost) = self.problem.costs[tail][head] else {
                continue;
            };
            let demand = if head <= n {
                self.problem.demands[head - 1]
            } else {
                0.0
            };
            let next_load = load + demand;
            if next_load > self.problem.capacity {
                continue;
            }
            let degree = |v: usize| {
                if v == 0 || v == n + 1 {
                    0.0
                } else {
                    self.duals[v - 1]
                }
            };
            let next_cost = cost + arc_cost;
            let next_reduced = reduced + self.weight * arc_cost - degree(tail) - degree(head)
                + self.load_price * demand;
            if !next_cost.is_finite() || !next_reduced.is_finite() {
                return Err(invalid("p-step enumeration cost overflow"));
            }
            self.path.push(head);
            if terminal {
                if next_reduced < self.best_cost {
                    self.best_cost = next_reduced;
                    self.best = Some(Step {
                        path: self.path.clone(),
                        cost: next_cost,
                        demand: next_load,
                        prior_demand: if self.job.maximum {
                            self.problem.capacity - next_load
                        } else {
                            0.0
                        },
                    });
                }
            } else {
                self.visited[head] = true;
                self.visit(next_load, next_cost, next_reduced)?;
                self.visited[head] = false;
            }
            self.path.pop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_pricing_bounds_enumeration_work() {
        assert_eq!(
            select_method(31, 3, PricingMethod::Auto),
            PricingMethod::Enumeration
        );
        assert_eq!(
            select_method(21, 4, PricingMethod::Auto),
            PricingMethod::Enumeration
        );
        assert_eq!(
            select_method(64, 4, PricingMethod::Auto),
            PricingMethod::Bucket
        );
        assert_eq!(
            select_method(usize::MAX, 6, PricingMethod::Auto),
            PricingMethod::Bucket
        );
    }
}
