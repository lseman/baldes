//! One pricing coordinator for collections of one or more networks.
use super::cg;
use crate::{incomplete, invalid, pricing::Solver, Result, Route};
use std::time::Instant;
mod additive;
pub(crate) use additive::Additive;
mod scheduling;
pub use scheduling::NetworkPricingSchedule;
pub(crate) use scheduling::Scheduler;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Estimate {
    pub minimum: f64,
    pub correction: f64,
    pub zero_quantity: bool,
}
impl Estimate {
    pub fn zero() -> Self {
        Self {
            minimum: 0.,
            correction: 0.,
            zero_quantity: true,
        }
    }
    pub fn contribution(minimum: f64, lower: f64, upper: f64) -> Option<Self> {
        if minimum >= 0. && lower == 0. {
            return Some(Self::zero());
        }
        let quantity = if minimum < 0. { upper } else { lower };
        if !quantity.is_finite() || quantity < 0. {
            return None;
        }
        let mut sum = crate::master::numerics::ExactSum::default();
        sum.product_upper(-minimum, quantity)?;
        Some(Self {
            minimum,
            correction: -sum.upper()?,
            zero_quantity: false,
        })
    }
}

/// Domain proofs supplied by a model adapter; admission is selected here.
#[derive(Default)]
pub(crate) struct Envelopes {
    pub nonnegative: bool,
    pub finite: Option<Estimate>,
    pub near_zero: Option<Estimate>,
    pub original: bool,
}
pub(crate) struct Request {
    pub phase: usize,
    pub exact: bool,
    pub offset: f64,
    pub lower: f64,
    pub upper: f64,
    pub envelopes: Envelopes,
}
/// Stabilization subgradient weight of a network's best path. The reference
/// weights it by the largest convexity right-hand side (the upper
/// multiplicity) whatever the sign of its reduced cost; the Lagrangian
/// correction keeps the sign-dependent quantity.
fn gradient_weight(request: &Request, quantity: f64) -> f64 {
    if request.upper.is_finite() {
        request.upper
    } else {
        quantity
    }
}
pub(crate) struct Collected {
    pub columns: Vec<cg::Column>,
    pub fresh: usize,
    pub best_entries: Option<Vec<(usize, f64)>>,
    /// The route behind `best_entries`, as returned by the oracle.
    pub best_route: Option<Route>,
}
pub(crate) enum BestSource {
    /// Coefficients are recomputed from the route at the current rows.
    Route(Route),
    /// A retained-domain column, valid while the row count is unchanged.
    Entries {
        entries: Vec<(usize, f64)>,
        rows: usize,
    },
}
pub(crate) struct BestPath {
    pub source: BestSource,
    pub weight: f64,
}
impl Scheduler {
    fn remember_best(&mut self, index: usize, best: Option<BestPath>) {
        if self.best_paths.len() <= index {
            self.best_paths.resize_with(index + 1, || None);
        }
        self.best_paths[index] = best;
    }
}
pub(crate) struct Retained {
    /// Independent lower enclosure of the retained domain, including controls
    /// not represented by a candidate column. Suppresses a guessed subgradient.
    pub minimum_bound: Option<f64>,
    pub columns: Vec<(cg::Column, f64)>,
    /// Independent route caps describe an explicit finite master. Otherwise
    /// one route pattern is selected with the network multiplicity.
    pub independent_caps: bool,
    pub maximum: usize,
}

/// Model-specific cost preparation, route identities and row coefficients.
/// Scheduling, oracle invocation, proof admission and batch accounting live
/// in `price`, including for the one-element collection.
pub(crate) trait Networks<const D: usize> {
    fn active(&self) -> Vec<usize>;
    fn prepare(
        &mut self,
        index: usize,
        phase: usize,
        duals: &[f64],
        admission: bool,
    ) -> Result<Request>;
    /// Record a physical entry and lend the solver to the common invocation.
    fn enter(&mut self, index: usize) -> &mut Solver<D>;
    fn retained(&mut self, _index: usize, _duals: &[f64]) -> Option<Result<Retained>> {
        None
    }
    fn collect(
        &mut self,
        index: usize,
        routes: Vec<Route>,
        offset: f64,
        duals: &[f64],
    ) -> Result<Collected>;
    fn static_contribution(&self, _duals: &[f64]) -> Option<(f64, Vec<(usize, f64)>)> {
        Some((0., vec![]))
    }
}

pub(crate) struct Context<'a> {
    pub phase: usize,
    pub duals: &'a [f64],
    pub rhs: &'a [f64],
    pub deadline: Option<Instant>,
    pub admission: bool,
    pub master: Option<(f64, f64)>,
    pub gap: (f64, f64),
}
pub(crate) struct Batch {
    pub answer: cg::Answer,
    pub correction: Option<f64>,
    pub gradient: Option<Vec<f64>>,
}

fn admission(context: &Context<'_>, hints: &Envelopes, bounded: bool) -> Option<Estimate> {
    if !context.admission {
        return None;
    }
    if !hints.original && hints.nonnegative {
        return Some(Estimate::zero());
    }
    if let Some(bound) = hints.finite.filter(|b| b.zero_quantity) {
        return Some(bound);
    }
    if !bounded || context.master.is_none() {
        return None;
    }
    if hints.original {
        hints.finite
    } else {
        hints.near_zero.or(hints.finite)
    }
}

pub(crate) fn price<const D: usize>(
    networks: &mut impl Networks<D>,
    scheduler: &mut Scheduler,
    context: Context<'_>,
) -> Result<Batch> {
    let mut bounded = true;
    loop {
        scheduler.complete = false;
        let order = scheduler.order(context.phase, networks.active());
        let mut columns = vec![];
        let mut minimum = f64::INFINITY;
        let mut exact = true;
        let mut correction = 0.;
        let mut gradient = context.rhs.to_vec();
        let mut gradient_available = true;
        let mut used_estimate = false;
        let mut fresh = false;
        let mut visited = 0;
        // Networks not priced in this batch, with their subgradient weight
        // when known. Their last best path stands in for the gradient term.
        let mut handled = vec![];
        let mut stale: Vec<(usize, Option<f64>)> = vec![];
        for &index in &order {
            if context.deadline.is_some_and(|end| Instant::now() >= end) {
                return Err(incomplete("network pricing time limit"));
            }
            let request =
                networks.prepare(index, context.phase, context.duals, context.admission)?;
            if let Some(bound) = admission(&context, &request.envelopes, bounded) {
                used_estimate |= !bound.zero_quantity;
                gradient_available &= bound.zero_quantity;
                handled.push(index);
                stale.push((index, Some(gradient_weight(&request, 0.))));
                correction += bound.correction;
                exact &= request.exact;
                minimum = minimum.min(bound.minimum);
                scheduler.record(index, bound.minimum);
                visited += 1;
                if scheduler.finish_network(index, 0) {
                    break;
                }
                continue;
            }
            if let Some(retained) = networks.retained(index, context.duals) {
                let mut retained = retained?;
                let mut network_minimum = f64::INFINITY;
                let mut best = None;
                retained.columns.retain(|(_, cap)| *cap != 0.);
                for (column, cap) in &mut retained.columns {
                    if context.deadline.is_some_and(|end| Instant::now() >= end) {
                        return Err(incomplete("network pricing time limit"));
                    }
                    if cap.is_nan() || *cap < 0. {
                        return Err(invalid("retained column cap"));
                    }
                    column.reduced_cost = column
                        .entries
                        .iter()
                        .try_fold(column.cost, |cost, &(row, a)| {
                            Some(cost - context.duals.get(row)? * a)
                        })
                        .ok_or_else(|| invalid("retained column row"))?;
                    let reduced = column.reduced_cost;
                    if !reduced.is_finite() {
                        return Err(invalid("retained reduced cost overflow"));
                    }
                    network_minimum = network_minimum.min(reduced);
                    if best.as_ref().map_or(true, |(value, _)| reduced < *value) {
                        best = Some((reduced, column.entries.clone()));
                    }
                    if retained.independent_caps && reduced < 0. {
                        correction += reduced * *cap;
                        for &(row, a) in &column.entries {
                            *gradient
                                .get_mut(row)
                                .ok_or_else(|| invalid("network gradient row"))? -= *cap * a;
                        }
                    }
                }
                if let Some(bound) = retained.minimum_bound {
                    if !bound.is_finite() {
                        return Err(incomplete("nonfinite retained pricing certificate"));
                    }
                    network_minimum = network_minimum.min(bound);
                    gradient_available = false;
                }
                if network_minimum == f64::INFINITY {
                    network_minimum = 0.;
                }
                handled.push(index);
                if !retained.independent_caps {
                    let quantity = if network_minimum < 0. {
                        request.upper
                    } else {
                        request.lower
                    };
                    correction += quantity * network_minimum;
                    let weight = gradient_weight(&request, quantity);
                    if let Some((_, entries)) = &best {
                        for &(row, a) in entries {
                            *gradient
                                .get_mut(row)
                                .ok_or_else(|| invalid("network gradient row"))? -= weight * a;
                        }
                    } else if weight != 0. {
                        gradient_available = false;
                    }
                    scheduler.remember_best(
                        index,
                        best.map(|(_, entries)| BestPath {
                            source: BestSource::Entries {
                                entries,
                                rows: context.rhs.len(),
                            },
                            weight,
                        }),
                    );
                    retained
                        .columns
                        .sort_by(|(a, _), (b, _)| a.reduced_cost.total_cmp(&b.reduced_cost));
                }
                columns.extend(
                    retained
                        .columns
                        .into_iter()
                        .take(retained.maximum)
                        .map(|(column, _)| column),
                );
                minimum = minimum.min(network_minimum);
                scheduler.record(index, network_minimum);
                visited += 1;
                if scheduler.finish_network(index, 0) {
                    break;
                }
                continue;
            }
            let remaining = context.deadline.map_or(std::time::Duration::MAX, |end| {
                end.saturating_duration_since(Instant::now())
            });
            if remaining.is_zero() {
                return Err(incomplete("network pricing time limit"));
            }
            let solver = networks.enter(index);
            let saved = solver.controls.time_limit;
            solver.controls.time_limit = saved.min(remaining);
            let result = solver.price_functor_phase(request.phase, solver.controls.max_columns);
            solver.controls.time_limit = saved;
            let result = result?;
            if request.exact && !result.exact {
                return Err(incomplete("network exact pricing did not finish"));
            }
            exact &= result.exact;
            let reduced = result.minimum_reduced_cost + request.offset;
            if !reduced.is_finite() {
                return Err(incomplete("network minimum is not finite"));
            }
            if result.columns.is_empty() && (reduced < 0. || request.lower > 0.) {
                return Err(incomplete("required network pricing witness unavailable"));
            }
            minimum = minimum.min(reduced);
            scheduler.record(index, reduced);
            visited += 1;
            let quantity = if reduced >= 0. {
                request.lower
            } else {
                request.upper
            };
            correction += quantity * reduced;
            let collected =
                networks.collect(index, result.columns, request.offset, context.duals)?;
            handled.push(index);
            let weight = gradient_weight(&request, quantity);
            if let Some(entries) = collected.best_entries {
                for (row, value) in entries {
                    *gradient
                        .get_mut(row)
                        .ok_or_else(|| invalid("network gradient row"))? -= weight * value;
                }
            } else if weight != 0. {
                // The reference requires a best solution for every
                // subproblem with a nonzero upper multiplicity.
                gradient_available = false;
            }
            scheduler.remember_best(
                index,
                collected.best_route.map(|route| BestPath {
                    source: BestSource::Route(route),
                    weight,
                }),
            );
            fresh |= collected.fresh > 0;
            columns.extend(collected.columns);
            if scheduler.finish_network(index, collected.fresh) {
                break;
            }
        }
        scheduler.complete = visited == order.len();
        exact &= scheduler.complete;
        if minimum == f64::INFINITY {
            minimum = 0.;
        }
        for index in networks.active() {
            if !handled.contains(&index) {
                stale.push((index, None));
            }
        }
        if gradient_available {
            gradient_available =
                stale_terms(networks, scheduler, &stale, context.duals, &mut gradient)?;
        }
        if !scheduler.complete {
            // No Lagrangian bound, but the stabilization subgradient still
            // follows the reference: the unpriced networks' last best paths.
            let gradient = gradient_available
                .then(|| {
                    let (_, entries) = networks.static_contribution(context.duals)?;
                    for (row, value) in entries {
                        *gradient.get_mut(row)? -= value;
                    }
                    gradient.iter().all(|v| v.is_finite()).then_some(gradient)
                })
                .flatten();
            return Ok(Batch {
                answer: answer(context.phase, columns, minimum, false),
                correction: None,
                gradient,
            });
        }
        let contribution =
            networks
                .static_contribution(context.duals)
                .and_then(|(value, entries)| {
                    correction += value;
                    for (row, value) in entries {
                        *gradient.get_mut(row)? -= value;
                    }
                    (correction.is_finite() && gradient.iter().all(|v| v.is_finite()))
                        .then_some(correction)
                });
        if used_estimate && !fresh {
            let closes =
                context
                    .master
                    .zip(contribution)
                    .is_some_and(|((primal, offset), correction)| {
                        let bound = context
                            .duals
                            .iter()
                            .zip(context.rhs)
                            .fold(offset, |v, (d, r)| v + d * r)
                            + correction;
                        bound.is_finite()
                            && primal.is_finite()
                            && cg::gap_smaller_than_tolerance(
                                bound,
                                primal,
                                context.gap.0,
                                context.gap.1,
                            )
                    });
            if !closes && bounded {
                bounded = false;
                continue;
            }
        }
        return Ok(Batch {
            answer: answer(context.phase, columns, minimum, exact),
            correction: contribution,
            gradient: contribution.and_then(|_| gradient_available.then_some(gradient)),
        });
    }
}

/// Subtract the last best path of each network not priced in this batch.
/// Returns false when such a network has a nonzero weight but no usable path,
/// so the reference subgradient is unknown.
fn stale_terms<const D: usize>(
    networks: &mut impl Networks<D>,
    scheduler: &Scheduler,
    stale: &[(usize, Option<f64>)],
    duals: &[f64],
    gradient: &mut [f64],
) -> Result<bool> {
    for &(index, weight) in stale {
        let cached = scheduler.best_paths.get(index).and_then(Option::as_ref);
        let Some(weight) = weight.or(cached.map(|best| best.weight)) else {
            return Ok(false);
        };
        if weight == 0. {
            continue;
        }
        let entries = match cached.map(|best| &best.source) {
            Some(BestSource::Route(route)) => {
                networks
                    .collect(index, vec![route.clone()], 0., duals)?
                    .best_entries
            }
            Some(BestSource::Entries { entries, rows }) if *rows == gradient.len() => {
                Some(entries.clone())
            }
            _ => None,
        };
        let Some(entries) = entries else {
            return Ok(false);
        };
        for (row, value) in entries {
            *gradient
                .get_mut(row)
                .ok_or_else(|| invalid("network gradient row"))? -= weight * value;
        }
    }
    Ok(true)
}

fn answer(phase: usize, columns: Vec<cg::Column>, minimum: f64, exact: bool) -> cg::Answer {
    if std::env::var_os("BALDES_DEBUG_COLUMNS").is_some() {
        eprintln!("price phase={phase} columns={}", columns.len());
        for c in &columns {
            eprintln!("  {:.17e} {}", c.reduced_cost, c.identity);
        }
    }
    cg::Answer {
        columns,
        minimum,
        exact,
        bound_minimum: None,
    }
}
