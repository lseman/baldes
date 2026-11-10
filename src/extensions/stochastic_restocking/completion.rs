//! Recovered load-walk and subset completion bounds, adapted to current plugins.
use super::{
    bounds::{LoadWalkBounds, SubsetWalkBounds},
    BellmanState, Problem, RestockingResource,
};
use crate::core::rounding::{add_down, add_up, multiply_down};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(super) struct Completion {
    pub costs: Vec<f64>,
    prefixes: SubsetWalkBounds,
    threshold: f64,
    cost_error: f64,
}

fn table_work(resource: &RestockingResource, subsets: usize) -> Option<usize> {
    let n = resource.problem.demands().len();
    let integral = resource
        .problem
        .demands()
        .iter()
        .all(|d| d.mean().fract() == 0.);
    let levels = if integral && resource.problem.capacity() <= super::bounds::MAX_LOAD_LEVELS {
        resource.problem.capacity().checked_add(1)?
    } else {
        super::bounds::MAX_LOAD_LEVELS + 1
    };
    let zero = resource.problem.demands().iter().any(|d| d.mean() < 1.);
    resource
        .arcs
        .len()
        .checked_add(n + 1)?
        .checked_mul(levels)?
        .checked_mul(if zero { n + 2 } else { 1 })?
        .checked_mul(subsets)
}
fn weighted(resource: &RestockingResource, costs: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
    if costs.len() != resource.arcs.len() || costs.iter().any(|c| !c.is_finite()) {
        return None;
    }
    Some(
        resource
            .arcs
            .iter()
            .zip(costs)
            .map(|(&(i, j), &c)| (i, j, c))
            .collect(),
    )
}

/// Recover customer row prices from the depot arc that carries the customer
/// coefficient in this physical orientation. Reversing the graph moves that
/// coefficient from original arrivals to original departures.
pub(super) fn customer_potentials(
    resource: &RestockingResource,
    arcs: &[(usize, usize, f64)],
) -> Vec<f64> {
    let mut potentials: Vec<f64> = vec![0.; resource.problem.demands().len() + 1];
    for &(i, j, cost) in arcs {
        if resource.reversed && j == 0 && i != 0 {
            potentials[i] = potentials[i].max(resource.problem.costs()[i][0] - cost);
        } else if !resource.reversed && i == 0 && j != 0 {
            potentials[j] = potentials[j].max(resource.problem.costs()[0][j] - cost);
        }
    }
    potentials
}

pub(super) fn minimum(
    resource: &RestockingResource,
    costs: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    if Instant::now() >= deadline {
        return None;
    }
    // Bounds are prepared over a route superset, ignoring branch restrictions.
    // Reserve/debit the complete bounded table setup, including zero-load scans.
    *work = work.checked_sub(table_work(resource, 1)?)?;
    let arcs = weighted(resource, costs)?;
    let p = &resource.problem;
    let table = LoadWalkBounds::new(p, &arcs, &vec![0.; p.demands().len()]);
    let recourse = super::certificate::objective_bounds(
        resource,
        p.demands().len() + 1,
        work,
        Some(deadline),
    )?[0];
    let result = add_down(table.route()?, recourse)?;
    (Instant::now() < deadline && result.is_finite()).then_some(result)
}

impl Completion {
    pub(super) fn new(
        resource: &RestockingResource,
        costs: &[f64],
        threshold: f64,
        deadline: Instant,
    ) -> Option<Self> {
        let p = &resource.problem;
        if !threshold.is_finite() || p.demands().len() > 63 || Instant::now() >= deadline {
            return None;
        }
        let started = Instant::now();
        let limit = deadline.min(started.checked_add(Duration::from_millis(500))?);
        // Literal normalized probabilities can sum slightly below one in
        // exact arithmetic. Discounting a prefix can then reduce its suffix
        // value. Bound that loss before applying the recovered completion rule.
        let mut mass_lower: f64 = 1.;
        for demand in p.demands() {
            let mass = demand
                .outcomes()
                .iter()
                .try_fold(0., |sum, &(_, probability)| add_down(sum, probability))?;
            mass_lower = mass_lower.min(mass);
        }
        let mut alpha = 1.;
        for _ in p.demands() {
            alpha = multiply_down(alpha, mass_lower)?;
        }
        let upper = super::certificate::objective_bounds(
            resource,
            p.demands().len() + 1,
            &mut 1_000_000,
            Some(limit),
        )?[1];
        // Prefix planned travel plus suffix cost is at most twice this bound.
        let deficit = add_up(1., -alpha)?.max(0.);
        let cost_error =
            -multiply_down(-upper, 2.).and_then(|twice| multiply_down(twice, deficit))?;
        let arcs = weighted(resource, costs)?;
        let potentials = customer_potentials(resource, &arcs);
        let mut members: Vec<_> = (1..=p.demands().len())
            .filter(|&j| potentials[j] > 0.)
            .collect();
        let reward = |j: usize| potentials[j] / p.demands()[j - 1].mean().max(1e-9);
        members.sort_by(|&a, &b| reward(b).total_cmp(&reward(a)).then(a.cmp(&b)));
        members.truncate(4);
        while !members.is_empty() && table_work(resource, 1usize << members.len())? > 16_000_000 {
            members.pop();
        }
        if table_work(resource, 1usize << members.len())? > 16_000_000 {
            return None;
        }
        let charges: Vec<_> = (1..=p.demands().len())
            .map(|j| {
                (0..=p.demands().len())
                    .map(|i| {
                        add_down(
                            add_down(p.costs()[i][0], p.costs()[0][j]).unwrap_or(f64::NEG_INFINITY),
                            -p.costs()[i][j],
                        )
                        .unwrap_or(f64::NEG_INFINITY)
                    })
                    .fold(0., f64::min)
            })
            .collect();
        if charges.iter().any(|c| !c.is_finite()) {
            return None;
        }
        let prefixes = SubsetWalkBounds::new(p, &arcs, &charges, members, false);
        if Instant::now() >= limit {
            return None;
        }
        Some(Self {
            costs: costs.to_vec(),
            prefixes,
            threshold,
            cost_error,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rejects(
        &self,
        p: &Problem,
        state: &BellmanState,
        old: &BellmanState,
        i: usize,
        j: usize,
        mean: f64,
        updated: bool,
    ) -> bool {
        let visited = state.visited;
        let room = add_up(p.capacity() as f64, -mean)
            .and_then(|room| add_up(room, p.demands()[i - 1].mean()));
        let Some(room) = room else {
            return false;
        };
        let prefix = self
            .prefixes
            .prefix(i, j, room + 1e-9 * (1. + p.capacity() as f64), visited);
        if prefix == f64::INFINITY {
            return true;
        }
        let continuation = if updated {
            state.values.iter().copied().fold(f64::INFINITY, f64::min)
        } else if j == 0 {
            0.
        } else {
            let detour = p.costs()[i][0] + p.costs()[0][j] - p.costs()[i][j];
            old.values.iter().copied().fold(f64::INFINITY, f64::min) + detour.min(0.)
        };
        let bound = state.reduced + continuation + prefix - self.cost_error;
        bound.is_finite()
            && bound
                >= self.threshold
                    + 1e-8
                        * (1.
                            + state.reduced.abs()
                            + continuation.abs()
                            + state.suffix_travel.abs()
                            + prefix.abs())
    }
}
