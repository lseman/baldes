//! Load-indexed walk relaxation (q-routes) for stochastic pricing bounds.
//!
//! Customers keep their mean demand, rounded down to `scale` units, and walks
//! may repeat customers. Every elementary mean-feasible route is such a walk,
//! and each visit is charged its minimum nonpositive preventive detour, which
//! bounds Bellman recourse from below. Minima are therefore valid lower bounds
//! for elementary stochastic reduced costs, also on nonmetric matrices.
use super::Problem;
use rayon::prelude::*;

/// Load levels are capped so that a table costs O(levels * arcs) to build.
const MAX_LOAD_LEVELS: usize = 1024;

#[derive(Clone, Debug)]
pub(super) struct LoadWalkBounds {
    scale: f64,
    levels: usize,
    width: usize,
    /// `values[k * width + v]`: minimum weights of walks from the depot that
    /// end by entering customer `v`, with at most `k` load units including
    /// `v` and no immediate customer reversal (a 2-cycle elimination).
    values: Vec<Entry>,
    route: f64,
}

impl LoadWalkBounds {
    /// `arcs` are problem-orientation `(from, to, weight)` with depot 0. The
    /// weight of an arc into a customer is extended by that customer's charge.
    pub(super) fn new(problem: &Problem, arcs: &[(usize, usize, f64)], charges: &[f64]) -> Self {
        let n = problem.demands.len();
        let width = n + 1;
        let capacity = problem.capacity.max(1) as f64;
        let integral = problem
            .demands
            .iter()
            .all(|d| (d.mean() - d.mean().round()).abs() <= 1e-9);
        // Integral means are exact at unit scale; finer scales only help
        // fractional means.
        let scale = if integral && problem.capacity <= MAX_LOAD_LEVELS {
            1.
        } else if problem.capacity <= MAX_LOAD_LEVELS {
            (MAX_LOAD_LEVELS / problem.capacity.max(1)) as f64
        } else {
            MAX_LOAD_LEVELS as f64 / capacity
        };
        let levels = (problem.capacity as f64 * scale).floor() as usize + 1;
        // Rounding down keeps every real walk inside the discretized domain.
        let units: Vec<usize> = std::iter::once(0)
            .chain(problem.demands.iter().map(|d| {
                let units = d.mean() * scale;
                // Within 1e-9 of an integer; query slack absorbs it.
                if integral {
                    units.round()
                } else {
                    units.floor()
                }
                .max(0.) as usize
            }))
            .collect();
        let mut incoming = vec![Vec::new(); width];
        let mut returns = vec![f64::INFINITY; width];
        for &(from, to, weight) in arcs {
            if to == 0 {
                returns[from] = returns[from].min(weight);
            } else if from != to {
                incoming[to].push((from, weight + charges[to - 1]));
            }
        }
        let zero_units = units.iter().skip(1).any(|&u| u == 0);
        let mut values = vec![Entry::EMPTY; levels * width];
        for k in 0..levels {
            let (lower, level) = values.split_at_mut(k * width);
            let level = &mut level[..width];
            if k > 0 {
                level.copy_from_slice(&lower[(k - 1) * width..k * width]);
            }
            // Positive-unit heads read only completed lower levels; zero-unit
            // heads need a same-level fixpoint, and a negative cycle among
            // them leaves the relaxation unbounded.
            let passes = if zero_units { width + 1 } else { 1 };
            let mut converged = false;
            for _ in 0..passes {
                let mut changed = false;
                for v in 1..width {
                    if units[v] > k {
                        continue;
                    }
                    let base = k - units[v];
                    for &(a, weight) in &incoming[v] {
                        let start = if a == 0 {
                            0.
                        } else if base == k {
                            level[a].avoiding(v)
                        } else {
                            lower[base * width + a].avoiding(v)
                        };
                        changed |= level[v].insert(start + weight, a);
                    }
                }
                if !changed || !zero_units {
                    converged = true;
                    break;
                }
            }
            if !converged {
                for entry in &mut values[k * width..] {
                    *entry = Entry::UNBOUNDED;
                }
                break;
            }
        }
        let top = (levels - 1) * width;
        let route = (1..width)
            .map(|v| values[top + v].best + returns[v])
            .fold(f64::INFINITY, f64::min);
        Self {
            scale,
            levels,
            width,
            values,
            route,
        }
    }

    /// Lower bound on a walk from the depot that enters `v` with total mean
    /// load at most `load` and continues to `next`; an elementary route never
    /// enters `v` from a customer `next`. Infinite when no such walk exists.
    pub(super) fn prefix(&self, v: usize, next: usize, load: f64) -> f64 {
        let units = load * self.scale;
        if units < -1e-9 * (1. + self.scale) {
            return f64::INFINITY;
        }
        let k = ((units + 1e-9 * (1. + units.abs())).floor() as usize).min(self.levels - 1);
        let entry = &self.values[k * self.width + v];
        if next == 0 {
            entry.best
        } else {
            entry.avoiding(next)
        }
    }

    /// Lower bound on every nonempty mean-feasible route; None if unbounded.
    pub(super) fn route(&self) -> Option<f64> {
        (self.route > f64::NEG_INFINITY).then_some(self.route)
    }
}

/// Walk tables for every subset `S` of a few customers `M` (Florio et al.
/// 2023, RCSP completion bound). Table `S` drops arcs leaving customers in
/// `S`. An elementary completion of a reverse label never passes through a
/// customer its suffix already visited, so its prefix walk is in the table of
/// `visited & M`; that table is never weaker than the unrestricted one. An
/// optional knapsack bound adds elementarity over all unvisited customers.
#[derive(Clone, Debug)]
pub(super) struct SubsetWalkBounds {
    members: Vec<usize>,
    tables: Vec<LoadWalkBounds>,
    knapsack: Option<KnapsackBound>,
}

/// Elementary knapsack completion bound (Florio et al. 2023). Arc costs are
/// nonnegative, so a prefix walk entering `v` costs at least the entering
/// weight `e_v` plus `e_u` for each distinct unvisited customer `u` it
/// serves within the remaining mean load. Its LP relaxation, greedy by
/// gain per unit of mean demand, overestimates the knapsack gain.
#[derive(Clone, Debug)]
struct KnapsackBound {
    /// Cheapest reduced arc into each vertex plus its recourse charge.
    entering: Vec<f64>,
    means: Vec<f64>,
    /// Customers with negative entering weight, best gain per unit first.
    order: Vec<usize>,
}

impl KnapsackBound {
    fn new(problem: &Problem, arcs: &[(usize, usize, f64)], charges: &[f64]) -> Self {
        let n = problem.demands.len();
        let mut entering = vec![f64::INFINITY; n + 1];
        for &(from, to, weight) in arcs {
            if to != 0 && from != to {
                entering[to] = entering[to].min(weight + charges[to - 1]);
            }
        }
        let means: Vec<f64> = std::iter::once(0.)
            .chain(problem.demands.iter().map(|d| d.mean()))
            .collect();
        let mut order: Vec<usize> = (1..=n).filter(|&u| entering[u] < 0.).collect();
        let ratio = |u: usize| {
            if means[u] > 0. {
                -entering[u] / means[u]
            } else {
                f64::INFINITY
            }
        };
        order.sort_by(|&a, &b| ratio(b).total_cmp(&ratio(a)).then(a.cmp(&b)));
        Self {
            entering,
            means,
            order,
        }
    }

    fn prefix(&self, v: usize, load: f64, visited: u64) -> f64 {
        let mut room = load - self.means[v];
        if room < -1e-9 * (1. + load.abs()) {
            return f64::INFINITY;
        }
        room = room.max(0.);
        let mut gain = 0.;
        for &u in &self.order {
            if u == v || visited & (1 << (u - 1)) != 0 {
                continue;
            }
            let weight = self.means[u];
            if weight <= room {
                room -= weight;
                gain -= self.entering[u];
            } else {
                gain -= self.entering[u] * room / weight;
                break;
            }
        }
        // Round the gain up so float error never tightens the bound.
        self.entering[v] - gain - 1e-9 * (1. + gain.abs())
    }
}

impl SubsetWalkBounds {
    pub(super) fn new(
        problem: &Problem,
        arcs: &[(usize, usize, f64)],
        charges: &[f64],
        members: Vec<usize>,
        knapsack: bool,
    ) -> Self {
        let knapsack = knapsack.then(|| KnapsackBound::new(problem, arcs, charges));
        let tables = (0..1usize << members.len())
            .into_par_iter()
            .map(|mask| {
                let blocked = |v: usize| {
                    members
                        .iter()
                        .enumerate()
                        .any(|(k, &c)| mask & (1 << k) != 0 && c == v)
                };
                let kept: Vec<_> = arcs
                    .iter()
                    .filter(|&&(from, _, _)| !blocked(from))
                    .copied()
                    .collect();
                LoadWalkBounds::new(problem, &kept, charges)
            })
            .collect();
        Self {
            members,
            tables,
            knapsack,
        }
    }

    /// The unrestricted table, `S` empty.
    pub(super) fn unrestricted(&self) -> &LoadWalkBounds {
        &self.tables[0]
    }

    /// `LoadWalkBounds::prefix` in the table of `visited & M`; bit `c - 1` of
    /// `visited` marks customer `c`.
    pub(super) fn prefix(&self, v: usize, next: usize, load: f64, visited: u64) -> f64 {
        let mask = self
            .members
            .iter()
            .enumerate()
            .filter(|&(_, &c)| visited & (1 << (c - 1)) != 0)
            .fold(0, |mask, (k, _)| mask | (1 << k));
        let walk = self.tables[mask].prefix(v, next, load);
        self.knapsack
            .as_ref()
            .map_or(walk, |k| walk.max(k.prefix(v, load, visited)))
    }
}

/// Best walk weight and the best weight with a different predecessor.
#[derive(Clone, Copy, Debug)]
struct Entry {
    best: f64,
    from: usize,
    second: f64,
}

impl Entry {
    const EMPTY: Self = Self {
        best: f64::INFINITY,
        from: usize::MAX,
        second: f64::INFINITY,
    };
    const UNBOUNDED: Self = Self {
        best: f64::NEG_INFINITY,
        from: usize::MAX,
        second: f64::NEG_INFINITY,
    };

    fn avoiding(&self, predecessor: usize) -> f64 {
        if self.from == predecessor {
            self.second
        } else {
            self.best
        }
    }

    fn insert(&mut self, value: f64, from: usize) -> bool {
        if from == self.from {
            if value < self.best {
                self.best = value;
                return true;
            }
        } else if value < self.best {
            self.second = self.best;
            self.best = value;
            self.from = from;
            return true;
        } else if value < self.second {
            self.second = value;
            return true;
        }
        false
    }
}
