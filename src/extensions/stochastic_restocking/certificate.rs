//! Independent outward Bellman evaluation for a complete physical route.
use super::RestockingResource;
use crate::core::rounding::{add_down, add_up, multiply_down};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Range(f64, f64);
impl Range {
    fn point(x: f64) -> Self {
        Self(x, x)
    }
    fn add(self, b: Self) -> Option<Self> {
        Some(Self(add_down(self.0, b.0)?, add_up(self.1, b.1)?))
    }
    fn scale(self, p: f64) -> Option<Self> {
        Some(Self(multiply_down(self.0, p)?, -multiply_down(-self.1, p)?))
    }
    fn min(self, b: Self) -> Self {
        Self(self.0.min(b.0), self.1.min(b.1))
    }
    fn subtract(self, b: Self) -> Option<Self> {
        self.add(Self(-b.1, -b.0))
    }
}
fn tick(work: &mut usize, deadline: Option<Instant>) -> Option<()> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return None;
    }
    *work = work.checked_sub(1)?;
    Some(())
}

/// A route has at most n distinct customers. With depot-triangle costs,
/// preventive replenishment cannot undercut a planned edge, and failure travel
/// is nonnegative. Account for literal PMF mass with alpha <= min(1, mass)^n:
/// total >= alpha * planned travel. If depot triangles fail by at most delta,
/// subtract at most delta per planned edge. The upper enclosure follows the direct-travel policy,
/// with at most one failure trip per customer (demand support <= capacity).
pub(super) fn objective_bounds(
    resource: &RestockingResource,
    max_arcs: usize,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<[f64; 2]> {
    tick(work, deadline)?;
    let n = resource.problem.demands().len().min(max_arcs);
    if n == 0 {
        return Some([0.; 2]);
    }
    let c = resource.problem.costs();
    let mut maximum: f64 = 0.;
    let mut deficit: f64 = 0.;
    for (i, row) in c.iter().enumerate() {
        for (j, &cost) in row.iter().enumerate() {
            tick(work, deadline)?;
            maximum = maximum.max(cost);
            deficit = deficit.max(add_up(cost, -add_down(c[i][0], c[0][j])?)?);
        }
    }
    let mut mass_lower: f64 = 1.;
    let mut mass_upper: f64 = 1.;
    for demand in resource.problem.demands() {
        let mut mass = Range::point(0.);
        for &(_, probability) in demand.outcomes() {
            tick(work, deadline)?;
            mass = mass.add(Range::point(probability))?;
        }
        mass_lower = mass_lower.min(mass.0);
        mass_upper = mass_upper.max(mass.1);
    }
    let mut alpha = 1.;
    let mut beta = Range::point(1.);
    for _ in 0..n {
        tick(work, deadline)?;
        alpha = multiply_down(alpha, mass_lower)?;
        beta = beta.scale(mass_upper)?;
    }
    let hops = n.checked_add(1)?;
    if hops as u128 > (1u128 << 53) {
        return None;
    }
    let travel = Range::point(maximum).scale(hops as f64)?;
    let lower = add_down(
        multiply_down(add_down(alpha, -1.)?, travel.1)?,
        multiply_down(-deficit, hops as f64)?,
    )?;
    let upper = travel.scale(4.)?.scale(beta.1)?.1;
    tick(work, deadline)?;
    (lower.is_finite() && upper.is_finite()).then_some([lower, upper])
}

/// Assignment relaxation with depot copies for unused vehicles. Hungarian
/// potentials are only proposals: recheck every permitted edge with directed
/// arithmetic before publishing the dual bound. Customer-only cycles enlarge
/// the routing domain, so they cannot invalidate the lower bound.
pub(super) fn routing_lower_bound(problem: &super::Problem, vehicles: usize) -> Option<f64> {
    let customers = problem.demands().len();
    if customers == 0 {
        return Some(0.);
    }
    let n = customers.checked_add(vehicles.min(customers))?;
    let deadline = Some(Instant::now().checked_add(std::time::Duration::from_millis(100))?);
    let mut work = 1_000_000;
    let vertex = |i: usize| if i <= customers { i } else { 0 };
    let cost = |i: usize, j: usize| {
        if i == j && i <= customers {
            f64::INFINITY
        } else {
            problem.costs()[vertex(i)][vertex(j)]
        }
    };
    let mut u = vec![0.; n + 1];
    let mut v = u.clone();
    let mut p = vec![0; n + 1];
    let mut way = p.clone();
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minimum = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                tick(&mut work, deadline)?;
                if !used[j] {
                    let reduced = cost(i0, j) - u[i0] - v[j];
                    if reduced < minimum[j] {
                        minimum[j] = reduced;
                        way[j] = j0;
                    }
                    if minimum[j] < delta {
                        delta = minimum[j];
                        j1 = j;
                    }
                }
            }
            if !delta.is_finite() {
                return None;
            }
            for j in 0..=n {
                tick(&mut work, deadline)?;
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minimum[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut bound = 0.;
    for i in 1..=n {
        let mut row = f64::INFINITY;
        for (j, &potential) in v.iter().enumerate().skip(1) {
            tick(&mut work, deadline)?;
            if cost(i, j).is_finite() {
                row = row.min(add_down(cost(i, j), -potential)?);
            }
        }
        bound = add_down(bound, row)?;
    }
    for &potential in &v[1..] {
        bound = add_down(bound, potential)?;
    }
    let resource = RestockingResource::new(problem.clone(), vec![]).ok()?;
    let recourse = objective_bounds(&resource, customers + 1, &mut work, deadline)?[0];
    // Each nonempty route can contribute the negative envelope at most once.
    bound = add_down(
        bound,
        multiply_down(recourse.min(0.), vehicles.min(customers) as f64)?,
    )?;
    Some(bound.max(0.))
}
fn expectation(
    resource: &RestockingResource,
    j: usize,
    load: usize,
    next: &[Range],
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<Range> {
    let q = resource.problem.capacity();
    let c = resource.problem.costs();
    let mut sum = Range::point(0.);
    for &(d, p) in resource.problem.demands()[j - 1].outcomes() {
        tick(work, deadline)?;
        let value = if d <= load {
            next[load - d]
        } else {
            Range::point(c[j][0])
                .add(Range::point(c[0][j]))?
                .add(next[q + load - d])?
        };
        sum = sum.add(value.scale(p)?)?;
    }
    Some(sum)
}
pub(super) fn certify(
    resource: &RestockingResource,
    arcs: &[usize],
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(bool, [f64; 2])> {
    tick(work, deadline)?;
    let mut route = Vec::new();
    let mut current = 0;
    for (position, &arc) in arcs.iter().enumerate() {
        tick(work, deadline)?;
        let &(i, j) = resource.arcs.get(arc)?;
        if i != current || (j == 0 && position + 1 != arcs.len()) || (j != 0 && route.contains(&j))
        {
            return Some((false, [0.; 2]));
        }
        if j != 0 {
            route.push(j);
        }
        current = j;
    }
    if current != 0 {
        return Some((false, [0.; 2]));
    }
    let mut mean = Range::point(0.);
    for (k, d) in resource.problem.demands().iter().enumerate() {
        tick(work, deadline)?;
        if route.contains(&(k + 1)) {
            mean = mean.add(Range::point(d.mean()))?;
        }
    }
    let q = resource.problem.capacity();
    if mean.0 > q as f64 {
        return Some((false, [0.; 2]));
    }
    if mean.1 > q as f64 {
        return None;
    }
    let Some(&last) = route.last() else {
        return Some((true, [0.; 2]));
    };
    // Debit allocation work as well; huge capacities cannot bypass the budget.
    *work = work.checked_sub(q.checked_add(1)?.checked_mul(2)?)?;
    tick(work, deadline)?;
    let c = resource.problem.costs();
    let mut next = Vec::new();
    next.try_reserve_exact(q.checked_add(1)?).ok()?;
    next.resize(q + 1, Range::point(c[last][0]));
    let mut values = next.clone();
    for pair in route.windows(2).rev() {
        let (i, j) = (pair[0], pair[1]);
        let refill = Range::point(c[i][0])
            .add(Range::point(c[0][j]))?
            .add(expectation(resource, j, q, &next, work, deadline)?)?;
        for (load, value) in values.iter_mut().enumerate() {
            tick(work, deadline)?;
            *value = Range::point(c[i][j])
                .add(expectation(resource, j, load, &next, work, deadline)?)?
                .min(refill);
        }
        std::mem::swap(&mut values, &mut next);
    }
    let total = Range::point(c[0][route[0]])
        .add(expectation(resource, route[0], q, &next, work, deadline)?)?;
    let mut internal = Range::point(0.);
    for pair in route.windows(2) {
        tick(work, deadline)?;
        internal = internal.add(Range::point(c[pair[0]][pair[1]]))?;
    }
    let travel = Range::point(c[0][route[0]])
        .add(internal)?
        .add(Range::point(c[last][0]))?;
    let recourse = total.subtract(travel)?;
    tick(work, deadline)?;
    Some((true, [recourse.0, recourse.1]))
}

/// Independent outward continuation values for the strict resource-only
/// comparison. The hot scalar-compensated comparison uses a uniform enclosure.
#[derive(Debug, PartialEq)]
pub(super) struct Continuation {
    values: Vec<Range>,
    travel: Range,
}
impl Continuation {
    pub(super) fn excess(&self, other: &Self, mass: [f64; 2]) -> Option<f64> {
        let mut delta = f64::NEG_INFINITY;
        for (a, b) in self.values.iter().zip(&other.values) {
            delta = delta.max(add_up(a.1, -b.0)?);
        }
        let factor = mass[usize::from(delta >= 0.)];
        let propagated = -multiply_down(-delta, factor)?;
        let travel = add_down(self.travel.0, -other.travel.1)?;
        add_up(propagated, -travel)
    }
    pub(super) fn dominates(&self, other: &Self, mass: [f64; 2]) -> bool {
        self.excess(other, mass).is_some_and(|gap| gap < -1e-8)
    }
}
pub(super) fn prefix_mass(problem: &super::Problem) -> Option<[f64; 2]> {
    let mut minimum: f64 = 1.;
    let mut maximum: f64 = 1.;
    for demand in problem.demands() {
        let mut mass = Range::point(0.);
        for &(_, p) in demand.outcomes() {
            mass = mass.add(Range::point(p))?;
        }
        minimum = minimum.min(mass.0);
        maximum = maximum.max(mass.1);
    }
    let mut product = Range::point(1.);
    for _ in problem.demands() {
        product = Range(
            multiply_down(product.0, minimum)?,
            -multiply_down(-product.1, maximum)?,
        );
    }
    Some([product.0, product.1])
}
pub(super) fn continuation(resource: &RestockingResource, route: &[usize]) -> Option<Continuation> {
    let &last = route.last()?;
    let c = resource.problem.costs();
    let q = resource.problem.capacity();
    let mut work = 10_000_000;
    let mut next = vec![Range::point(c[last][0]); q.checked_add(1)?];
    let mut values = next.clone();
    let mut travel = Range::point(c[last][0]);
    for pair in route.windows(2).rev() {
        let (i, j) = (pair[0], pair[1]);
        travel = travel.add(Range::point(c[i][j]))?;
        let refill = Range::point(c[i][0])
            .add(Range::point(c[0][j]))?
            .add(expectation(resource, j, q, &next, &mut work, None)?)?;
        for (load, value) in values.iter_mut().enumerate() {
            *value = Range::point(c[i][j])
                .add(expectation(resource, j, load, &next, &mut work, None)?)?
                .min(refill);
        }
        std::mem::swap(&mut values, &mut next);
    }
    Some(Continuation {
        values: next,
        travel,
    })
}

/// Standard absolute forward-error allowance for both recourse-vector and
/// canonical common-prefix evaluations. Full Bellman costs are nonnegative and bounded by `upper`.
/// Each customer performs O(outcomes) additions/products per load; the factor
/// 128 covers both evaluations, travel subtraction and interval propagation.
/// This allowance is paid by scalar-cost dominance, never silently ignored.
pub(super) fn dominance_roundoff(resource: &RestockingResource) -> Option<f64> {
    let n = resource.problem.demands().len();
    let upper = objective_bounds(resource, n.checked_add(1)?, &mut 1_000_000, None)?[1];
    let outcomes = resource
        .problem
        .demands()
        .iter()
        .map(|d| d.outcomes().len())
        .max()
        .unwrap_or(0);
    let operations = n
        .checked_add(1)?
        .checked_mul(outcomes.checked_add(1)?)?
        .checked_mul(128)?;
    if operations as u128 > (1u128 << 53) || operations as f64 * f64::EPSILON > 0.25 {
        return None;
    }
    let allowance = -multiply_down(-upper, f64::EPSILON)?;
    let allowance = -multiply_down(-allowance, operations as f64)?;
    // Account for gradual underflow as well as relative rounding.
    add_up(
        allowance,
        -multiply_down(-f64::from_bits(1), operations as f64)?,
    )
}

/// A common prefix with literal probability-mass product w transforms a
/// componentwise recourse difference d and planned-travel difference t into
/// at most w*d + (w-1)*t. Maximize this affine expression at both mass bounds.
/// The uniform forward-error enclosure covers the two stored recourse vectors,
/// their planned sums and both canonical complete-route evaluations. This
/// avoids replaying two full interval Bellman programs per label comparison.
pub(super) fn recourse_excess(
    a: &super::BellmanState,
    b: &super::BellmanState,
    mass: [f64; 2],
    error: f64,
) -> Option<f64> {
    let mut delta = f64::NEG_INFINITY;
    for (&left, &right) in a.values.iter().zip(&b.values) {
        delta = delta.max(add_up(left, -right)?);
    }
    let travel = Range::point(a.suffix_travel).subtract(Range::point(b.suffix_travel))?;
    let full = Range::point(delta).add(travel)?;
    let lower = full.scale(mass[0])?.subtract(travel)?.1;
    let upper = full.scale(mass[1])?.subtract(travel)?.1;
    add_up(lower.max(upper), error)
}
