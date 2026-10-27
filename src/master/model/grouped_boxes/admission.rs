//! Original-column admission with verified intrinsic activity envelopes.
use super::*;
use crate::core::rounding::{add_up, multiply_down};

#[derive(Clone, Copy)]
struct Range {
    lo: f64,
    hi: f64,
}
impl Range {
    fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            lo: add_down(self.lo, other.lo)?,
            hi: add_up(self.hi, other.hi)?,
        })
    }
    fn scale(self, coefficient: f64) -> Option<Self> {
        let (lo, hi) = if coefficient >= 0. {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        Some(Self {
            lo: multiply_down(lo, coefficient)?,
            hi: -multiply_down(-hi, coefficient)?,
        })
    }
}
fn push(rows: &mut BTreeMap<usize, Range>, row: usize, value: Range) -> Option<()> {
    let current = rows.entry(row).or_insert(Range::point(0.));
    *current = current.add(value)?;
    Some(())
}

pub(in crate::master::model) fn coupled<const D: usize>(
    oracle: &Oracle<D>,
    groups: &PreparedBox,
    duals: &[f64],
    offset: usize,
) -> Option<crate::master::pricing::Estimate> {
    if !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
    {
        return None;
    }
    let graph = &oracle.network.graph;
    let hops = graph.max_route_arcs().or_else(|| {
        graph
            .is_acyclic()
            .then(|| graph.vertices.len().saturating_sub(1))
    })?;
    let deadline = Instant::now().checked_add(std::time::Duration::from_secs(1))?;
    let mut work = 250_000usize;
    work = work.checked_sub(hops)?;
    // Preserve the original transition program's objective evaluation order.
    // q * gamma is not interchangeable with its per-arc floating operations.
    let resource_cost = graph
        .coupled_kernel()?
        .objective_walk_lower_bound(graph, hops, &mut work, deadline)?;
    let cheapest = graph.arcs.iter().map(|a| a.cost).fold(0_f64, f64::min);
    let mut lower = 0.;
    for _ in 0..hops {
        lower = add_down(lower, cheapest)?;
    }
    lower = add_down(lower, resource_cost)?;
    let mut rows = BTreeMap::new();
    for &(r, a) in &oracle.route_entries {
        push(&mut rows, r, Range::point(a))?;
    }
    for &(r, a) in oracle.network.route_entries.iter() {
        push(&mut rows, offset.checked_add(r)?, Range::point(a))?;
    }
    // Every row gets at most H * max(entries per arc) physical additions.
    // Padding missing terms with zero and replaying outward additions covers
    // the actual coefficient accumulation, including cancellation/rounding.
    let mut arc_rows = BTreeMap::<usize, (f64, f64, usize)>::new();
    for entries in oracle.network.arc_entries.iter() {
        let mut counts = BTreeMap::<usize, usize>::new();
        for &(r, a) in entries {
            let bounds = arc_rows.entry(r).or_insert((0., 0., 0));
            bounds.0 = bounds.0.min(a);
            bounds.1 = bounds.1.max(a);
            *counts.entry(r).or_default() += 1;
        }
        for (r, count) in counts {
            let current = arc_rows.get_mut(&r)?;
            current.2 = current.2.max(count);
        }
    }
    for (r, (lo, hi, per_arc)) in arc_rows {
        let count = hops.checked_mul(per_arc)?;
        work = work.checked_sub(count)?;
        for _ in 0..count {
            push(&mut rows, offset.checked_add(r)?, Range { lo, hi })?;
        }
    }
    let mut activities = BTreeMap::new();
    for c in &oracle.activity_rows {
        let mut envelope = Range {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        };
        let mut found = false;
        for v in groups
            .variables
            .iter()
            .filter(|v| v.accumulated_resource == Some(c.resource))
        {
            envelope.lo = envelope.lo.max(v.bounds.lower);
            envelope.hi = envelope.hi.min(v.bounds.upper);
            found = true;
        }
        if !found || envelope.lo > envelope.hi {
            return None;
        }
        push(&mut activities, c.row, envelope.scale(c.coefficient)?)?;
    }
    // The activity kernel first aggregates its own terms, then the master
    // adds one value per shared row to the additive coefficients.
    for (r, value) in activities {
        push(&mut rows, offset.checked_add(r)?, value)?;
    }
    for (r, value) in rows {
        lower = add_down(lower, value.scale(-*duals.get(r)?)?.lo)?;
    }
    let _ = work;
    if Instant::now() >= deadline {
        return None;
    }
    crate::master::pricing::Estimate::contribution(
        lower,
        oracle.network.multiplicity.lower,
        oracle.multiplicity_upper(),
    )
}
