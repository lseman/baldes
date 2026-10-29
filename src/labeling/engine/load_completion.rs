//! Capacity-indexed shortest-walk bounds for additive pricing objectives.
use super::*;
use crate::core::rounding::{add_down, add_up, multiply_down};

const LOAD_LEVELS: usize = 256;

#[derive(Clone, Debug)]
pub(super) struct LoadCompletion {
    resource: usize,
    capacity: f64,
    scale: f64,
    width: usize,
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}

impl LoadCompletion {
    pub(super) fn prepare<const D: usize>(
        g: &Graph<D>,
        resource: usize,
        bidirectional: bool,
        started: Instant,
        limit: std::time::Duration,
    ) -> Option<Self> {
        if g.resources.get(resource) != Some(&ResourceKind::Disposable)
            || g.has_coupled_resources()
            || g.vertices.iter().any(|v| {
                let b = v.bounds[resource];
                b.lower < 0. || !b.lower.is_finite() || !b.upper.is_finite()
            })
            || g.arcs.iter().any(|a| {
                a.consumption[resource] < 0.
                    || !a.consumption[resource].is_finite()
                    || !a.reduced_cost.is_finite()
            })
        {
            return None;
        }
        let capacity = g
            .vertices
            .iter()
            .map(|v| v.bounds[resource].upper)
            .fold(0., f64::max);
        if capacity <= 0. {
            return None;
        }
        let scale = LOAD_LEVELS as f64 / capacity;
        if !scale.is_finite() {
            return None;
        }
        let edges: Vec<_> = g
            .arcs
            .iter()
            .map(|a| {
                Some((
                    a.tail,
                    a.head,
                    multiply_down(a.consumption[resource], scale)?
                        .floor()
                        .max(0.) as usize,
                    a.reduced_cost,
                ))
            })
            .collect::<Option<_>>()?;
        let width = g.vertices.len();
        let suffix = Self::table(width, g.sink, &edges, false, started, limit)?;
        let prefix = if bidirectional {
            Self::table(width, g.source, &edges, true, started, limit)?
        } else {
            vec![]
        };
        Some(Self {
            resource,
            capacity,
            scale,
            width,
            prefix,
            suffix,
        })
    }

    fn table(
        width: usize,
        terminal: usize,
        edges: &[(usize, usize, usize, f64)],
        reverse: bool,
        started: Instant,
        limit: std::time::Duration,
    ) -> Option<Vec<f64>> {
        // One extra capacity cell encloses upward-rounded query budgets.
        let mut values = vec![f64::INFINITY; (LOAD_LEVELS + 2) * width];
        for k in 0..LOAD_LEVELS + 2 {
            if started.elapsed() >= limit {
                return None;
            }
            let (previous, layer) = values.split_at_mut(k * width);
            let layer = &mut layer[..width];
            if k > 0 {
                layer.copy_from_slice(&previous[(k - 1) * width..]);
            }
            layer[terminal] = layer[terminal].min(0.);
            for &(a, b, units, cost) in edges {
                if units == 0 || units > k {
                    continue;
                }
                let (from, to) = if reverse { (b, a) } else { (a, b) };
                let rest = previous[(k - units) * width + to];
                if rest.is_finite() {
                    layer[from] = layer[from].min(add_down(cost, rest)?);
                }
            }
            // Zero-demand/rounded-zero arcs form a same-capacity subproblem.
            // A reachable negative cycle makes this walk relaxation unbounded;
            // declining the table leaves ordinary pricing intact.
            for pass in 0..width {
                let mut changed = false;
                for &(a, b, units, cost) in edges {
                    if units != 0 {
                        continue;
                    }
                    let (from, to) = if reverse { (b, a) } else { (a, b) };
                    if layer[to].is_finite() {
                        let candidate = add_down(cost, layer[to])?;
                        if candidate < layer[from] {
                            layer[from] = candidate;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
                if pass + 1 == width {
                    return None;
                }
            }
        }
        Some(values)
    }

    pub(super) fn rejects<const D: usize>(
        &self,
        label: &StoredLabel<D>,
        forward: bool,
        threshold: f64,
    ) -> bool {
        let q = label.resources[self.resource];
        // Relax every intermediate window and all NG/branch memories. Never
        // exclude a previously visited customer from this walk relaxation.
        let room = if forward {
            add_up(self.capacity, -q)
        } else {
            Some(q)
        };
        let Some(room) = room else {
            return false;
        };
        if !room.is_finite() || room < 0. || room > self.capacity {
            return false;
        }
        let Some(scaled) = add_up(room * self.scale, 1.) else {
            return false;
        };
        let level = scaled.ceil().max(0.) as usize;
        if level >= LOAD_LEVELS + 2 {
            return false;
        }
        let table = if forward { &self.suffix } else { &self.prefix };
        let rest = table[level * self.width + label.vertex];
        if rest == f64::INFINITY {
            return true;
        }
        let Some(partial) = add_down(label.cost, label.additional_cost) else {
            return false;
        };
        let Some(sum) = add_down(partial, rest) else {
            return false;
        };
        // Scalar labeling uses tolerance-based resource arithmetic. Keep a
        // numerical margin in addition to downward-rounded walk additions.
        sum - 1e-8
            * (1. + label.cost.abs() + label.additional_cost.abs() + rest.abs() + threshold.abs())
            >= threshold
    }
}

impl<const D: usize> LabelingEngine<D> {
    pub(super) fn prepare_load_completion(
        &self,
        g: &Graph<D>,
        started: Instant,
        limit: std::time::Duration,
    ) -> Option<LoadCompletion> {
        let resource = self.controls.load_completion_resource?;
        // Symmetric execution uses a transformed arc objective. Unsupported
        // nonadditive costs must not be silently dropped from a lower bound.
        if !self.phase.exact()
            || !self.controls.load_completion_threshold.is_finite()
            || self.consumption_joins
            || self.cuts.presence.active()
            || self.cuts.knapsack.active
            || !self.cuts.kpath.is_empty()
            || self.cuts.inputs.iter().any(|c| {
                c.covering
                    || c.type2
                    || !c.dual.is_finite()
                    || c.dual < 0.
                    || c.denominator <= 0
                    || c.vertex_coefficient.iter().any(|&v| v < 0)
                    || c.arc_coefficient.values().any(|&v| v < 0)
            })
        {
            return None;
        }
        // Packing rank-one increments are nonnegative. Keep the penalties
        // already paid by the label and relax all future increments to zero.
        LoadCompletion::prepare(g, resource, self.runtime_bidirectional(), started, limit)
    }
}
