//! Shared finite-horizon signed order and exact numeric-prefix compensation.
//! No mixed physical/front key is consumed.
use super::*;
use crate::core::rounding::multiply_down;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct BoundedOrder {
    steps: usize,
    slots: Vec<(f64, f64, i8)>,
    completion_weights: Vec<(f64, f64)>,
    integral: bool,
}

fn multiply_up(a: f64, b: f64) -> Option<f64> {
    Some(-multiply_down(-a, b)?)
}

impl BoundedOrder {
    pub(super) fn scoped<const D: usize>(
        &self,
        graph: &Graph<D>,
        definitions: &[CoupledResource],
    ) -> Option<Vec<Self>> {
        let topology = graph.topological_order()?;
        let mut outgoing = vec![Vec::new(); graph.vertices.len()];
        for (position, arc) in graph.arcs.iter().enumerate() {
            outgoing[arc.tail].push((position, arc.head));
        }
        let mut depth = vec![0usize; graph.vertices.len()];
        // Maximum absolute increment and both weight signs among reachable
        // arcs. Prefix-only arcs never enter a descendant's summary.
        let mut summaries = vec![vec![(0f64, 0f64, 0f64); self.slots.len()]; graph.vertices.len()];
        for &vertex in topology.iter().rev() {
            for &(arc, head) in &outgoing[vertex] {
                depth[vertex] = depth[vertex].max(depth[head].checked_add(1)?);
                for (resource, definition) in definitions.iter().enumerate() {
                    let (increments, weights) = match definition {
                        CoupledResource::CustomResource {
                            increments,
                            weights,
                            ..
                        }
                        | CoupledResource::CumulativeCost {
                            increments,
                            weights,
                            ..
                        } => (increments, weights),
                        _ => return None,
                    };
                    let child = summaries[head][resource];
                    let current = &mut summaries[vertex][resource];
                    current.0 = current.0.max(increments[arc].abs()).max(child.0);
                    current.1 = current.1.min(weights[arc]).min(child.1);
                    current.2 = current.2.max(weights[arc]).max(child.2);
                }
            }
        }
        let mut profiles = Vec::with_capacity(graph.vertices.len());
        for (vertex, summary) in summaries.iter().enumerate() {
            let mut profile = self.clone();
            profile.steps = self.steps.min(depth[vertex]);
            for (resource, &(increment, lo, hi)) in summary.iter().enumerate() {
                let direction = if lo >= 0. {
                    1
                } else if hi <= 0. {
                    -1
                } else {
                    0
                };
                profile.slots[resource] = (increment, lo.abs().max(hi.abs()), direction);
                profile.completion_weights[resource] = (
                    self.completion_weights[resource]
                        .0
                        .max(-multiply_up(-lo, profile.steps as f64)?),
                    self.completion_weights[resource]
                        .1
                        .min(multiply_up(hi, profile.steps as f64)?),
                );
            }
            profiles.push(profile);
        }
        Some(profiles)
    }

    pub(super) fn new(steps: usize, definitions: &[CoupledResource]) -> Option<Self> {
        let mut slots = Vec::new();
        for resource in definitions {
            let (increments, weights) = match resource {
                CoupledResource::CustomResource {
                    increments,
                    weights,
                    ..
                }
                | CoupledResource::CumulativeCost {
                    increments,
                    weights,
                    ..
                } => (increments, weights),
                _ => return None,
            };
            let direction = if weights.iter().all(|&w| w >= 0.) {
                1
            } else if weights.iter().all(|&w| w <= 0.) {
                -1
            } else {
                0
            };
            slots.push((
                increments.iter().map(|v| v.abs()).fold(0., f64::max),
                weights.iter().map(|v| v.abs()).fold(0., f64::max),
                direction,
            ));
        }
        if steps > (1u64 << 53) as usize {
            return None;
        }
        let mut completion_weights = Vec::new();
        let mut integral = true;
        for resource in definitions {
            let (increments, weights, upper) = match resource {
                CoupledResource::CustomResource {
                    increments,
                    weights,
                    forward_bound,
                    ..
                } => (increments, weights, forward_bound),
                CoupledResource::CumulativeCost {
                    increments,
                    weights,
                    sink_upper,
                    ..
                } => (increments, weights, sink_upper),
                _ => return None,
            };
            integral &= increments.iter().chain(weights).all(|v| v.fract() == 0.);
            let lower = weights.iter().copied().fold(0., f64::min);
            let higher = weights.iter().copied().fold(0., f64::max);
            completion_weights.push((
                -multiply_up(-lower, steps as f64)?,
                upper.unwrap_or(multiply_up(higher, steps as f64)?),
            ));
        }
        Some(Self {
            steps,
            slots,
            completion_weights,
            integral,
        })
    }

    pub(super) fn preserves(&self, a: &[f64], b: &[f64], ac: f64, bc: f64) -> bool {
        if a.len() != self.slots.len() || b.len() != self.slots.len() {
            return false;
        }
        let proof = || -> Option<()> {
            let mut charge = 0.;
            for ((&a, &b), &(increment, weight, direction)) in a.iter().zip(b).zip(&self.slots) {
                if !a.is_finite()
                    || !b.is_finite()
                    || match direction {
                        1 => a > b,
                        -1 => a < b,
                        _ => a != b,
                    }
                {
                    return None;
                }
                // Include one upward rounding allowance per state addition.
                // Repeated additions have at most steps roundings. The factor
                // two bounds their growth for realistic addressable DAGs only
                // when the product of steps and epsilon is at most one half.
                if self.steps as f64 * f64::EPSILON > 0.5 {
                    return None;
                }
                let growth = multiply_up(self.steps as f64, increment)?;
                let state = multiply_up(add_up(a.abs().max(b.abs()), growth)?, 2.)?;
                charge = add_up(charge, multiply_up(state, weight)?)?;
            }
            // Bound every rounded prefix accumulation, including additions
            // between multiple descriptors on the same arc. A factor two is
            // valid only under the same operation-count rounding guard.
            let operations = self.steps.checked_mul(self.slots.len().checked_add(1)?)?;
            if operations as f64 * f64::EPSILON > 0.5 {
                return None;
            }
            let total = add_up(
                ac.abs().max(bc.abs()),
                multiply_up(self.steps as f64, charge)?,
            )?;
            let total = multiply_up(total, 2.)?;
            total.is_finite().then_some(())
        };
        proof().is_some() || self.exact_compensation(a, b, ac, bc).is_some()
    }

    // Integral arithmetic is exact only while every intermediate state,
    // product and partial cost sum stays within binary64's integer lattice.
    // Independent absolute envelopes establish that before using the shared
    // continuation's weight interval to charge the numeric prefix gap.
    fn exact_compensation(&self, a: &[f64], b: &[f64], ac: f64, bc: f64) -> Option<()> {
        const LIMIT: f64 = (1u64 << 53) as f64;
        if !self.integral || ac.fract() != 0. || bc.fract() != 0. {
            return None;
        }
        let mut charge = 0.;
        let mut penalty = 0.;
        for (((&a, &b), &(increment, weight, _)), &(lo, hi)) in a
            .iter()
            .zip(b)
            .zip(&self.slots)
            .zip(&self.completion_weights)
        {
            if !a.is_finite() || !b.is_finite() || a.fract() != 0. || b.fract() != 0. {
                return None;
            }
            let state = add_up(
                a.abs().max(b.abs()),
                multiply_up(self.steps as f64, increment)?,
            )?;
            if state > LIMIT {
                return None;
            }
            charge = add_up(charge, multiply_up(state, weight)?)?;
            if charge > LIMIT {
                return None;
            }
            // Exact common updates retain the original state difference.
            // Include the empty completion and do not spend physical cost.
            let extra = Range::point(a)
                .subtract(Range::point(b))
                .multiply(Range { lo, hi })
                .hi
                .max(0.);
            penalty = add_up(penalty, extra)?;
        }
        let total = add_up(
            ac.abs().max(bc.abs()),
            multiply_up(self.steps as f64, charge)?,
        )?;
        (total <= LIMIT && add_up(ac, penalty)? <= bc).then_some(())
    }
}
