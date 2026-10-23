//! Bounded, descending interval propagation for complete physical routes.
//!
//! Every intermediate iterate encloses all feasible source/sink route states.
//! Reaching the work limit weakens pruning; it never proves infeasibility by
//! exhaustion. No finite-walk or acyclicity assumption is introduced here.
use super::bounds::Range;
use super::transitions::NumericOperation;
use super::*;

type Domain = Option<Vec<Range>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RouteDomains {
    topology: Vec<(usize, usize)>,
    source: usize,
    sink: usize,
    vertices: usize,
    pub(super) possible: Vec<bool>,
    pub(super) forward: Vec<Domain>,
    pub(super) backward: Vec<Domain>,
    pub(super) passes: usize,
    pub(super) converged: bool,
}
fn intersect(a: &[Range], b: &[Range]) -> Domain {
    let mut result = Vec::with_capacity(a.len());
    for (a, b) in a.iter().zip(b) {
        let range = Range {
            lo: a.lo.max(b.lo),
            hi: a.hi.min(b.hi),
        };
        if range.lo > range.hi {
            return None;
        }
        result.push(range);
    }
    Some(result)
}
fn include(target: &mut Domain, value: Domain) {
    if let Some(value) = value {
        if let Some(target) = target {
            for (target, value) in target.iter_mut().zip(value) {
                target.lo = target.lo.min(value.lo);
                target.hi = target.hi.max(value.hi);
            }
        } else {
            *target = Some(value);
        }
    }
}
fn constrain(value: Range, bounds: Window) -> Option<Range> {
    let value = Range {
        lo: value.lo.max(bounds.lower),
        hi: value.hi.min(bounds.upper),
    };
    (value.lo <= value.hi).then_some(value)
}
fn image(operations: &[NumericOperation], input: &[Range]) -> Domain {
    let mut values = input.to_vec();
    for operation in operations {
        match *operation {
            NumericOperation::BaseMap {
                slot,
                bounds,
                ref function,
            } => {
                values[slot] = constrain(function.range(values[slot])?, bounds)?;
            }
            NumericOperation::StateCost {
                parent,
                ref function,
            } => {
                values[parent] = constrain(values[parent], function.prepared_domain())?;
            }
            NumericOperation::Cumulative {
                slot, increment, ..
            } => {
                values[slot] = values[slot].add(Range::point(increment));
            }
            NumericOperation::PiecewiseLinear {
                slot,
                parent,
                bounds,
                ref function,
            } => {
                values[parent] = constrain(values[parent], function.prepared_domain())?;
                let delta = function.range(values[parent])?;
                values[slot] = constrain(values[slot].add(delta), bounds)?;
            }
            NumericOperation::Dependent {
                slot,
                parent,
                offset,
                factor,
                bounds,
            } => {
                let delta = Range::point(offset).add(Range::point(factor).multiply(values[parent]));
                values[slot] = constrain(values[slot].add(delta), bounds)?;
            }
        }
    }
    Some(values)
}
// Enclose exact operands of a rounded addition producing a value in this
// interval. A subtraction alone would miss preimages at rounding boundaries.
fn unround(value: Range) -> Range {
    fn down(x: f64) -> f64 {
        if x == f64::NEG_INFINITY {
            return x;
        }
        if x == 0. {
            return -f64::from_bits(1);
        }
        f64::from_bits(if x > 0. {
            x.to_bits() - 1
        } else {
            x.to_bits() + 1
        })
    }
    Range {
        lo: down(value.lo),
        hi: -down(-value.hi),
    }
}
fn preimage(operations: &[NumericOperation], entry: &[Range], output: &[Range]) -> Domain {
    let mut values = intersect(&image(operations, entry)?, output)?;
    for operation in operations.iter().rev() {
        let (slot, delta) = match *operation {
            NumericOperation::BaseMap {
                slot, ref function, ..
            } => {
                values[slot] = function.preimage(entry[slot], values[slot])?;
                continue;
            }
            NumericOperation::StateCost {
                parent,
                ref function,
            } => {
                values[parent] = constrain(values[parent], function.prepared_domain())?;
                continue;
            }
            NumericOperation::Cumulative {
                slot, increment, ..
            } => (slot, Range::point(increment)),
            NumericOperation::PiecewiseLinear {
                slot,
                parent,
                ref function,
                ..
            } => {
                values[parent] = constrain(values[parent], function.prepared_domain())?;
                (slot, function.range(values[parent])?)
            }
            NumericOperation::Dependent {
                slot,
                parent,
                offset,
                factor,
                ..
            } => (
                slot,
                Range::point(offset).add(Range::point(factor).multiply(values[parent])),
            ),
        };
        let value = unround(values[slot]).subtract(delta);
        values[slot] = Range {
            lo: value.lo.max(entry[slot].lo),
            hi: value.hi.min(entry[slot].hi),
        };
        if values[slot].lo > values[slot].hi {
            return None;
        }
    }
    Some(values)
}
impl RouteDomains {
    pub(super) fn prepare<const D: usize>(
        g: &Graph<D>,
        transitions: &PreparedTransitions,
        baseline: &[Range],
        initial: &[f64],
    ) -> Self {
        let count = g.vertices.len();
        let mut result = Self {
            topology: g.arcs.iter().map(|arc| (arc.tail, arc.head)).collect(),
            source: g.source,
            sink: g.sink,
            vertices: count,
            possible: transitions.arcs.iter().map(|arc| !arc.impossible).collect(),
            forward: vec![Some(baseline.to_vec()); count],
            backward: vec![Some(baseline.to_vec()); count],
            passes: 0,
            converged: false,
        };
        // Graph validation owns malformed topology. Preparation must not panic
        // when called before that validation, or infer pruning from it.
        if g.source >= count
            || g.sink >= count
            || g.arcs
                .iter()
                .any(|arc| arc.tail >= count || arc.head >= count)
        {
            return result;
        }
        let root: Vec<_> = initial.iter().copied().map(Range::point).collect();
        for _ in 0..count.clamp(1, 64) {
            let mut forward = vec![None; count];
            forward[g.source] = Some(root.clone());
            for (index, arc) in g.arcs.iter().enumerate() {
                if !result.possible[index] {
                    continue;
                }
                if let Some(entry) = &result.forward[arc.tail] {
                    include(
                        &mut forward[arc.head],
                        image(&transitions.arcs[index].operations, entry),
                    );
                }
            }
            for (vertex, value) in forward.iter_mut().enumerate() {
                *value = match (
                    value.as_ref(),
                    result.forward[vertex].as_ref(),
                    result.backward[vertex].as_ref(),
                ) {
                    (Some(value), Some(old), Some(backward)) => {
                        intersect(value, old).and_then(|value| intersect(&value, backward))
                    }
                    _ => None,
                };
            }
            let mut backward = vec![None; count];
            backward[g.sink] = forward[g.sink].clone();
            for (index, arc) in g.arcs.iter().enumerate() {
                if !result.possible[index] {
                    continue;
                }
                if let (Some(entry), Some(exit)) = (&forward[arc.tail], &result.backward[arc.head])
                {
                    include(
                        &mut backward[arc.tail],
                        preimage(&transitions.arcs[index].operations, entry, exit),
                    );
                }
            }
            for (vertex, value) in backward.iter_mut().enumerate() {
                *value = match (
                    value.as_ref(),
                    result.backward[vertex].as_ref(),
                    forward[vertex].as_ref(),
                ) {
                    (Some(value), Some(old), Some(forward)) => {
                        intersect(value, old).and_then(|value| intersect(&value, forward))
                    }
                    _ => None,
                };
            }
            let unchanged = forward == result.forward && backward == result.backward;
            result.forward = forward;
            result.backward = backward;
            result.passes += 1;
            let mut removed = false;
            for (index, arc) in g.arcs.iter().enumerate() {
                if !result.possible[index] {
                    continue;
                }
                let possible = match (&result.forward[arc.tail], &result.backward[arc.head]) {
                    (Some(entry), Some(exit)) => {
                        preimage(&transitions.arcs[index].operations, entry, exit).is_some()
                    }
                    _ => false,
                };
                if !possible {
                    result.possible[index] = false;
                    removed = true;
                }
            }
            if unchanged && !removed {
                result.converged = true;
                break;
            }
        }
        result
    }
    pub(super) fn matches<const D: usize>(&self, g: &Graph<D>) -> bool {
        self.source == g.source
            && self.sink == g.sink
            && self.vertices == g.vertices.len()
            && self.topology.len() == g.arcs.len()
            && self
                .topology
                .iter()
                .zip(&g.arcs)
                .all(|(&(tail, head), arc)| tail == arc.tail && head == arc.head)
    }
}
