//! Sufficient forward dependency orders for every continuation in the domain.
use super::bounds::Range;
use super::*;

const LESS: u8 = 1;
const GREATER: u8 = 2;

#[derive(Clone, Debug, PartialEq)]
struct Dependency {
    slot: usize,
    parent: usize,
    factors: u8,
    lower_safe: bool,
    upper_safe: bool,
    finite: bool,
    parent_domain_safe: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DependencyOrder {
    dependencies: Vec<Dependency>,
    numeric_orders: Vec<[bool; 2]>,
    required_equal: Vec<bool>,
}
impl DependencyOrder {
    pub(super) fn prepare(
        definitions: &[CoupledResource],
        slots: &[Option<usize>],
        domains: &[Range],
    ) -> Self {
        let mut required_equal = vec![false; domains.len()];
        for (r, definition) in definitions.iter().enumerate() {
            match definition {
                CoupledResource::PiecewiseLinearResource { .. }
                | CoupledResource::EncodedPiecewiseLinearResource { .. } => {
                    required_equal[slots[r].unwrap()] = true
                }
                CoupledResource::PiecewiseLinearCost { parent, .. }
                | CoupledResource::EncodedPiecewiseLinearCost { parent, .. } => {
                    required_equal[slots[*parent].unwrap()] = true
                }
                CoupledResource::EncodedPiecewiseLinearDependent { parent, .. } => {
                    required_equal[slots[r].unwrap()] = true;
                    required_equal[slots[*parent].unwrap()] = true;
                }
                _ => {}
            }
        }
        let mut numeric_orders = vec![[true; 2]; domains.len()];
        for (r, definition) in definitions.iter().enumerate() {
            if let CoupledResource::CumulativeCost { increments, .. }
            | CoupledResource::CustomResource { increments, .. } = definition
            {
                let slot = slots[r].unwrap();
                let bounded = domains[slot].finite()
                    && increments
                        .iter()
                        .all(|&delta| domains[slot].add(Range::point(delta)).finite());
                // For unbounded domains, only the comparison direction that
                // cannot overflow away from a finite comparator is safe.
                numeric_orders[slot] = [
                    bounded || increments.iter().all(|&delta| delta >= 0.),
                    bounded || increments.iter().all(|&delta| delta <= 0.),
                ];
            }
        }
        let dependencies = definitions
            .iter()
            .enumerate()
            .filter_map(|(r, definition)| {
                if let CoupledResource::PiecewiseLinearDependent {
                    parent,
                    bounds,
                    functions,
                    ..
                } = definition
                {
                    let slot = slots[r].unwrap();
                    let parent = slots[*parent].unwrap();
                    let mut direction = 0;
                    let (mut lower_safe, mut upper_safe, mut finite) = (true, true, true);
                    let mut parent_domain_safe = true;
                    for function in functions {
                        for pair in function.points.windows(2) {
                            direction |= if pair[1][1] > pair[0][1] {
                                GREATER
                            } else if pair[1][1] < pair[0][1] {
                                LESS
                            } else {
                                0
                            };
                        }
                        let domain = function.prepared_domain();
                        parent_domain_safe &= domains[parent].lo >= domain.lower
                            && domains[parent].hi <= domain.upper;
                        let delta = function
                            .range(domains[parent])
                            .unwrap_or_else(Range::unknown);
                        finite &= domains[parent].lo >= domain.lower
                            && domains[parent].hi <= domain.upper
                            && delta.finite()
                            && domains[slot].add(delta).finite();
                        lower_safe &= delta.lo >= 0.;
                        upper_safe &= delta.hi <= 0.;
                    }
                    return Some(Dependency {
                        slot,
                        parent,
                        factors: direction,
                        lower_safe: bounds.lower == f64::NEG_INFINITY || lower_safe,
                        upper_safe: bounds.upper == f64::INFINITY || upper_safe,
                        finite,
                        parent_domain_safe,
                    });
                }
                let CoupledResource::Dependent {
                    parent,
                    bounds,
                    offsets,
                    factors,
                    ..
                } = definition
                else {
                    return None;
                };
                let slot = slots[r].unwrap();
                let parent = slots[*parent].unwrap();
                let mut direction = 0;
                let mut lower_safe = true;
                let mut upper_safe = true;
                let mut finite = true;
                for (&offset, &factor) in offsets.iter().zip(factors) {
                    direction |= if factor > 0. {
                        GREATER
                    } else if factor < 0. {
                        LESS
                    } else {
                        0
                    };
                    // Enclose each literal operation separately. Apart from proving
                    // increment signs, require all intermediates to stay finite so
                    // deletion cannot conceal an overflow unique to the dominator.
                    let product = Range::point(factor).multiply(domains[parent]);
                    let delta = Range::point(offset).add(product);
                    let image = domains[slot].add(delta);
                    finite &= product.finite() && delta.finite() && image.finite();
                    lower_safe &= delta.lo >= 0.;
                    upper_safe &= delta.hi <= 0.;
                }
                Some(Dependency {
                    slot,
                    parent,
                    factors: direction,
                    lower_safe: bounds.lower == f64::NEG_INFINITY || lower_safe,
                    upper_safe: bounds.upper == f64::INFINITY || upper_safe,
                    finite,
                    parent_domain_safe: true,
                })
            })
            .collect();
        Self {
            dependencies,
            numeric_orders,
            required_equal,
        }
    }

    pub(super) fn preserves(&self, a: &[f64], b: &[f64], domains: &[Range]) -> bool {
        if a.len() != b.len() || a.len() != domains.len() {
            return false;
        }
        if a == b
            || (self.dependencies.is_empty() && !self.required_equal.iter().any(|&equal| equal))
        {
            return true;
        }
        // Bounds used in preparation must actually contain both entry states.
        // Check this even in phase I; feasibility is independent of cost scale.
        let mut directions = StateValues::<u8, 8>::filled(a.len(), 0);
        for (slot, ((&a, &b), domain)) in a.iter().zip(b).zip(domains).enumerate() {
            if !a.is_finite()
                || !b.is_finite()
                || a < domain.lo
                || a > domain.hi
                || b < domain.lo
                || b > domain.hi
            {
                return false;
            }
            if (a < b && !self.numeric_orders[slot][0]) || (a > b && !self.numeric_orders[slot][1])
            {
                return false;
            }
            directions[slot] = if a < b {
                LESS
            } else if a > b {
                GREATER
            } else {
                0
            };
        }
        // Earlier-parent validation gives a DAG in declaration order. The
        // union includes the child's retained difference and all signs its
        // already-closed parent can induce. It is invariant under any number
        // and ordering of physical arcs, including rounded monotone updates.
        for dependency in &self.dependencies {
            let parent = directions[dependency.parent];
            // A constant function can hide every value difference while its
            // argument domain still distinguishes the two routes.
            if parent != 0 && !dependency.parent_domain_safe {
                return false;
            }
            let mut direction = directions[dependency.slot];
            if dependency.factors & GREATER != 0 {
                direction |= parent;
            }
            if dependency.factors & LESS != 0 {
                direction |= ((parent & LESS) << 1) | ((parent & GREATER) >> 1);
            }
            if direction != 0
                && (!dependency.finite
                    || (direction & LESS != 0 && !dependency.lower_safe)
                    || (direction & GREATER != 0 && !dependency.upper_safe))
            {
                return false;
            }
            directions[dependency.slot] = direction;
        }
        self.required_equal
            .iter()
            .zip(directions.iter().copied())
            .all(|(&equal, direction)| !equal || direction == 0)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/dependency_order/tests.rs"]
mod tests;
