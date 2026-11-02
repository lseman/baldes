//! Component child rows and the ancestor/sibling rows used for duplicate skipping.
use super::*;
use crate::master::preprocessing::Bounds;

#[derive(Clone, Debug)]
pub(in crate::master::model) struct BranchRow {
    pub network: i32,
    pub sequence: ComponentSequence,
    pub lower: bool,
    pub rhs: f64,
    // Explicit registrations may reuse an ID for a different expression.
    // Keep only coordinates used by this group's original sequence.
    coordinates: Arc<Vec<ComponentCoordinate>>,
}

fn cardinality_equal(a: f64, b: f64) -> bool {
    let band = 1e-6 * (1. + a.abs().max(b.abs()));
    a >= b - band && b >= a - band
}

impl BranchRow {
    fn equivalent(&self, other: &Self) -> bool {
        self.network == other.network
            && self.lower == other.lower
            && cardinality_equal(self.rhs, other.rhs)
            && self.sequence.bounds.len() == other.sequence.bounds.len()
            && self
                .sequence
                .bounds
                .iter()
                .zip(&other.sequence.bounds)
                .all(|(a, b)| {
                    let band = 1e-6 + a.threshold.abs().max(b.threshold.abs()) * 1e-10;
                    let same_mapping = self
                        .coordinates
                        .iter()
                        .find(|c| c.id == a.coordinate)
                        .zip(other.coordinates.iter().find(|c| c.id == b.coordinate))
                        .is_some_and(|(a, b)| a.arc_values == b.arc_values);
                    a.coordinate == b.coordinate
                        && same_mapping
                        && a.sense == b.sense
                        && a.threshold >= b.threshold - band
                        && b.threshold >= a.threshold - band
                })
    }
}

fn incompatible(sequence: &ComponentSequence) -> bool {
    let mut ranges = BTreeMap::<usize, (f64, f64)>::new();
    for bound in &sequence.bounds {
        let range = ranges
            .entry(bound.coordinate)
            .or_insert((f64::NEG_INFINITY, f64::INFINITY));
        match bound.sense {
            ComponentSense::GreaterEqual => range.0 = range.0.max(bound.threshold),
            ComponentSense::LessEqual => range.1 = range.1.min(bound.threshold),
        }
        if range.0.is_finite()
            && range.1.is_finite()
            && range.0 > range.1 + (1e-6 + range.0.abs().max(range.1.abs()) * 1e-10)
        {
            return true;
        }
    }
    false
}

pub(in crate::master::model) fn record_count_bounds<const D: usize>(
    node: &mut NodeSnapshot<D>,
    network: i32,
    bounds: Bounds,
) {
    let retained = node
        .state
        .component_count_bounds
        .entry(network)
        .or_insert(Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        });
    retained.lower = retained.lower.max(bounds.lower);
    retained.upper = retained.upper.min(bounds.upper);
}

/// The current node itself is excluded by the native ancestor-child scan. Its
/// local row enters the exclusions only when descending to the next generation.
pub(in crate::master::model) fn inherit_branch_rows<const D: usize>(
    parent: &NodeSnapshot<D>,
    children: &mut [NodeSnapshot<D>],
) {
    for child in children.iter_mut() {
        if Arc::ptr_eq(
            &child.state.component_local_rows,
            &parent.state.component_local_rows,
        ) {
            child.state.component_local_rows = Arc::new(vec![]);
        }
    }
    let locals: Vec<_> = children
        .iter()
        .map(|child| child.state.component_local_rows.clone())
        .collect();
    for (side, child) in children.iter_mut().enumerate() {
        let mut rows = parent.state.component_branch_exclusions.as_ref().clone();
        if parent.state.node_depth > 0 {
            rows.extend(parent.state.component_local_rows.iter().cloned());
        }
        for (other, local) in locals.iter().enumerate() {
            if other != side {
                rows.extend(local.iter().cloned());
            }
        }
        child.state.component_branch_exclusions = Arc::new(rows);
    }
}

impl ComponentClassBranching {
    fn snapshot_activity<const D: usize>(&self, node: &NodeSnapshot<D>) -> Result<f64> {
        let routes = node
            .warm
            .columns
            .iter()
            .zip(&node.master.values)
            .filter_map(|(metadata, &quantity)| {
                let metadata = metadata.as_ref()?;
                (metadata.tag >= 0)
                    .then(|| (node.state.routes[metadata.tag as usize].as_ref(), quantity))
            })
            .chain(
                node.state
                    .fixed_routes
                    .iter()
                    .map(|fixed| (&fixed.route, fixed.quantity)),
            );
        self.activity(node, routes)
    }

    fn count_bounds<const D: usize>(&self, node: &NodeSnapshot<D>) -> Result<Bounds> {
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
            .ok_or_else(|| invalid("component class network"))?;
        let fixed = node
            .state
            .fixed_routes
            .iter()
            .filter(|r| r.route.network == self.network)
            .map(|r| r.quantity)
            .sum::<f64>();
        let mut bounds = Bounds {
            lower: oracle.network.multiplicity.lower + fixed,
            upper: oracle.network.multiplicity.upper + fixed,
        };
        if let Some(retained) = node.state.component_count_bounds.get(&self.network) {
            bounds.lower = bounds.lower.max(retained.lower);
            bounds.upper = bounds.upper.min(retained.upper);
        }
        Ok(bounds)
    }

    pub(in crate::master::model) fn branching_children<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
        value: f64,
    ) -> Result<Vec<NodeSnapshot<D>>> {
        let coordinates = Arc::new(
            self.coordinates
                .iter()
                .filter(|coordinate| {
                    self.sequence
                        .bounds
                        .iter()
                        .any(|bound| bound.coordinate == coordinate.id)
                })
                .cloned()
                .collect::<Vec<_>>(),
        );
        let binary = || -> Result<Vec<(Self, BranchRow)>> {
            let (lower, upper) = covering_split_bounds(value, !self.sequence.bounds.is_empty())
                .ok_or_else(|| invalid("component split activity"))?;
            Ok([(false, lower), (true, upper)]
                .into_iter()
                .map(|(lower, rhs)| {
                    (
                        self.clone(),
                        BranchRow {
                            network: self.network,
                            coordinates: coordinates.clone(),
                            sequence: self.sequence.clone(),
                            lower,
                            rhs,
                        },
                    )
                })
                .collect())
        };
        let integer_partition = self.sequence.bounds.iter().all(|bound| {
            let sibling = match bound.sense {
                ComponentSense::GreaterEqual => bound.threshold - 1.,
                ComponentSense::LessEqual => bound.threshold + 1.,
            };
            bound.threshold.fract() == 0.
                && sibling.is_finite()
                && (sibling - bound.threshold).abs() == 1.
        });
        let mut rows = if self.sequence.bounds.is_empty() || !integer_partition {
            binary()?
        } else {
            let count = self.count_bounds(node)?;
            let fixed = count.lower.is_finite()
                && count.upper.is_finite()
                && cardinality_equal(count.lower, count.upper);
            let mut class = self.clone();
            let mut rows = vec![];
            let mut capacity = 0.;
            let full_upper = split_bounds(value, true)
                .ok_or_else(|| invalid("component split activity"))?
                .1;
            rows.push((
                class.clone(),
                BranchRow {
                    network: self.network,
                    coordinates: coordinates.clone(),
                    sequence: class.sequence.clone(),
                    lower: true,
                    rhs: full_upper,
                },
            ));
            capacity += full_upper - 1.;
            while !class.sequence.bounds.is_empty() {
                class.sequence = class.sequence.sibling()?;
                let activity = class.snapshot_activity(node)?;
                let upper = split_bounds(activity, true)
                    .ok_or_else(|| invalid("component complement activity"))?
                    .1;
                rows.push((
                    class.clone(),
                    BranchRow {
                        network: self.network,
                        coordinates: coordinates.clone(),
                        sequence: class.sequence.clone(),
                        lower: true,
                        rhs: upper,
                    },
                ));
                capacity += upper - 1.;
                class.sequence.bounds.pop();
            }
            let total = class.snapshot_activity(node)?;
            let lower = split_bounds(total, false)
                .ok_or_else(|| invalid("component total activity"))?
                .0;
            // Relative rounding can move endpoints at very large magnitudes.
            // Retain a valid binary cover when the multiway integer cover fails.
            if !capacity.is_finite()
                || if fixed {
                    count.lower <= capacity
                } else {
                    lower < capacity
                }
            {
                binary()?
            } else {
                if !fixed {
                    rows.push((
                        class.clone(),
                        BranchRow {
                            network: self.network,
                            coordinates: coordinates.clone(),
                            sequence: class.sequence.clone(),
                            lower: false,
                            rhs: lower,
                        },
                    ));
                }
                rows
            }
        };
        rows.retain(|(_, row)| {
            !incompatible(&row.sequence)
                && !node
                    .state
                    .component_branch_exclusions
                    .iter()
                    .any(|existing| row.equivalent(existing))
        });
        rows.into_iter()
            .map(|(class, row)| {
                let bounds = if row.lower {
                    Bounds {
                        lower: row.rhs,
                        upper: f64::INFINITY,
                    }
                } else {
                    Bounds {
                        lower: f64::NEG_INFINITY,
                        upper: row.rhs,
                    }
                };
                let mut child = node.with_component_bounds(&class, bounds)?;
                child.state.component_local_rows = Arc::new(vec![row]);
                Ok(child)
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/component_branching_tests.rs"]
mod tests;
