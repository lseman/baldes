//! Distinct aggregate admission and full row activity in registration order.
use super::{activity_rows, NetworkRoute, OriginalPricingVariable, Output};
use std::collections::HashMap;

mod bounds;
mod quantity;

pub(super) fn split_bounds(value: f64) -> Option<(f64, f64)> {
    bounds::covering_split_bounds(value)
}

pub(super) fn fractional_distances(value: f64) -> Option<(f64, f64)> {
    bounds::fractional_distances(value)
}

/// Retained original-variable coefficients for aggregate candidate admission.
///
/// Registration order is preserved independently of canonical pricing/reporting
/// order. This projection covers generated master columns in strong branching;
/// ordinary selection has no generated-original input. Child expressions retain
/// their complete membership separately. Nonpaired integral activities use their
/// stored-column activity replay; paired associations are outside this contract.
#[derive(Clone, Debug)]
pub struct AggregateProjection {
    members: std::sync::Arc<[AggregateMember]>,
}

#[derive(Debug)]
struct AggregateMember {
    network: i32,
    integer: bool,
    accumulated_resource: Option<usize>,
    associations: HashMap<i32, f64>,
}

impl AggregateProjection {
    pub(crate) fn new(members: Vec<OriginalPricingVariable>) -> Self {
        // Last setter wins. Lookups never iterate these maps, so their hash
        // layout cannot change the original-member or arc encounter order.
        Self {
            members: members
                .into_iter()
                .map(|member| AggregateMember {
                    network: member.network,
                    integer: member.integer,
                    accumulated_resource: member.accumulated_resource,
                    associations: member.arcs.into_iter().collect(),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub(crate) fn value<const D: usize>(&self, output: &Output<D>, strong: bool) -> Option<f64> {
        if !strong {
            return Some(0.);
        }
        let mut group = 0.;
        for member in self.members.iter().filter(|member| member.integer) {
            let mut original = 0.;
            for (&tag, &quantity) in output
                .master
                .column_tags
                .iter()
                .zip(&output.snapshot.master.values)
            {
                if tag < 0 || !quantity::fractional_quantity(quantity) {
                    continue;
                }
                let route = output.snapshot.state.routes.get(tag as usize)?;
                if route.network != member.network {
                    continue;
                }
                let coefficient = Self::coefficient(member, route, output)?;
                if let Some(coefficient) = coefficient {
                    original += coefficient * quantity;
                    if !original.is_finite() {
                        return None;
                    }
                }
            }
            group += original;
            if !group.is_finite() {
                return None;
            }
        }
        Some(group)
    }
    /// Full residual row activity, separate from candidate admission.
    ///
    /// All original members participate, including continuous members. Raw
    /// physical columns are traversed before original registration order; fixed
    /// contributions belong to the caller's RHS translation, not this sum.
    pub(crate) fn full_value<const D: usize>(&self, output: &Output<D>) -> Option<f64> {
        let mut lhs = 0.;
        for (&tag, &quantity) in output
            .master
            .column_tags
            .iter()
            .zip(&output.snapshot.master.values)
        {
            if tag < 0 {
                continue;
            }
            let route = output.snapshot.state.routes.get(tag as usize)?;
            for member in self.members.iter() {
                if member.network != route.network {
                    continue;
                }
                if let Some(coefficient) = Self::coefficient(member, route, output)? {
                    lhs += (coefficient * 1.) * quantity;
                    if !lhs.is_finite() {
                        return None;
                    }
                }
            }
        }
        Some(lhs)
    }

    fn coefficient<const D: usize>(
        member: &AggregateMember,
        route: &NetworkRoute,
        output: &Output<D>,
    ) -> Option<Option<f64>> {
        if let Some(resource) = member.accumulated_resource {
            let oracle = output
                .snapshot
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == member.network)?;
            if activity_rows::paired(&oracle.network.graph, resource) {
                return None;
            }
            let positions: Option<Vec<_>> = route
                .route
                .arcs
                .iter()
                .map(|id| oracle.index.get(id).copied())
                .collect();
            // Reuse supported integral activity replay. The fixed-arc native
            // arithmetic fixtures do not independently recover resource replay.
            let value =
                activity_rows::column_value(oracle, &route.route, &positions?, resource).ok()?;
            return value.is_finite().then_some(Some(value));
        }
        let mut coefficient = None;
        for arc in &route.route.arcs {
            if let Some(&a) = member.associations.get(arc) {
                let value = coefficient.unwrap_or(0.) + a;
                if !value.is_finite() {
                    return None;
                }
                coefficient = Some(value);
            }
        }
        Some(coefficient)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/aggregate_projection_tests.rs"]
mod tests;
