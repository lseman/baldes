//! Route recourse imposed by Theta optimality cuts (Tomazella, Munari and
//! Morabito 2026, eq. 31). Pricing and the master stay those of the
//! deterministic problem: each anchor customer `j` gets a variable
//! `Theta_j >= 0` of unit cost, and a route `r` whose lowest customer is `j`
//! and whose cost exceeds its arc cost by `excess(r) > 0` receives the cut
//!
//! `Theta_j / excess(r) - sum_{a in A(r)} x_a >= 1 - |A(r)|`,
//!
//! binding only when every arc of `r` is used. Integral routes are separated
//! before any other cut, so an integral master solution pays its exact cost.
use super::*;
use std::sync::Arc;

/// Exact route cost in excess of its arc costs.
pub trait RouteRecourse: Send + Sync {
    /// Nonnegative excess of the route visiting `customers` (graph vertex
    /// indices, in visiting order, depot and sink excluded).
    fn excess(&self, customers: &[usize]) -> Result<f64>;
}

#[derive(Clone)]
pub(super) struct Recourse {
    pub(super) oracle: Arc<dyn RouteRecourse>,
    /// Master column of `Theta_j` by graph vertex; none for depot and sink.
    pub(super) columns: Vec<Option<usize>>,
}

impl std::fmt::Debug for Recourse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recourse")
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

/// One separated Theta cut, scaled by `1 / excess`.
#[derive(Clone, Debug)]
pub(super) struct ThetaCut {
    /// Sorted arc IDs of the cut route, depot arcs included.
    pub(super) arcs: Vec<i32>,
    pub(super) anchor: usize,
    pub(super) excess: f64,
}

impl ThetaCut {
    /// Route coefficient: minus the number of route arcs in the cut route.
    pub(super) fn coefficient(&self, route: &Route) -> f64 {
        -(route
            .arcs
            .iter()
            .filter(|id| self.arcs.binary_search(id).is_ok())
            .count() as f64)
    }
}

impl<const D: usize> RoutingContext<D> {
    /// Impose `oracle`'s route excess through Theta optimality cuts. Theta
    /// variables are created by the root solve; this context must not have
    /// been solved yet.
    pub fn with_route_recourse(mut self, oracle: Arc<dyn RouteRecourse>) -> Self {
        self.recourse = Some(Recourse {
            oracle,
            columns: vec![],
        });
        self
    }

    pub fn has_route_recourse(&self) -> bool {
        self.recourse.is_some()
    }

    /// Exact excess of a route; zero without a recourse oracle.
    pub(super) fn route_excess(&self, route: &Route) -> Result<f64> {
        let Some(recourse) = &self.recourse else {
            return Ok(0.);
        };
        let excess = recourse.oracle.excess(&State::<D>::customers(route))?;
        if !excess.is_finite() || excess < 0. {
            return Err(invalid(
                "route recourse excess must be finite and nonnegative",
            ));
        }
        Ok(excess)
    }

    /// Whether master column `column` is a Theta variable.
    pub(super) fn is_theta_column(&self, column: usize) -> bool {
        self.recourse
            .as_ref()
            .is_some_and(|r| r.columns.contains(&Some(column)))
    }
}
