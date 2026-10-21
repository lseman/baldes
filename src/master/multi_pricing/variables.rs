//! Original aggregate activity bounds, independent of generated columns.
use super::*;
use crate::master::preprocessing::Bounds;

impl<const D: usize> NodeSnapshot<D> {
    /// Restrict an original activity summed over all networks and route copies.
    /// This is a bound on the aggregate variable, not a per-route restriction.
    /// Coefficients apply to retained, pooled, enumerated and future columns.
    /// Fixed contributions are subtracted by the shared-row lifecycle.
    pub fn with_activity_bounds(
        &self,
        bounds: Bounds,
        coefficients: &[RowCoefficients],
    ) -> Result<Self> {
        if bounds.lower.is_nan()
            || bounds.upper.is_nan()
            || bounds.lower == f64::INFINITY
            || bounds.upper == f64::NEG_INFINITY
            || bounds.lower > bounds.upper
        {
            return Err(invalid("original activity bounds"));
        }
        if bounds.lower == bounds.upper {
            return self.with_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: bounds.lower,
                },
                coefficients,
            );
        }
        // Validate coefficients even for an unrestricted activity, using a
        // discarded child rather than allowing malformed definitions through.
        if !bounds.lower.is_finite() && !bounds.upper.is_finite() {
            self.with_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: 0.,
                },
                coefficients,
            )?;
            return self.fork();
        }
        let mut node = self.fork()?;
        if bounds.lower.is_finite() {
            node = node.with_row(
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: bounds.lower,
                },
                coefficients,
            )?;
        }
        if bounds.upper.is_finite() {
            node = node.with_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: bounds.upper,
                },
                coefficients,
            )?;
        }
        Ok(node)
    }

    /// Bound total route multiplicity, including already fixed copies. The
    /// once-per-route linking coefficient remains present in future pricing.
    /// Existing multiplicity limits also apply; this cannot widen a parent.
    pub fn with_multiplicity(&self, network: i32, bounds: Bounds) -> Result<Self> {
        let oracle = self
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == network)
            .ok_or_else(|| invalid("unknown multiplicity network"))?;
        if bounds.lower < 0. {
            return Err(invalid("nonnegative multiplicity lower bound required"));
        }
        self.with_activity_bounds(
            bounds,
            &[RowCoefficients {
                network,
                arcs: vec![0.; oracle.network.graph.arcs.len()],
                per_route: 1.,
            }],
        )
    }
}
