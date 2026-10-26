//! Box relaxation of all permitted future candidate reduced costs.
use super::*;

impl<const D: usize> ImportPricing<D> {
    /// Lower bound on reduced cost of every column permitted by this group's
    /// envelope, at these stored duals. Correlations are relaxed. This is not
    /// a full-master bound or a domain-completion/feasibility certificate.
    pub fn envelope_reduced_cost_bound(&self) -> Result<Option<f64>> {
        self.envelope_bound_at(self.position)
    }
    pub(super) fn envelope_bound_at(&self, position: usize) -> Result<Option<f64>> {
        let group = &self.definition.subproblems[position];
        let Some(envelope) = self.definition.envelopes.get(&group.id) else {
            return Ok(None);
        };
        let cost = if self.kind() == ImportPricingKind::Objective {
            envelope.cost_lower
        } else {
            0.
        };
        if cost == f64::NEG_INFINITY {
            return Ok(Some(f64::NEG_INFINITY));
        }
        let rows = &self.registry.group_rows[&group.id];
        let mut entries = vec![];
        for &(id, bounds) in &envelope.coordinates {
            let price = self.duals[self.shared_offset + rows.links[&id]];
            let value = if price > 0. {
                bounds.lower
            } else if price < 0. {
                bounds.upper
            } else {
                0_f64.max(bounds.lower).min(bounds.upper)
            };
            if !value.is_finite() {
                return Ok(Some(f64::NEG_INFINITY));
            }
            entries.push((id, value));
        }
        let id = rows
            .next_column
            .ok_or_else(|| invalid("pricing candidate IDs exhausted"))?;
        Ok(self
            .bounds_at(
                position,
                Column {
                    id,
                    cost,
                    integer: false,
                    bounds: Bounds {
                        lower: 0.,
                        upper: f64::INFINITY,
                    },
                    entries,
                },
            )?
            .map(|b| b.lower))
    }
}
