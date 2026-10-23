//! Directed enclosures for individual candidate reduced costs.
use super::super::super::master_variables::add_product;
use super::*;
use crate::core::rounding::{add_down, add_up, multiply_down};
use crate::master::root::ExactSum;

fn enclose(cost: f64, terms: impl IntoIterator<Item = (f64, f64)>) -> Option<Bounds> {
    if !cost.is_finite() {
        return None;
    }
    let mut positive = ExactSum::default();
    let mut negative = ExactSum::default();
    positive.add(cost)?;
    negative.add(-cost)?;
    let mut exact = true;
    let mut interval = Some(Bounds {
        lower: cost,
        upper: cost,
    });
    for (a, dual) in terms {
        if !a.is_finite() || !dual.is_finite() {
            return None;
        }
        if exact {
            exact = add_product(&mut positive, -a, dual).is_some()
                && add_product(&mut negative, a, dual).is_some();
        }
        interval = interval.and_then(|bounds| {
            Some(Bounds {
                lower: add_down(bounds.lower, multiply_down(-a, dual)?)?,
                upper: add_up(bounds.upper, -multiply_down(a, dual)?)?,
            })
        });
    }
    if exact {
        if let (Some(upper), Some(opposite)) = (positive.upper(), negative.upper()) {
            return Some(Bounds {
                lower: -opposite,
                upper,
            });
        }
    }
    interval
}

impl<const D: usize> ImportPricing<D> {
    /// Enclose the exact-real reduced cost of one validated candidate at this
    /// stored binary64 dual point. Cancellation uses exact expansions; tiny
    /// fractional products fall back to directed intervals. Unsupported finite
    /// arithmetic returns None, never an assumed zero or global certificate.
    pub fn reduced_cost_bounds(&self, column: Column) -> Result<Option<Bounds>> {
        self.bounds_at(self.position, column)
    }
    pub(super) fn bounds_at(&self, position: usize, column: Column) -> Result<Option<Bounds>> {
        let variable = self.candidate_at(position, column)?;
        let cost = if self.kind == ImportPricingKind::Objective {
            variable.cost
        } else {
            0.
        };
        let mut terms = Vec::with_capacity(variable.entries.len());
        for (row, a) in variable.entries {
            let dual = *self
                .duals
                .get(self.shared_offset + row)
                .ok_or_else(|| invalid("restricted import dual row mapping"))?;
            terms.push((a, dual));
        }
        Ok(enclose(cost, terms))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/mp/imported/pricing/enclosure/tests.rs"]
mod tests;
