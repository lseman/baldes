//! Request-bound numerical oracle results, distinct from directed certificates.
use super::*;
use crate::master::rounding::{add_down, multiply_down};

/// Caller-supplied global numerical reduced-cost lower bound for every possible
/// missing column in this group's intended domain, at this exact request.
/// A minimum over a heuristic candidate list is not sufficient. This is trusted
/// numerical oracle output, never a directed-arithmetic certificate.
#[derive(Clone, Debug)]
pub struct ImportPricingMinimum {
    epoch: Rc<()>,
    position: usize,
    value: f64,
}
impl ImportPricingMinimum {
    pub fn lower_bound(&self) -> f64 {
        self.value
    }
}
#[derive(Clone, Debug)]
pub struct ImportOracleReply {
    pub batch: ColumnBatch,
    pub minimum: Option<ImportPricingMinimum>,
}
impl ImportOracleReply {
    pub fn columns(batch: ColumnBatch) -> Self {
        Self {
            batch,
            minimum: None,
        }
    }
}
impl<const D: usize> ImportPricing<D> {
    /// Bind a trusted numerical oracle lower bound to this point, phase and
    /// group. It must cover all permitted future columns, not only those found.
    /// Negative infinity means unknown; positive infinity declares no missing
    /// column. Neither value declares exhaustive enumeration complete.
    pub fn numerical_minimum(&self, lower_bound: f64) -> Result<ImportPricingMinimum> {
        if lower_bound.is_nan() {
            return Err(invalid("NaN import pricing minimum"));
        }
        Ok(ImportPricingMinimum {
            epoch: self.epoch.clone(),
            position: self.position,
            value: lower_bound,
        })
    }
    /// Retarget the same stored dual point to another unfinished group. This
    /// performs no LP solve and retains the request epoch for joint coverage.
    pub fn for_subproblem(&self, group: i32) -> Result<Self> {
        if !self.open_groups.contains(&group) {
            return Err(invalid("group is not open at this pricing point"));
        }
        let position = self
            .definition
            .subproblems
            .iter()
            .position(|g| g.id == group)
            .ok_or_else(|| invalid("unknown import pricing group"))?;
        let mut point = self.clone();
        point.position = position;
        Ok(point)
    }
    pub(in crate::master::multi_pricing::imported) fn validate_reply(
        &self,
        reply: &ImportOracleReply,
        tolerance: f64,
    ) -> Result<()> {
        let Some(minimum) = &reply.minimum else {
            return Ok(());
        };
        if !Rc::ptr_eq(&minimum.epoch, &self.epoch) || minimum.position != self.position {
            return Err(invalid(
                "stale, wrong-phase or wrong-group import pricing minimum",
            ));
        }
        // Catch a direct contradiction with a returned vector. Coverage of the
        // unobserved domain is the external optimizer's declared contract.
        for column in &reply.batch.columns {
            let reduced = self.reduced_cost(column.clone())?;
            if minimum.value > reduced && minimum.value - reduced > tolerance {
                return Err(invalid(
                    "import pricing minimum exceeds returned column reduced cost",
                ));
            }
        }
        Ok(())
    }
    /// The retained base is independently bounded; supplied numerical minima
    /// contribute only to numerical publication. Missing groups can still use
    /// a declared envelope. Every adverse missing quantity needs a finite cap.
    pub(in crate::master::multi_pricing::imported) fn numerical_bound(
        &self,
        minima: &[ImportPricingMinimum],
    ) -> Result<Option<f64>> {
        let mut supplied = BTreeMap::new();
        for minimum in minima {
            if !Rc::ptr_eq(&minimum.epoch, &self.epoch)
                || !self
                    .open_groups
                    .contains(&self.definition.subproblems[minimum.position].id)
                || supplied.insert(minimum.position, minimum.value).is_some()
            {
                return Err(invalid("incompatible or duplicate import pricing minima"));
            }
        }
        let Some(mut value) = self.retained_bound else {
            return Ok(None);
        };
        for id in &self.open_groups {
            let position = self
                .definition
                .subproblems
                .iter()
                .position(|g| g.id == *id)
                .unwrap();
            let group = &self.definition.subproblems[position];
            let cap = self
                .context
                .multiplicity_upper(*id, group.multiplicity.upper);
            if cap == 0. {
                continue;
            }
            let envelope = self.envelope_bound_at(position)?;
            let minimum = match (supplied.get(&position), envelope) {
                (Some(&a), Some(b)) => a.max(b),
                (Some(&a), None) => a,
                (None, Some(b)) => b,
                (None, None) => return Ok(None),
            };
            if minimum < 0. {
                if !cap.is_finite() || !minimum.is_finite() {
                    return Ok(None);
                }
                let Some(correction) = multiply_down(minimum, cap) else {
                    return Ok(None);
                };
                let Some(next) = add_down(value, correction) else {
                    return Ok(None);
                };
                value = next;
            }
        }
        Ok(value.is_finite().then_some(value))
    }
}
