//! Original-domain bounds covering every open, envelope-constrained group.
use super::*;
use crate::core::rounding::{add_down, multiply_down};

/// Bound over the retained finite groups, full routing domains, and every
/// future column permitted by the open groups' declared envelopes. A positive
/// feasibility bound proves infeasibility for that model, not source completion.
#[derive(Clone, Debug)]
pub struct ImportCertificate {
    kind: ImportPricingKind,
    lower_bound: f64,
    open_groups: Vec<i32>,
}
impl ImportCertificate {
    /// Unfinished external groups covered by this certificate.
    pub fn open_groups(&self) -> &[i32] {
        &self.open_groups
    }
    pub fn kind(&self) -> ImportPricingKind {
        self.kind
    }
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }
    pub fn proves_infeasible(&self) -> bool {
        self.kind == ImportPricingKind::Feasibility && self.lower_bound > 0.
    }
}
impl<const D: usize> ColumnImport<D> {
    /// Opt-in original-domain objective bound. Missing envelopes, adverse
    /// unlimited multiplicities, invalid dual signs or unsupported arithmetic
    /// stay unknown. No numerical-only bound is promoted to a certificate.
    pub fn objective_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        self.objective_point(options, true)?.domain_certificate()
    }
    /// Opt-in zero-real-objective bound. Strict positivity establishes original
    /// infeasibility across all future columns permitted by the envelope.
    pub fn feasibility_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        self.feasibility_point(options, true)?.domain_certificate()
    }
    /// Return Some only for a strictly positive, original-domain feasibility
    /// bound. Unknown or nonpositive bounds do not establish infeasibility.
    pub fn infeasibility_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        Ok(self
            .feasibility_certificate(options)?
            .filter(ImportCertificate::proves_infeasible))
    }
}
impl<const D: usize> ImportPricing<D> {
    pub(in crate::master::mp::imported) fn domain_certificate(
        &self,
    ) -> Result<Option<ImportCertificate>> {
        let Some(mut bound) = self.retained_bound else {
            return Ok(None);
        };
        for id in &self.open_groups {
            let position = self
                .definition
                .subproblems
                .iter()
                .position(|g| g.id == *id)
                .ok_or_else(|| invalid("unknown open certificate group"))?;
            let group = &self.definition.subproblems[position];
            let cap = self
                .context
                .multiplicity_upper(group.id, group.multiplicity.upper);
            if cap == 0. {
                continue;
            }
            let Some(minimum) = self.envelope_bound_at(position)? else {
                return Ok(None);
            };
            if minimum < 0. {
                if !cap.is_finite() || !minimum.is_finite() {
                    return Ok(None);
                }
                let Some(correction) = multiply_down(minimum, cap) else {
                    return Ok(None);
                };
                let Some(corrected) = add_down(bound, correction) else {
                    return Ok(None);
                };
                bound = corrected;
            }
        }
        if !bound.is_finite() {
            return Ok(None);
        }
        Ok(Some(ImportCertificate {
            kind: self.kind,
            lower_bound: bound,
            open_groups: self.open_groups.iter().copied().collect(),
        }))
    }
}

pub(super) fn retained_bound<const D: usize>(
    session: &ColumnImport<D>,
    output: &Output<D>,
) -> Option<f64> {
    if session.open_groups.is_empty() {
        super::super::super::bounds::certificate(
            &output.snapshot.state,
            &output.snapshot.master,
            &output.duals,
        )
    } else {
        super::super::super::bounds::extension_certificate(
            &output.snapshot.state,
            &output.snapshot.master,
            &output.duals,
        )
    }
}

/// Derive the open-domain bound at the exact stored duals of this RMP.
pub(in crate::master::mp::imported) fn objective_request_at_output<const D: usize>(
    session: &ColumnImport<D>,
    output: &Output<D>,
    shared_rows: usize,
) -> ImportPricing<D> {
    ImportPricing {
        epoch: Rc::new(()),
        context: session.context.clone(),
        definition: session.definition.clone(),
        position: session.position,
        duals: output.duals.clone(),
        shared_offset: output.snapshot.state.shared_offset,
        shared_rows,
        kind: ImportPricingKind::Objective,
        violation: None,
        routing: vec![],
        retained_bound: retained_bound(session, output),
        open_groups: session.open_groups.clone(),
    }
}
