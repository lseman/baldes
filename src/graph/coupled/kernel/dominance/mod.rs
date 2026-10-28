//! Directional resource order and conservative cumulative cost compensation.
use super::bounds::Range;
use super::*;
use crate::core::rounding::add_up;

mod numeric;
pub(super) use numeric::NumericCostOrder;

/// `tolerances` admits the same per-resource slack as the capacity checks;
/// missing entries compare exactly.
pub(super) fn loads_dominate(
    a: &CoupledLabelState,
    b: &CoupledLabelState,
    tolerances: &[f64],
) -> bool {
    if a.loads.len() != b.loads.len() || a.suffix.is_some() != b.suffix.is_some() {
        return false;
    }
    if a.reversal_peaks.is_some() != b.reversal_peaks.is_some() {
        return false;
    }
    let backward = a.suffix.is_some();
    let reversed = a.reversed_use || b.reversed_use;
    a.loads
        .iter()
        .zip(&b.loads)
        .enumerate()
        .all(|(index, (la, lb))| {
            let tolerance = tolerances.get(index).copied().unwrap_or(0.);
            let fits = |x: f64, y: f64| x <= y + tolerance;
            // A suffix is prepended to and joined only through its deliveries
            // D and peak D+M; a prefix is appended to and joined only through
            // its pickups D+N and peak D+M.
            let peak = fits(la[0] + la[2], lb[0] + lb[2]);
            if backward {
                return fits(la[0], lb[0]) && peak;
            }
            let prefix = fits(la[0] + la[1], lb[0] + lb[1]) && peak;
            if !reversed {
                return prefix;
            }
            // A reversed prefix acts as a suffix whose peak is its reversal
            // peak when mixed-sign states retain one, otherwise M.
            let (ra, rb) = match (&a.reversal_peaks, &b.reversal_peaks) {
                (Some(pa), Some(pb)) => (pa[index], pb[index]),
                _ => (la[2], lb[2]),
            };
            prefix && fits(la[0], lb[0]) && fits(la[0] + ra, lb[0] + rb)
        })
}

fn weaker_bounds(a: &Suffix, b: &Suffix) -> bool {
    a.bounds.iter().all(|a| {
        b.bounds.iter().any(|b| {
            a.coefficients == b.coefficients
                && a.offset == b.offset
                && a.bounds.lower <= b.bounds.lower
                && a.bounds.upper >= b.bounds.upper
        })
    })
}
impl CoupledKernel {
    /// A nonnegative upper bound on the additional continuation cost of a
    /// relative to b. None means resource feasibility/order is not established.
    /// Nonnegative penalties preserve the existing hot-key scan ordering.
    pub fn dominance_penalty(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
    ) -> Option<f64> {
        self.dominance_penalty_with_frozen_fronts(a, b, forward, false)
    }

    /// Compare states at a shared physical vertex, using its prepared
    /// continuation envelope when available. Both states must be interpreted
    /// at that vertex; no per-label topology is stored by the resource kernel.
    pub fn dominance_penalty_at_vertex(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        vertex: usize,
    ) -> Option<f64> {
        self.dominance_penalty_in_context(a, b, forward, false, Some(vertex))
    }

    /// `frozen_fronts` is a caller-established forward-only proof: every
    /// reachable future interval-resource transition has zero increment and
    /// zero scaled charge, and preserves all of a's current front domains.
    /// Consequently every retained minimum and every future min-delta remain
    /// unchanged. This location-specific proof must not be inferred from a
    /// current normalized-function difference or a rounded reconstructed key.
    /// The caller must additionally establish that the current comparison keys
    /// preserve the canonical physical-prefix cost order. Freezing future
    /// operations cannot repair roundoff already accumulated in earlier deltas.
    pub(super) fn dominance_penalty_with_frozen_fronts(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        frozen_fronts: bool,
    ) -> Option<f64> {
        self.dominance_penalty_in_context(a, b, forward, frozen_fronts, None)
    }

    fn dominance_penalty_in_context(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        frozen_fronts: bool,
        vertex: Option<usize>,
    ) -> Option<f64> {
        if vertex.is_some_and(|vertex| !self.route_domains.contains_vertex(vertex)) {
            return None;
        }
        // Reject cheap direction/discrete/component mismatches before
        // traversing paired front pieces or validating retained suffix rows.
        if a.suffix.is_none() != forward
            || b.suffix.is_none() != forward
            || !loads_dominate(a, b, &self.load_tolerances)
            || !self.budgets_dominate(&a.budgets, &b.budgets, forward)
            || a.selection
                .iter()
                .zip(&b.selection)
                .any(|(a, b)| a & b != *b)
        {
            return None;
        }
        if self.has_negative_paired_objectives() {
            self.check_state(a).ok()?;
            self.check_state(b).ok()?;
            // A signed front alone does not identify the selected source
            // control used by its terminal charge. Keep even identical
            // unselected states unknown; path-level pricing/certification
            // supplies that control separately.
            if a == b {
                return None;
            }
            // Signed paired fronts compare only when their directed coverage
            // proofs succeed and every other continuation/accounting
            // component is identical. The hot label key checks current cost.
            // Native tolerance is never used as a proof rule.
            return self.signed_paired_fronts_dominate(a, b, forward);
        }
        if self.weighted_objective || !self.pricing_activities.is_empty() {
            // Until a joint component proof is available, only identical
            // coupled/pricing states may dominate. A base-resource penalty
            // alone cannot pay for a signed activity adjustment.
            return (a == b).then_some(0.);
        }
        // Compare every active resource-cost family's independent scalar and
        // resource components before the hot mixed key. Different prefix
        // histories can otherwise round to the same key while their physical
        // scalar sums differ, even when the final front objects are identical.
        let charged = self.has_costs();
        // Delay the per-slot proof until cheap scalar/component checks pass.
        let numeric_order = || {
            !charged
                // Baseline contract: identical numeric inputs replay the same
                // future operations. Common finite additions preserve prefix
                // cost order; unsupported overflow aborts pricing globally.
                || a.numeric == b.numeric
                || (forward && vertex.is_some_and(|vertex| self.numeric_cost_order.preserves_at_vertex(
                    vertex, &a.numeric, &b.numeric, a.numeric_cost, b.numeric_cost,
                )))
                || (forward && self.numeric_cost_order.deploys_unequal()
                    && self.numeric_cost_order.preserves(
                        &a.numeric, &b.numeric, a.numeric_cost, b.numeric_cost,
                    ))
        };
        let tracked_order = if !charged
            && self.front_indices.is_empty()
            && self.slope_indices.is_empty()
            && self.relation_indices.is_empty()
        {
            false
        } else {
            match (a.accounting, b.accounting) {
                (Some(a_cost), Some(b_cost)) => {
                    if a_cost.reduced > b_cost.reduced
                        || a.numeric_cost > b.numeric_cost
                        || !numeric_order()
                        || a.fronts
                            .iter()
                            .zip(&b.fronts)
                            .any(|(a, b)| a.minimum() > b.minimum())
                    {
                        return None;
                    }
                    true
                }
                (None, None) => {
                    // Public resource-only states carry no scalar history,
                    // but their already accumulated numeric objective is
                    // still part of the continuation component contract.
                    if (charged && a.numeric_cost > b.numeric_cost) || !numeric_order() {
                        return None;
                    }
                    false
                }
                _ => return None,
            }
        };
        let frozen_fronts = frozen_fronts && tracked_order;
        if !forward
            && a.suffix.as_ref()?.program != b.suffix.as_ref()?.program
            && !self.backward_programs_free(a, b)
            && !self.backward_programs_equivalent(a, b)
        {
            return None;
        }
        // Retain full validation before any accepted dominance result. A
        // malformed state rejected by an earlier necessary condition also
        // yields None, exactly as failed validation did before reordering.
        self.check_state(a).ok()?;
        self.check_state(b).ok()?;
        let mut penalty = 0.;
        if a.fronts.len() != b.fronts.len() {
            return None;
        }
        for ((a, b), &index) in a.fronts.iter().zip(&b.fronts).zip(&self.front_indices) {
            if a == b {
                continue;
            }
            if a.domain().lower > b.domain().lower || a.domain().upper < b.domain().upper {
                return None;
            }
            if frozen_fronts && forward {
                continue;
            }
            // A bound on the CURRENT normalized difference is not a bound on
            // future rounded additions. Even gap 1 can become gap 2 after a
            // common charge at 2^53. Preserve general unequal costly fronts.
            // An all-zero front whose every future charge is suppressed/zero
            // is a safe broader subset: all subsequent minima/deltas stay 0,
            // including after clipping or translating its feasible domain.
            let CoupledResource::IntervalCost { functions, .. } = &self.definitions[index] else {
                unreachable!()
            };
            let zero_charges = self.cost_scale == 0.
                || functions
                    .iter()
                    .all(|function| function.points.iter().all(|p| p[1] == 0.));
            if !zero_charges || a.points().iter().chain(b.points()).any(|p| p[1] != 0.) {
                return None;
            }
        }
        // Raw slope-interval fronts: bitwise-equal labels replay identical
        // future operations and compare through the scalar order above, their
        // minima being equal. Unequal forward labels need the per-policy rule.
        // Backward labels hold no front and match by suffix program.
        if a.slope_fronts.len() != b.slope_fronts.len() {
            return None;
        }
        let mut slope_penalty = 0.;
        for ((left, right), prepared) in a
            .slope_fronts
            .iter()
            .zip(&b.slope_fronts)
            .zip(&self.slope_dominance)
        {
            if left == right {
                continue;
            }
            if !forward {
                return None;
            }
            let gap = prepared.penalty(left, right, self.cost_scale, tracked_order)?;
            slope_penalty = add_up(slope_penalty, gap)?;
        }
        // Paired checks are deliberately last among these necessary
        // preconditions: rejected scalar/discrete/program pairs need no walk
        // through the retained function and backward-adjoint pieces.
        if !self.plugins_dominate(a, b, forward) {
            return None;
        }
        let relation_penalty = add_up(
            self.relations_dominate(a, b, forward, tracked_order)?,
            slope_penalty,
        )?;
        let relation_penalty = add_up(
            relation_penalty,
            self.slope_relations_dominate(a, b, forward, tracked_order)?,
        )?;
        if !forward {
            let a = a.suffix.as_ref()?;
            let b = b.suffix.as_ref()?;
            if a.bounds != b.bounds && !weaker_bounds(a, b) {
                return None;
            }
            return add_up(penalty, self.backward_penalty(a, b, vertex)?);
        }
        if !self.optimization.dependency_order.preserves(
            &a.numeric,
            &b.numeric,
            &self.optimization.numeric,
        ) {
            return None;
        }
        if charged {
            // The prepared per-operation proof above establishes literal
            // numeric component order. Only a cost-unbounded relation's
            // shifted front adds compensation.
            return Some(relation_penalty);
        }
        penalty = add_up(penalty, relation_penalty)?;
        for (i, (&a, &b)) in a.numeric.iter().zip(&b.numeric).enumerate() {
            if a == b {
                continue;
            }
            if self.cost_scale == 0. {
                continue;
            }
            let weights = self.optimization.weights[i];
            // These orders are preserved even by rounded arithmetic because
            // the same additive numeric transition is applied to both labels.
            if (weights.lo >= 0. && a <= b) || (weights.hi <= 0. && a >= b) {
                continue;
            }
            // Cost tradeoffs use finite walk-weight bounds only where all
            // cumulative state/product sums remain exact binary64 integers.
            // Other signed/unbounded domains retain the conservative rule.
            if !self.optimization.exact_costs {
                return None;
            }
            let domain = self.optimization.numeric[i];
            if a.fract() != 0.
                || b.fract() != 0.
                || a < domain.lo
                || a > domain.hi
                || b < domain.lo
                || b > domain.hi
            {
                return None;
            }
            let gap = Range::point(a).subtract(Range::point(b)).multiply(weights);
            if !gap.hi.is_finite() {
                return None;
            }
            penalty = add_up(penalty, gap.hi.max(0.))?;
        }
        Some(penalty)
    }

    fn signed_paired_fronts_dominate(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
    ) -> Option<f64> {
        let relation_changed = a.relations != b.relations;
        let slope_changed = a.slope_relations != b.slope_relations;
        if !forward
            || self.cost_scale <= 0.
            || (!relation_changed && !slope_changed)
            || a.accounting.is_none()
            || a.accounting != b.accounting
            || a.numeric_cost != b.numeric_cost
            || a.numeric != b.numeric
            || a.relations.len() != self.paired.len()
            || b.relations.len() != self.paired.len()
            || a.relations.iter().zip(&b.relations).zip(&self.paired).any(
                |((left, right), prepared)| {
                    let scale = self.cost_scale * prepared.objective_coefficient;
                    left != right && (!prepared.has_costs() || !scale.is_finite() || scale >= 0.)
                },
            )
            || a.slope_relations.len() != self.slope_paired.len()
            || b.slope_relations.len() != self.slope_paired.len()
            || a.slope_relations
                .iter()
                .zip(&b.slope_relations)
                .zip(&self.slope_paired)
                .any(|((left, right), prepared)| {
                    left != right && !prepared.uses_strict_front_policy()
                })
        {
            return None;
        }
        // Permit only paired labels and their recomputed objective total to
        // differ. Scalar sums, other resource fronts, plugins, row adjustments,
        // route masks, and suffix state must match exactly.
        let mut normalized = a.clone();
        normalized.relations.clone_from(&b.relations);
        normalized.slope_relations.clone_from(&b.slope_relations);
        normalized.resource_objective = b.resource_objective;
        if normalized != *b {
            return None;
        }
        let relations = self.relations_dominate(a, b, true, true)?;
        let slopes = self.slope_relations_dominate(a, b, true, true)?;
        add_up(relations, slopes)
    }

    /// Backward labels with different suffix arc programs may still compare
    /// when every replay-dependent relation has a sufficient suffix summary:
    /// a feasibility upper front or an exact bounded-cost transfer. Other
    /// resources retain the common-program restriction.
    fn backward_programs_free(&self, a: &CoupledLabelState, b: &CoupledLabelState) -> bool {
        !self.transitions.deferred
            && self.plugins.is_empty()
            && self.slope_indices.is_empty()
            && self
                .slope_paired
                .iter()
                .zip(&a.slope_relations)
                .zip(&b.slope_relations)
                .all(|((prepared, a), b)| {
                    !prepared.has_costs() || (a.bounded_cost.is_some() && b.bounded_cost.is_some())
                })
            && a.slope_relations
                .iter()
                .chain(&b.slope_relations)
                .all(|label| label.upper.is_some() || label.bounded_cost.is_some())
            && self
                .paired
                .iter()
                .zip(&a.relations)
                .zip(&b.relations)
                .all(|((prepared, a), b)| {
                    !prepared.has_costs()
                        || (a.backward_cost().is_some() && b.backward_cost().is_some())
                })
            && a.relations
                .iter()
                .chain(&b.relations)
                .all(|label| label.backward_upper().is_some() || label.backward_cost().is_some())
    }

    /// Different physical paths can execute exactly the same paired program.
    /// Comparing the ordered source operators proves identical replay without
    /// requiring a numerical adjoint. Keep every other replaying family out of
    /// this proof; their aggregate backward states alone are not sufficient.
    fn backward_programs_equivalent(&self, a: &CoupledLabelState, b: &CoupledLabelState) -> bool {
        if !self.initial.is_empty()
            || !self.selection_initial.is_empty()
            || !self.load_capacities.is_empty()
            || !self.budgeted.is_empty()
            || !self.front_indices.is_empty()
            || !self.slope_indices.is_empty()
            || !self.plugins.is_empty()
        {
            return false;
        }
        let (Some(a), Some(b)) = (&a.suffix, &b.suffix) else {
            return false;
        };
        a.program.equivalent_by(&b.program, |a, b| {
            a < self.arc_count
                && b < self.arc_count
                && self.paired.iter().all(|p| p.same_arc_operation(a, b))
                && self.slope_paired.iter().all(|p| p.same_arc_operation(a, b))
        })
    }

    /// Paired slope relations: forward fronts cover and stay below, with the
    /// same scalar-order requirement as function relations when charged. A
    /// shifting cost-unbounded relation may exceed and pay. Returns the penalty.
    fn slope_relations_dominate(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        tracked_order: bool,
    ) -> Option<f64> {
        if a.slope_relations.len() != b.slope_relations.len() {
            return None;
        }
        let mut penalty = 0.;
        for ((left, right), prepared) in a
            .slope_relations
            .iter()
            .zip(&b.slope_relations)
            .zip(&self.slope_paired)
        {
            if left == right {
                continue;
            }
            let charged = prepared.has_costs() && self.cost_scale != 0.;
            if charged && !(tracked_order && self.cost_scale > 0.) {
                return None;
            }
            let gap = prepared.penalty(left, right, forward, self.cost_scale)?;
            penalty = add_up(penalty, gap)?;
        }
        Some(penalty)
    }

    /// Every plugin's own direction-specific dominance test.
    fn plugins_dominate(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
    ) -> bool {
        a.plugins.len() == b.plugins.len()
            && a.plugins
                .iter()
                .zip(&b.plugins)
                .zip(&self.plugins)
                .all(|((a, b), plugin)| {
                    if plugin.0.has_costs() && !plugin.0.supports_cost_dominance() {
                        a == b
                    } else {
                        plugin.0.dominates(a, b, forward)
                    }
                })
    }

    /// Forward relation fronts dominate by covering and staying below; the
    /// final minimum is then no higher for every completion. A charged
    /// relation also needs the scalar prefix order proven by `tracked_order`,
    /// since its own objective term is then no higher as well. In
    /// cost-unbounded mode without upper bounds a front may exceed the other
    /// by delta and pay scale * (delta + min_b - min_a) (the reference's
    /// max(a - b) cost charge). Backward labels compare feasibility upper
    /// fronts or exact bounded-cost transfers; unknown transfers retain the
    /// common-program restriction. Returns the penalty.
    fn relations_dominate(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        tracked_order: bool,
    ) -> Option<f64> {
        if a.relations.len() != b.relations.len() {
            return None;
        }
        let mut penalty = 0.;
        for ((left, right), prepared) in a.relations.iter().zip(&b.relations).zip(&self.paired) {
            if left == right {
                continue;
            }
            if !forward {
                if prepared.backward_dominates(left, right, self.cost_scale) {
                    continue;
                }
                return None;
            }
            let upper_left = left.forward_upper_negative();
            let upper_right = right.forward_upper_negative();
            let (Some(left), Some(right)) = (left.forward(), right.forward()) else {
                return None;
            };
            let charged = prepared.has_costs() && self.cost_scale != 0.;
            if !forward || (charged && !(tracked_order && self.cost_scale > 0.)) {
                return None;
            }
            let signed_scale = self.cost_scale * prepared.objective_coefficient;
            if !signed_scale.is_finite() {
                return None;
            }
            if charged && signed_scale < 0. {
                let (Some(upper_left), Some(upper_right)) = (upper_left, upper_right) else {
                    return None;
                };
                if left.covers_below_strict(right) && upper_left.covers_below_strict(upper_right) {
                    continue;
                }
                return None;
            }
            let excess = left.excess(right)?;
            if excess <= 0. {
                continue;
            }
            if !(charged && prepared.shifts()) {
                return None;
            }
            let gap = Range::point(excess)
                .add(Range::point(right.minimum()))
                .subtract(Range::point(left.minimum()))
                .multiply(Range::point(self.cost_scale))
                .hi;
            penalty = add_up(penalty, gap.max(0.))?;
        }
        Some(penalty)
    }
}
impl<const D: usize> Graph<D> {
    pub(crate) fn coupled_dominance_penalty(
        &self,
        a: Option<&CoupledLabelState>,
        b: Option<&CoupledLabelState>,
        forward: bool,
        vertex: usize,
    ) -> Option<f64> {
        match (self.coupled_kernel(), a, b) {
            (None, None, None) => Some(0.),
            (Some(kernel), Some(a), Some(b)) => {
                let frozen = forward
                    && a.fronts.iter().enumerate().all(|(slot, front)| {
                        kernel
                            .frozen
                            .contains(slot, vertex, front.domain(), kernel.cost_scale)
                    });
                kernel.dominance_penalty_in_context(a, b, forward, frozen, Some(vertex))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/dominance/front_continuation_tests.rs"]
mod front_continuation_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/dominance/load_dominance_tests.rs"]
mod load_dominance_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/dominance/load_tolerance_tests.rs"]
mod load_tolerance_tests;
