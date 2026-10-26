//! Per-solve branch evaluation history. Estimates rank candidates, never prune.
use super::candidates::BranchCandidate;
use super::*;
mod admission;
mod generators;
pub(in crate::master) use admission::fresh_batches_with_budget;
pub(in crate::master::tree) use admission::registered as registered_priorities;
mod resource_generators;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvaluationKind {
    RestrictedMaster,
    HeuristicPricing,
    ExactPricing,
    NodePricing,
}

#[derive(Clone, Debug)]
pub struct BranchEvaluation {
    pub constraint: BranchConstraint,
    pub depth: usize,
    pub kind: EvaluationKind,
    /// One-based evaluation count on this candidate side at the current node;
    /// skipped phases consume no stage. Absent for ordinary nodes.
    pub stage: Option<usize>,
    pub parent_bound: f64,
    pub estimate: f64,
    pub distance: f64,
    pub seconds: f64,
}
#[derive(Clone, Debug)]
pub struct BranchHistoryOptions {
    /// Observations required on each side for `Reliable` (at least one).
    /// Ignored by `Default`.
    pub minimum_observations: usize,
    pub policy: BranchHistoryPolicy,
}
/// Ranking rules only. Neither policy produces a pruning bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BranchHistoryPolicy {
    /// Earlier local policy, retained for controlled comparisons in Rust.
    Reliable,
    /// Phase-aware mean-cost ranking with canonical constraint names.
    #[default]
    Default,
}
impl Default for BranchHistoryOptions {
    fn default() -> Self {
        Self {
            minimum_observations: 2,
            policy: BranchHistoryPolicy::Default,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct BranchEvaluationHistory {
    pub evaluations: Vec<BranchEvaluation>,
    pub ranked_candidates: usize,
    costs: BTreeMap<String, BTreeMap<EvaluationKind, [RunningCost; 2]>>,
    phase_costs: BTreeMap<String, BTreeMap<usize, [RunningCost; 2]>>,
    resource_generators: BTreeMap<String, BranchCandidate>,
    generators: BTreeMap<String, BranchCandidate>,
}
#[derive(Clone, Debug, Default)]
struct RunningCost {
    mean: f64,
    count: usize,
    signed_mean: f64,
    signed_count: usize,
}

use crate::master::branching::generator_identity::key;

// Native phased strong branching excludes component-set generators before
// attaching evaluation history. Per-graph route count is that generator: it
// sums the pricing configuration's columns, rather than an original variable.
fn history_eligible(constraint: &BranchConstraint) -> bool {
    !matches!(constraint, BranchConstraint::NumberOfVehicles { .. })
}

impl BranchEvaluationHistory {
    #[cfg(test)]
    fn record(
        &mut self,
        candidate: &BranchCandidate,
        side: usize,
        depth: usize,
        kind: EvaluationKind,
        parent: f64,
        estimate: f64,
        seconds: f64,
    ) {
        self.record_at_stage(
            candidate,
            side,
            depth,
            kind,
            Some(kind as usize + 1),
            parent,
            estimate,
            seconds,
        );
    }
    pub(super) fn record_at_stage(
        &mut self,
        candidate: &BranchCandidate,
        side: usize,
        depth: usize,
        kind: EvaluationKind,
        stage: Option<usize>,
        parent: f64,
        estimate: f64,
        seconds: f64,
    ) {
        if !parent.is_finite() || !estimate.is_finite() || estimate >= 1e12 {
            return;
        }
        let fraction = candidate.value - candidate.value.floor();
        let distance = if side == 0 { fraction } else { 1. - fraction };
        if distance <= 1e-6 || !distance.is_finite() {
            return;
        }
        let cost = ((estimate - parent).max(0.) / distance).min(1e100);
        if !cost.is_finite() {
            return;
        }
        let stats = &mut self
            .costs
            .entry(key(&candidate.down))
            .or_default()
            .entry(kind)
            .or_default()[side];
        stats.count += 1;
        stats.mean += (cost - stats.mean) / stats.count as f64;
        let signed_cost = (estimate - parent) / distance;
        if signed_cost.is_finite() {
            stats.signed_count += 1;
            // Keep the recovered operation order for native trace comparisons.
            stats.signed_mean = ((stats.signed_count - 1) as f64 * stats.signed_mean + signed_cost)
                / stats.signed_count as f64;
        }
        if let Some(stage) = stage.filter(|&stage| stage > 0) {
            if signed_cost.is_finite() && history_eligible(&candidate.down) {
                let registry = if matches!(
                    candidate.down,
                    BranchConstraint::AccumulatedResourceGroup(_)
                ) {
                    &mut self.resource_generators
                } else {
                    &mut self.generators
                };
                registry
                    .entry(key(&candidate.down))
                    .or_insert_with(|| candidate.clone());
            }
            let stats = &mut self
                .phase_costs
                .entry(key(&candidate.down))
                .or_default()
                .entry(stage)
                .or_default()[side];
            stats.count += 1;
            stats.mean += (cost - stats.mean) / stats.count as f64;
            if signed_cost.is_finite() {
                stats.signed_count += 1;
                stats.signed_mean = ((stats.signed_count - 1) as f64 * stats.signed_mean
                    + signed_cost)
                    / stats.signed_count as f64;
            }
        }
        self.evaluations.push(BranchEvaluation {
            constraint: if side == 0 {
                candidate.down.clone()
            } else {
                candidate.up.clone()
            },
            depth,
            kind,
            stage,
            parent_bound: parent,
            estimate,
            distance,
            seconds,
        });
    }
    /// Predict a ranking score without an incumbent cap. Tree selection uses
    /// `score_with_gap`/`prioritize` with the actual incumbent gap.
    pub fn score(
        &self,
        candidate: &BranchCandidate,
        options: &BranchHistoryOptions,
    ) -> Option<f64> {
        self.score_with_gap(candidate, options, f64::INFINITY)
    }
    /// Predict a ranking score for the supplied incumbent-minus-parent gap.
    pub fn score_with_gap(
        &self,
        candidate: &BranchCandidate,
        options: &BranchHistoryOptions,
        gap: f64,
    ) -> Option<f64> {
        self.prediction(candidate, options, gap)
            .map(|(_, score)| score)
    }
    fn prediction(
        &self,
        candidate: &BranchCandidate,
        options: &BranchHistoryOptions,
        gap: f64,
    ) -> Option<(usize, f64)> {
        if options.policy == BranchHistoryPolicy::Default && !history_eligible(&candidate.down) {
            return None;
        }
        let fraction = candidate.value - candidate.value.floor();
        if !fraction.is_finite() || fraction <= 1e-6 || fraction >= 1. - 1e-6 {
            return None;
        }
        let identity = key(&candidate.down);
        let (stage, stats) = match options.policy {
            BranchHistoryPolicy::Reliable => {
                let (&kind, stats) =
                    self.costs.get(&identity)?.iter().rev().find(|(_, sides)| {
                        sides
                            .iter()
                            .all(|s| s.count >= options.minimum_observations.max(1))
                    })?;
                (kind as usize + 1, stats)
            }
            BranchHistoryPolicy::Default => {
                let (&stage, stats) = self
                    .phase_costs
                    .get(&identity)?
                    .iter()
                    .rev()
                    .find(|(_, sides)| sides.iter().any(|s| s.signed_count > 0))?;
                (stage, stats)
            }
        };
        let score = match options.policy {
            BranchHistoryPolicy::Reliable => {
                (stats[0].mean * fraction).max(1e-6) * (stats[1].mean * (1. - fraction)).max(1e-6)
            }
            BranchHistoryPolicy::Default => {
                if gap.is_nan() || gap <= 0. || stats.iter().any(|s| !s.signed_mean.is_finite()) {
                    return None;
                }
                (stats[0].signed_mean * fraction).min(gap)
                    * (stats[1].signed_mean * (1. - fraction)).min(gap)
            }
        };
        score.is_finite().then_some((stage, score))
    }
    /// At non-root nodes reserve half the first-stage budget for history,
    /// filling the remainder in the supplied fractional order.
    pub(super) fn prioritize(
        &mut self,
        candidates: &mut Vec<BranchCandidate>,
        limit: usize,
        options: &BranchHistoryOptions,
        gap: f64,
    ) {
        let mut ranked: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                self.prediction(c, options, gap).map(|(kind, score)| {
                    let name = match options.policy {
                        BranchHistoryPolicy::Reliable => String::new(),
                        BranchHistoryPolicy::Default => match &c.down {
                            BranchConstraint::AccumulatedResourceGroup(group) => {
                                group.generator_name()
                            }
                            _ => key(&c.down),
                        },
                    };
                    (i, kind, score, name)
                })
            })
            .collect();
        ranked.sort_by(|a, b| {
            match options.policy {
                BranchHistoryPolicy::Reliable => b.2.total_cmp(&a.2),
                BranchHistoryPolicy::Default => {
                    b.1.cmp(&a.1)
                        .then_with(|| b.2.partial_cmp(&a.2).unwrap())
                        .then_with(|| a.3.cmp(&b.3))
                }
            }
            .then(a.0.cmp(&b.0))
        });
        ranked.truncate((limit / 2).min(candidates.len()));
        self.ranked_candidates += ranked.len();
        let mut used = vec![false; candidates.len()];
        let mut next = Vec::with_capacity(candidates.len());
        for (i, _, _, _) in ranked {
            used[i] = true;
            next.push(candidates[i].clone());
        }
        next.extend(
            candidates
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(_, c)| c.clone()),
        );
        *candidates = next;
    }
    pub(super) fn to_json(&self) -> String {
        self.evaluations.iter().map(|e| format!(
            "{{\"constraint\":{:?},\"depth\":{},\"kind\":\"{:?}\",\"stage\":{},\"parent_bound\":{},\"estimate\":{},\"distance\":{},\"seconds\":{}}}",
            format!("{:?}", e.constraint), e.depth, e.kind, e.stage.map_or_else(|| "null".into(), |stage| stage.to_string()), e.parent_bound, e.estimate, e.distance, e.seconds,
        )).collect::<Vec<_>>().join(",")
    }
}
