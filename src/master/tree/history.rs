//! Per-solve branch evaluation history. Estimates rank candidates, never prune.
use super::candidates::BranchCandidate;
use super::*;

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
    pub parent_bound: f64,
    pub estimate: f64,
    pub distance: f64,
    pub seconds: f64,
}
#[derive(Clone, Debug)]
pub struct BranchHistoryOptions {
    /// Observations required on each side, in the same evaluation stage.
    pub minimum_observations: usize,
}
impl Default for BranchHistoryOptions {
    fn default() -> Self {
        Self {
            minimum_observations: 2,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct BranchEvaluationHistory {
    pub evaluations: Vec<BranchEvaluation>,
    pub ranked_candidates: usize,
    costs: BTreeMap<String, BTreeMap<EvaluationKind, [RunningCost; 2]>>,
}
#[derive(Clone, Debug, Default)]
struct RunningCost {
    mean: f64,
    count: usize,
}

// Drop RHS/sense so estimates generalize between bounds on the same variable.
// Keep schedule thresholds: distinct resource disjunctions need distinct costs.
fn key(c: &BranchConstraint) -> String {
    match c {
        BranchConstraint::ArcFlow { arc, .. } => format!("arc:{arc}"),
        BranchConstraint::NumberOfVehicles { graph_id, .. } => format!("vehicles:{graph_id}"),
        BranchConstraint::PackingSetAssignment {
            vertex, graph_id, ..
        } => format!("assignment:{vertex}:{graph_id}"),
        BranchConstraint::RyanFoster(c) => {
            format!("rf:{}:{}", c.first.min(c.second), c.first.max(c.second))
        }
        BranchConstraint::ClusterBoundary { vertices, .. } => {
            let mut v = vertices.clone();
            v.sort_unstable();
            format!("cluster:{v:?}")
        }
        BranchConstraint::ClusterPair { first, second, .. } => {
            let mut a = first.clone();
            let mut b = second.clone();
            a.sort_unstable();
            b.sort_unstable();
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            format!("pair:{a:?}:{b:?}")
        }
        BranchConstraint::AccumulatedResource(c) => format!(
            "resource:{}:{}:{:016x}",
            c.arc,
            c.resource,
            if c.threshold == 0. {
                0
            } else {
                c.threshold.to_bits()
            }
        ),
    }
}
impl BranchEvaluationHistory {
    pub(super) fn record(
        &mut self,
        candidate: &BranchCandidate,
        side: usize,
        depth: usize,
        kind: EvaluationKind,
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
        self.evaluations.push(BranchEvaluation {
            constraint: if side == 0 {
                candidate.down.clone()
            } else {
                candidate.up.clone()
            },
            depth,
            kind,
            parent_bound: parent,
            estimate,
            distance,
            seconds,
        });
    }
    pub fn score(
        &self,
        candidate: &BranchCandidate,
        options: &BranchHistoryOptions,
    ) -> Option<f64> {
        let costs = self.costs.get(&key(&candidate.down))?;
        let stats = costs
            .iter()
            .rev()
            .find(|(_, sides)| {
                sides
                    .iter()
                    .all(|s| s.count >= options.minimum_observations)
            })?
            .1;
        let fraction = candidate.value - candidate.value.floor();
        Some((stats[0].mean * fraction).max(1e-6) * (stats[1].mean * (1. - fraction)).max(1e-6))
    }
    /// At non-root nodes reserve half the first-stage budget for reliable
    /// pseudo-costs, filling the remainder in the supplied fractional order.
    pub(super) fn prioritize(
        &mut self,
        candidates: &mut Vec<BranchCandidate>,
        limit: usize,
        options: &BranchHistoryOptions,
    ) {
        let mut ranked: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| self.score(c, options).map(|s| (i, s)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate((limit / 2).min(candidates.len()));
        self.ranked_candidates += ranked.len();
        let mut used = vec![false; candidates.len()];
        let mut next = Vec::with_capacity(candidates.len());
        for (i, _) in ranked {
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
            "{{\"constraint\":{:?},\"depth\":{},\"kind\":\"{:?}\",\"parent_bound\":{},\"estimate\":{},\"distance\":{},\"seconds\":{}}}",
            format!("{:?}", e.constraint), e.depth, e.kind, e.parent_bound, e.estimate, e.distance, e.seconds,
        )).collect::<Vec<_>>().join(",")
    }
}
