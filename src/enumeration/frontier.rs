//! Observations from one enumeration cost epoch, not a lower-bound certificate.

/// Cached scores observed when enumeration rejected candidates. Values describe
/// the graph/costs of that run and are not repriced when retained paths change.
/// Completion, representative, preexisting-grid, resource and branch exclusions
/// require separate proofs before these observations can certify an original domain.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PruningFrontier {
    pub terminal_minimum: Option<f64>,
    pub terminal_rejections: usize,
    pub join_minimum: Option<f64>,
    pub join_rejections: usize,
    pub ordered_join_minimum: Option<f64>,
    pub ordered_join_breaks: usize,
    /// The completion callback returned false without exposing a numeric floor.
    pub completion_rejections: usize,
    /// Label/path dominance may discard equivalent representatives in this run.
    pub representative_pruning_enabled: bool,
    /// This run first transformed asymmetric costs for consumption joins.
    pub symmetry_transformed: bool,
    /// Rejected scores that cannot supply a finite numeric frontier.
    pub nonfinite_scores: usize,
}
impl PruningFrontier {
    pub(super) fn for_run(preserve_arc_paths: bool) -> Self {
        Self {
            representative_pruning_enabled: !preserve_arc_paths,
            ..Self::default()
        }
    }
    fn observe(minimum: &mut Option<f64>, nonfinite: &mut usize, score: f64) {
        if score.is_finite() {
            *minimum = Some(minimum.map_or(score, |previous| previous.min(score)));
        } else {
            *nonfinite += 1;
        }
    }
    pub(super) fn terminal(&mut self, score: f64) {
        self.terminal_rejections += 1;
        Self::observe(
            &mut self.terminal_minimum,
            &mut self.nonfinite_scores,
            score,
        );
    }
    pub(super) fn joined(&mut self, score: f64) {
        self.join_rejections += 1;
        Self::observe(&mut self.join_minimum, &mut self.nonfinite_scores, score);
    }
    pub(super) fn ordered(&mut self, score: f64) {
        self.ordered_join_breaks += 1;
        Self::observe(
            &mut self.ordered_join_minimum,
            &mut self.nonfinite_scores,
            score,
        );
    }
}
