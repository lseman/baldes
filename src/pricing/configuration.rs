//! Pricing options, time budgets and labeling controls.
use super::*;

impl<const D: usize> Solver<D> {
    pub fn set_option(&mut self, name: &str, value: f64) -> Result<()> {
        let name = crate::core::options::Options::canonical_name(name)?;
        // Validate all fallible conversions before publishing any state changes.
        let mut next = self.options.clone();
        next.set(name, value)?;
        let mut controls = self.controls.clone();
        let mut next_phases = None;
        match name {
            "BALDEShardTimeThresholdInPricing" => {
                controls.time_limit = if value <= 0. {
                    Duration::ZERO
                } else {
                    Duration::try_from_secs_f64(value * 50.)
                        .map_err(|_| invalid("pricing time limit"))?
                };
            }
            "BALDESmaxNumOfColsPerExactIteration" => {
                if value < 1. {
                    return Err(invalid("positive exact column limit required"));
                }
                controls.max_columns = value as usize;
            }
            "BALDESheurLabelingStrategy" => {
                next_phases = Some(phases::initialize(value as i32, &[])?)
            }
            "BALDESlabelDominanceStrategy" if value != 0. && value != 1. => {
                return Err(invalid("unsupported label dominance strategy"));
            }
            "BALDESuseBidirectionalSearch" if !(0. ..=2.).contains(&value) => {
                return Err(invalid("unsupported bidirectional mode"));
            }
            "BALDESdynamicBucketSteps" if value != 0. && value != 1. => {
                return Err(invalid("dynamic bucket step mode must be 0 or 1"));
            }
            "BALDEScapacityCutsSeparator" if value == 0. => {
                return Err(invalid(
                    "external CVRPSEP capacity separator is not implemented",
                ));
            }
            "BALDEScapCutsMaxNumPerRound" | "BALDESmaxNumOfColsPerIteration" if value < 1. => {
                return Err(invalid("positive cut/column limit required"));
            }
            "BALDESmaxNumOfEnumSolutionsForMIP" | "BALDESmaxNumOfEnumSolsForEndOfNodeMIP"
                if value < 0. =>
            {
                return Err(invalid("enumerated MIP route limits must be nonnegative"));
            }
            "BALDESmaxNumOfLabelsInEnumeration"
            | "BALDESmaxNumOfLabelsInHeurEnumeration"
            | "BALDESmaxNumOfEnumeratedSolutions"
                if value < 0. =>
            {
                return Err(invalid("enumeration limits must be nonnegative"));
            }
            _ => {}
        }
        let changed = self.options.get(name)? != value;
        self.options = next;
        self.controls = controls;
        if let Some(phases) = next_phases {
            self.phases = phases;
        }
        if name == "BALDESlabelDominanceStrategy" {
            self.dominance_mode = value as i32;
        }
        // Runtime budgets and column limits do not change the bucket graph.
        // Keep inherited fixing when a node installs its own work budget.
        if changed
            && !matches!(
                name,
                "BALDEShardTimeThresholdInPricing"
                    | "BALDESmaxNumOfColsPerExactIteration"
                    | "BALDESmaxNumOfColsPerIteration"
                    | "BALDESheurLabelingStrategy"
            )
        {
            self.engine = None;
        }
        Ok(())
    }
    /// The native hard threshold is scaled by fifty until a rollback point
    /// exists. An explicit caller budget remains an upper limit in either mode.
    pub(super) fn pricing_time_limit(&self) -> Result<Duration> {
        let factor = if self.rollback_point_saved { 1. } else { 50. };
        let seconds = self
            .options
            .get("BALDEShardTimeThresholdInPricing")?
            .max(0.)
            * factor;
        if seconds >= self.controls.time_limit.as_secs_f64() {
            return Ok(self.controls.time_limit);
        }
        Duration::try_from_secs_f64(seconds).map_err(|_| invalid("pricing time limit"))
    }
    pub(super) fn engine_controls(&self, exact: bool) -> EngineControls<D> {
        EngineControls {
            bidirectional: self
                .options
                .get("BALDESuseBidirectionalSearch")
                .unwrap_or(2.)
                != 0.,
            bidirectional_mode: self
                .options
                .get("BALDESuseBidirectionalSearch")
                .unwrap_or(2.) as i32,
            dominance_mode: self.dominance_mode,
            steps: self.bucket_steps,
            target_buckets: self
                .options
                .get("BALDESnumberOfBucketsPerVertex")
                .unwrap_or(25.) as i32,
            group_limit: if exact {
                0
            } else {
                self.options
                    .get("BALDESmaxNumOfColsPerIteration")
                    .unwrap_or(30.) as i32
            },
            filter_other_buckets: self
                .options
                .get("BALDEScheckDominInOtherBuckets")
                .unwrap_or(1.)
                != 0.,
            ..Default::default()
        }
    }
}
