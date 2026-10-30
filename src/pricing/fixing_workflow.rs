//! Native fixing/enumeration mode dispatch and limits.
use super::*;
impl<const D: usize> Solver<D> {
    pub fn run_fixing_enumeration(
        &mut self,
        mode: i32,
        zero: f64,
        mut cutoff: f64,
    ) -> Result<fixing::FixingResult> {
        if !(-1..=1).contains(&mode) || !zero.is_finite() || !cutoff.is_finite() {
            return Err(invalid("fixing/enumeration arguments"));
        }
        let mut report = fixing::FixingResult::default();
        if self.options.get("BALDESapplyReducedCostFixing")? == 0. {
            return Ok(report);
        }
        self.refresh_cuts()?;
        if let Some(routes) = self.inspect_enumeration(cutoff)? {
            report.state = 3;
            report.routes = routes.len();
            return Ok(report);
        }
        if mode == 0 {
            let gap = self.options.get("BALDESredCostFixingFalseGap")?;
            if gap > 1. || (gap < -1. && self.has_cuts()) {
                cutoff = (cutoff - zero) / gap.abs() + zero;
            }
        }
        // The enumeration concatenation phase has its own limit derived from
        // the preceding pricing work. Capture it before full-label completion
        // resets the engine statistics.
        let (pricing_seconds, previous_concatenation) =
            self.engine.as_ref().map_or((0., 0.), |engine| {
                let stats = &engine.statistics;
                (
                    (stats.forward_time + stats.backward_time) + stats.concatenation_time,
                    stats.concatenation_time,
                )
            });
        let base = if mode == 1 {
            100.
        } else {
            pricing_seconds.max(0.01)
        };
        let ratio = self.options.get("BALDESstopRatioForConcatenationInEnum")?;
        // The reference compares against accumulated concatenation time,
        // including the preceding pricing pass, not just the current phase.
        let remaining = (base * ratio - previous_concatenation).max(0.);
        let concatenation_limit = Duration::try_from_secs_f64(remaining).unwrap_or(Duration::MAX);
        let saved_time = self.controls.time_limit;
        let saved_options = self.options.clone();
        let outcome = (|| {
            if mode == 0 {
                self.controls.time_limit = Duration::try_from_secs_f64(
                    self.options
                        .get("BALDESredCostFixingTimeThreshold")?
                        .max(0.),
                )
                .map_err(|_| invalid("fixing time threshold"))?;
            }
            let fixing_deadline = std::time::Instant::now().checked_add(self.controls.time_limit);
            self.complete_labels(true)?;
            let fixed = self.fix(cutoff)?;
            report.eliminated = fixed.eliminated_forward + fixed.eliminated_backward;
            report.state = 4;
            if mode == -1 {
                return Ok(report);
            }
            let labels = self.options.get(if mode == 0 {
                "BALDESmaxNumOfLabelsInEnumeration"
            } else {
                "BALDESmaxNumOfLabelsInHeurEnumeration"
            })?;
            let paths = self.options.get("BALDESmaxNumOfEnumeratedSolutions")?;
            if labels == 0. || paths == 0. {
                return Ok(report);
            }
            if labels < 0. || paths < 0. {
                return Err(invalid("negative enumeration limit"));
            }
            self.options
                .set("BALDESmaxNumOfLabelsInEnumeration", labels)?;
            // Label completion and enumeration share the fixing call's budget.
            if let Some(end) = fixing_deadline {
                self.controls.time_limit = end.saturating_duration_since(std::time::Instant::now());
            }
            match self.enumerate_with_concatenation_limit(cutoff, Some(concatenation_limit)) {
                Ok(routes) => {
                    report.state = 2;
                    report.routes = routes.len();
                    Ok(report)
                }
                Err(e) if super::universe::is_enumeration_interruption(&e.message)
                    || e.message == "pricing hard time limit exceeded" => {
                    report.state = 1;
                    Ok(report)
                }
                Err(e) => Err(e),
            }
        })();
        self.controls.time_limit = saved_time;
        self.options = saved_options;
        outcome
    }
}
