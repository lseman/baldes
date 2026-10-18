//! Column-generation round statistics and deferred bucket-step adjustment.
use crate::core::options::Options;
use crate::graph::buckets::BucketGraph;
use crate::{invalid, Result};
#[derive(Clone, Debug, Default)]
pub struct RoundState {
    pub times: Vec<f64>,
    pub dominance_ratios: Vec<f64>,
    pub pending_decrease: bool,
}
impl RoundState {
    pub fn collect(&mut self, time: f64, checks: usize, labels: usize, dynamic: bool) {
        self.times.push(time);
        if dynamic {
            self.dominance_ratios.push(checks as f64 / labels as f64);
        }
    }
    pub fn action(
        &mut self,
        options: &Options,
        grid: Option<&BucketGraph>,
        enumerated: bool,
        end: bool,
        flag3: i32,
        flag5: i32,
    ) -> Result<(bool, bool)> {
        let low = flag3 as u8;
        if end {
            self.times.clear();
            self.dominance_ratios.clear();
            if low != 0 {
                return Ok((false, false));
            }
            let decrease = self.pending_decrease;
            self.pending_decrease = false;
            return Ok((false, decrease));
        }
        let average = self.times.iter().sum::<f64>() / self.times.len() as f64;
        let stop = !enumerated
            && !self.times.is_empty()
            && average > options.get("BALDESstopCutGenTimeThresholdInPricing")?;
        if flag5 == 0 && low != 1 && !self.dominance_ratios.is_empty() {
            let mode = options.get("BALDESdynamicBucketSteps")? as i32;
            if mode != 0 {
                if mode != 1 {
                    return Err(invalid("dynamic bucket step mode must be 0 or 1"));
                }
                let ratio = self.dominance_ratios.iter().sum::<f64>() / self.times.len() as f64;
                let arcs = grid.map_or(0., |g| {
                    let mut count = g
                        .forward
                        .buckets
                        .iter()
                        .flatten()
                        .map(|b| b.arcs.len())
                        .sum::<usize>();
                    if g.bidirectional {
                        count += g
                            .backward
                            .buckets
                            .iter()
                            .flatten()
                            .map(|b| b.arcs.len())
                            .sum::<usize>();
                    }
                    count as f64 / g.order.len() as f64
                });
                if ratio > options.get("BALDESdynBuckStepAdjustRatioThreshold")?
                    && arcs < options.get("BALDESdynBuckStepAdjustNumBuckArcsThreshold")?
                {
                    self.pending_decrease = true;
                }
            }
        }
        Ok((stop, false))
    }
}
