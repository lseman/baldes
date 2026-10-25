//! Ordering hints never carry pricing certificates across dual points.
use super::*;

/// Network visitation order within a pricing batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkPricingSchedule {
    /// Visit every enabled network in input order.
    #[default]
    FullSweep,
    /// Resume after the last visited network and return on new improvement.
    Cyclic,
    /// Visit unseen networks first, then most negative recent minima first.
    /// Return on new improvement. Scores only affect order, never bounds.
    Sorted,
    /// Native strategy 2: retain promising networks before unvisited primary
    /// networks, then unvisited secondary networks before unpromising ones.
    /// Normal solved modern pricing results are classified as unpromising.
    NativePromisingFirst,
    /// Native strategy 3: retain unvisited primary networks before promising
    /// networks; the secondary queue follows the same rule as strategy 2.
    NativeUnvisitedFirst,
}

impl NetworkPricingSchedule {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "full-sweep" => Ok(Self::FullSweep),
            "cyclic" => Ok(Self::Cyclic),
            "sorted" => Ok(Self::Sorted),
            "native-promising-first" => Ok(Self::NativePromisingFirst),
            "native-unvisited-first" => Ok(Self::NativeUnvisitedFirst),
            _ => Err(invalid(
                "network pricing schedule must be full-sweep, cyclic, sorted, native-promising-first or native-unvisited-first",
            )),
        }
    }
}

pub(super) struct Scheduler {
    pub policy: NetworkPricingSchedule,
    pub tolerance: f64,
    /// Stop after this many normal solved networks once a new improving
    /// column has appeared in the current batch. Does not limit empty sweeps.
    pub max_unpromising: usize,
    primary: Vec<usize>,
    secondary: Vec<usize>,
    promising: Vec<usize>,
    unpromising: Vec<usize>,
    found_columns: bool,
    unpromising_seen: usize,
    cursor: usize,
    phase: Option<usize>,
    scores: Vec<Option<f64>>,
    /// Coverage of the last batch at its actual dual point, not cached scores.
    pub complete: bool,
}

impl Scheduler {
    pub fn new(policy: NetworkPricingSchedule, networks: usize, tolerance: f64) -> Self {
        Self {
            policy,
            tolerance,
            max_unpromising: 1,
            primary: (0..networks).collect(),
            secondary: vec![],
            promising: vec![],
            unpromising: vec![],
            found_columns: false,
            unpromising_seen: 0,
            cursor: 0,
            phase: None,
            scores: vec![None; networks],
            complete: false,
        }
    }

    /// Preserve scheduling configuration while starting a fresh node scan.
    pub fn fresh(&self) -> Self {
        let mut fresh = Self::new(self.policy, self.scores.len(), self.tolerance);
        fresh.max_unpromising = self.max_unpromising;
        fresh
    }

    pub fn order(&mut self, phase: usize, mut active: Vec<usize>) -> Vec<usize> {
        self.complete = false;
        self.settle_queues();
        self.found_columns = false;
        self.unpromising_seen = 0;
        if self.phase != Some(phase) {
            self.phase = Some(phase);
            self.cursor = 0;
            self.scores.fill(None);
        }
        match self.policy {
            NetworkPricingSchedule::FullSweep => {}
            NetworkPricingSchedule::Cyclic => {
                let first = active.partition_point(|&index| index < self.cursor);
                active.rotate_left(first);
            }
            NetworkPricingSchedule::Sorted => active.sort_by(|&a, &b| {
                match (self.scores[a], self.scores[b]) {
                    (None, None) => std::cmp::Ordering::Equal,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (Some(a), Some(b)) => a.total_cmp(&b),
                }
                .then_with(|| a.cmp(&b))
            }),
            NetworkPricingSchedule::NativePromisingFirst
            | NetworkPricingSchedule::NativeUnvisitedFirst => {
                self.primary.retain(|i| active.contains(i));
                self.secondary.retain(|i| active.contains(i));
                for &index in &active {
                    if !self.primary.contains(&index) && !self.secondary.contains(&index) {
                        self.primary.push(index);
                    }
                }
                active = self
                    .primary
                    .iter()
                    .chain(&self.secondary)
                    .copied()
                    .collect();
            }
        }
        active
    }

    fn settle_queues(&mut self) {
        if self.policy == NetworkPricingSchedule::NativePromisingFirst {
            self.promising.append(&mut self.primary);
            std::mem::swap(&mut self.primary, &mut self.promising);
        } else {
            self.primary.append(&mut self.promising);
        }
        self.secondary.append(&mut self.unpromising);
    }

    /// Solved modern native pricing returns status zero even when it adds
    /// columns. Positive native status is a distinct, unobserved classifier;
    /// do not infer it from a negative minimum or from the number of columns.
    pub fn finish_network(&mut self, index: usize, new_improving_columns: usize) -> bool {
        self.finish_classified(index, new_improving_columns, false)
    }

    fn finish_classified(&mut self, index: usize, new_columns: usize, promising: bool) -> bool {
        match self.policy {
            NetworkPricingSchedule::FullSweep => false,
            NetworkPricingSchedule::Cyclic | NetworkPricingSchedule::Sorted => new_columns > 0,
            NetworkPricingSchedule::NativePromisingFirst
            | NetworkPricingSchedule::NativeUnvisitedFirst => {
                self.primary.retain(|&i| i != index);
                self.secondary.retain(|&i| i != index);
                self.found_columns |= new_columns > 0;
                if promising {
                    self.promising.push(index);
                    // The native default promising budget is one million;
                    // normal solved modern results never take this branch.
                    self.promising.len() >= 1_000_000
                } else {
                    self.unpromising.push(index);
                    if self.found_columns {
                        self.unpromising_seen += 1;
                    }
                    self.found_columns && self.unpromising_seen >= self.max_unpromising
                }
            }
        }
    }

    pub fn record(&mut self, index: usize, minimum: f64) {
        self.scores[index] = Some(minimum);
        self.cursor = (index + 1) % self.scores.len();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/scheduling_tests.rs"]
mod tests;
