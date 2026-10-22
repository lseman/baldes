//! Fenchel separation for the unbounded integer packing row sum(q*z_q)<=B.
//! An auxiliary LP searches monotone nonnegative load coefficients. Knapsack
//! DP supplies missing pattern constraints. Every returned integer table is
//! checked by an independent integer-profit DP, including future loads.
#[cfg(feature = "clp")]
use super::KnapsackKind;
use super::RouteLoadKnapsackCut;
use crate::{invalid, Result};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
pub struct FenchelParameters {
    pub max_capacity: usize,
    pub max_load_types: usize,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub coefficient_scale: i32,
    /// Relative to max(1, integer cut RHS), after exact validity repair.
    pub violation_tolerance: f64,
}
impl Default for FenchelParameters {
    fn default() -> Self {
        Self {
            max_capacity: 4096,
            max_load_types: 128,
            max_iterations: 128,
            time_limit: Duration::from_secs(2),
            coefficient_scale: 10_000,
            violation_tolerance: 1e-6,
        }
    }
}
impl FenchelParameters {
    pub fn validate(&self) -> Result<()> {
        if self.max_capacity == 0
            || self.max_capacity > i32::MAX as usize / 8
            || self.max_load_types == 0
            || self.coefficient_scale <= 0
            || (self.max_capacity as u64).saturating_mul(self.coefficient_scale as u64)
                > i32::MAX as u64 / 8
            || !self.violation_tolerance.is_finite()
            || self.violation_tolerance < 0.0
        {
            return Err(invalid(
                "invalid Fenchel separation limits, scale or tolerance",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenchelStatus {
    Separated,
    /// No representable violated cut found; numerical and quantization
    /// tolerances apply. This is not an exact polyhedral membership proof.
    NoCutFound,
    DomainLimit,
    IterationLimit,
    TimeLimit,
    LpStopped(crate::master::LpSolveOutcome),
}

#[derive(Clone, Debug)]
pub struct FenchelResult {
    pub status: FenchelStatus,
    pub cut: Option<RouteLoadKnapsackCut>,
    pub iterations: usize,
    pub patterns: usize,
}
#[cfg(feature = "clp")]
impl FenchelResult {
    fn stopped(status: FenchelStatus, iterations: usize, patterns: usize) -> Self {
        Self {
            status,
            cut: None,
            iterations,
            patterns,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FenchelSeparator {
    pub parameters: FenchelParameters,
}
impl FenchelSeparator {
    pub fn with_parameters(parameters: FenchelParameters) -> Result<Self> {
        parameters.validate()?;
        Ok(Self { parameters })
    }

    pub fn separate(&self, loads: &[i32], values: &[f64], rhs: i32) -> Result<FenchelResult> {
        if loads.len() != values.len() {
            return Err(invalid("Fenchel load/value dimensions"));
        }
        let mut totals = BTreeMap::new();
        for (&load, &value) in loads.iter().zip(values) {
            if load < 0 || !value.is_finite() || value < 0.0 {
                return Err(invalid(
                    "Fenchel requires nonnegative finite loads and values",
                ));
            }
            let total = totals.entry(load).or_insert(0.0);
            *total += value;
            if !total.is_finite() {
                return Err(invalid("Fenchel aggregate value overflow"));
            }
        }
        self.separate_totals(&totals, rhs)
    }

    #[cfg(not(feature = "clp"))]
    pub(super) fn separate_totals(&self, _: &BTreeMap<i32, f64>, _: i32) -> Result<FenchelResult> {
        Err(invalid("Fenchel separation requires the clp feature"))
    }

    #[cfg(feature = "clp")]
    pub(super) fn separate_totals(
        &self,
        totals: &BTreeMap<i32, f64>,
        rhs: i32,
    ) -> Result<FenchelResult> {
        use crate::master::{clp::ClpMaster, LpMaster, LpSolveOutcome, RowSense};
        use std::collections::BTreeSet;
        use std::time::Instant;
        self.parameters.validate()?;
        if rhs < 0
            || totals
                .iter()
                .any(|(&q, &x)| q < 0 || !x.is_finite() || x < 0.0)
        {
            return Err(invalid("invalid Fenchel packing row"));
        }
        let options = &self.parameters;
        let magnitude: f64 = totals.iter().filter(|(q, _)| **q > 0).map(|(_, x)| x).sum();
        if !(magnitude * f64::from(options.coefficient_scale)).is_finite() {
            return Err(invalid("Fenchel scaled activity overflow"));
        }
        if options.time_limit.is_zero() {
            return Ok(FenchelResult::stopped(FenchelStatus::TimeLimit, 0, 0));
        }
        // A positive integer quantity at any load>B is impossible. Unlike
        // simply clamping to B, this handles overweight future columns too.
        let oversized: f64 = totals
            .range((std::ops::Bound::Excluded(rhs), std::ops::Bound::Unbounded))
            .map(|(_, x)| x)
            .sum();
        if oversized > options.violation_tolerance {
            let threshold = rhs
                .checked_add(1)
                .ok_or_else(|| invalid("Fenchel capacity overflow"))?;
            return Ok(FenchelResult {
                status: FenchelStatus::Separated,
                iterations: 0,
                patterns: 0,
                cut: Some(RouteLoadKnapsackCut {
                    kind: KnapsackKind::Fenchel,
                    numerator: 0,
                    denominator: 0,
                    rhs: 0,
                    lhs: oversized,
                    violation: oversized,
                    thresholds: vec![],
                    coefficients: BTreeMap::from([(threshold, 1)]),
                }),
            });
        }
        let weights: Vec<_> = totals
            .iter()
            .filter_map(|(&q, &x)| (q > 0 && q <= rhs && x > 0.0).then_some(q as usize))
            .collect();
        if weights.is_empty() {
            return Ok(FenchelResult::stopped(FenchelStatus::NoCutFound, 0, 0));
        }
        if rhs as usize > options.max_capacity || weights.len() > options.max_load_types {
            return Ok(FenchelResult::stopped(FenchelStatus::DomainLimit, 0, 0));
        }
        let capacity = rhs as usize;
        let started = Instant::now();
        let mut lp = ClpMaster::new()?;
        lp.set_tolerances(1e-9, 1e-9)?;
        for _ in 1..weights.len() {
            lp.add_row(RowSense::LessEqual, 0.0, &[])?;
        }
        for (i, &w) in weights.iter().enumerate() {
            let mut entries = Vec::new();
            if i > 0 {
                entries.push((i - 1, -1.0));
            }
            if i + 1 < weights.len() {
                entries.push((i, 1.0));
            }
            let column = lp.add_column(-totals[&(w as i32)], &entries)?;
            // Pure repeated-item packing constraints, available from the start.
            lp.set_upper(column, 1.0 / (capacity / w) as f64)?;
        }
        let mut patterns = BTreeSet::new();
        for iteration in 0..options.max_iterations {
            let budget = options.time_limit.saturating_sub(started.elapsed());
            if budget.is_zero() {
                return Ok(FenchelResult::stopped(
                    FenchelStatus::TimeLimit,
                    iteration,
                    patterns.len(),
                ));
            }
            let outcome = lp.solve_with_time_limit(budget)?;
            if outcome != LpSolveOutcome::Optimal {
                return Ok(FenchelResult::stopped(
                    if outcome == LpSolveOutcome::TimeLimit {
                        FenchelStatus::TimeLimit
                    } else {
                        FenchelStatus::LpStopped(outcome)
                    },
                    iteration + 1,
                    patterns.len(),
                ));
            }
            if lp.values().len() != weights.len() || lp.values().iter().any(|v| !v.is_finite()) {
                return Err(crate::incomplete(
                    "Fenchel auxiliary LP returned invalid coefficients",
                ));
            }
            let mut alpha = Vec::new();
            let mut previous = 0.0_f64;
            for v in lp.values() {
                previous = previous.max(v.clamp(0.0, 1.0));
                alpha.push(previous);
            }
            let (activity, pattern) = maximum_pattern(&weights, &alpha, capacity);
            if activity <= 1.0 + 1e-8 {
                // The recovered auxiliary model is solved before extracting
                // its duals. A repaired but suboptimal intermediate table is
                // valid, but does not reproduce that separation objective.
                let objective: f64 = weights
                    .iter()
                    .zip(&alpha)
                    .map(|(&q, &a)| totals[&(q as i32)] * a)
                    .sum();
                // Native extraction is gated by auxiliary LP > 1.000001.
                let cut = if objective > 1.0 + options.violation_tolerance {
                    certify(&weights, &alpha, totals, capacity, options)
                } else {
                    None
                };
                return Ok(FenchelResult {
                    status: if cut.is_some() {
                        FenchelStatus::Separated
                    } else {
                        FenchelStatus::NoCutFound
                    },
                    cut,
                    iterations: iteration + 1,
                    patterns: patterns.len(),
                });
            }
            if !patterns.insert(pattern.clone()) {
                return Err(crate::incomplete(
                    "Fenchel auxiliary LP repeated a violated pattern",
                ));
            }
            let entries: Vec<_> = pattern
                .into_iter()
                .enumerate()
                .filter_map(|(i, count)| (count > 0).then_some((i, count as f64)))
                .collect();
            lp.add_row(RowSense::LessEqual, 1.0, &entries)?;
        }
        Ok(FenchelResult::stopped(
            FenchelStatus::IterationLimit,
            options.max_iterations,
            patterns.len(),
        ))
    }
}

#[cfg(feature = "clp")]
fn maximum_pattern(weights: &[usize], profits: &[f64], capacity: usize) -> (f64, Vec<usize>) {
    let mut best = vec![0.0; capacity + 1];
    let mut predecessor = vec![None; capacity + 1];
    for load in 1..=capacity {
        best[load] = best[load - 1];
        for (i, &weight) in weights.iter().enumerate() {
            if weight > load {
                break;
            }
            let score = best[load - weight] + profits[i];
            if score > best[load] {
                best[load] = score;
                predecessor[load] = Some(i);
            }
        }
    }
    let mut counts = vec![0; weights.len()];
    let mut load = capacity;
    while load > 0 {
        if let Some(i) = predecessor[load] {
            counts[i] += 1;
            load -= weights[i];
        } else {
            load -= 1;
        }
    }
    (best[capacity], counts)
}

#[cfg(feature = "clp")]
fn certify(
    weights: &[usize],
    alpha: &[f64],
    totals: &BTreeMap<i32, f64>,
    capacity: usize,
    options: &FenchelParameters,
) -> Option<RouteLoadKnapsackCut> {
    let mut coefficients: Vec<_> = alpha
        .iter()
        .map(|v| (v * options.coefficient_scale as f64).floor() as i32)
        .collect();
    // Integer DP computes the exact support function for this coefficient
    // vector, so validity never depends on LP optimality or float rounding.
    let mut best = vec![0_i64; capacity + 1];
    for load in 1..=capacity {
        best[load] = best[load - 1];
        for (&w, &profit) in weights.iter().zip(&coefficients) {
            if w > load {
                break;
            }
            best[load] = best[load].max(best[load - w] + i64::from(profit));
        }
    }
    let mut rhs = i32::try_from(best[capacity]).ok()?;
    fn gcd(mut a: i32, mut b: i32) -> i32 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let divisor = coefficients.iter().fold(rhs, |g, &c| gcd(g, c)).max(1);
    rhs /= divisor;
    let mut table = BTreeMap::new();
    let mut previous = 0;
    for (&weight, c) in weights.iter().zip(&mut coefficients) {
        *c /= divisor;
        if *c != previous {
            table.insert(weight as i32, *c);
            previous = *c;
        }
    }
    let mut cut = RouteLoadKnapsackCut {
        kind: KnapsackKind::Fenchel,
        numerator: 0,
        denominator: 0,
        rhs,
        lhs: 0.0,
        violation: 0.0,
        thresholds: vec![],
        coefficients: table,
    };
    // For an unseen load q, the table uses a supported weight w<=q. Replacing
    // each q in any feasible integer packing by w remains feasible, so the
    // same integer DP bound proves the row for every future load, not just
    // the observed pool. Zero loads have coefficient zero.
    cut.lhs = totals
        .iter()
        .map(|(&q, &x)| x * cut.coefficient(q) as f64)
        .sum();
    cut.violation = cut.lhs - f64::from(rhs);
    (cut.violation > options.violation_tolerance * f64::from(rhs.max(1))).then_some(cut)
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/knapsack/fenchel/tests.rs"]
mod tests;
