//! Exact integer row ranges over canonical residual customer partitions.
use super::{CanonicalCoverRelaxation, ConservativeCoverRelaxation, RootProblem, RowSense};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct CoverRowLimits {
    /// Logical DP entries, not a total heap-byte limit.
    pub max_states: usize,
    pub max_work: usize,
    pub deadline: Option<Instant>,
}
impl Default for CoverRowLimits {
    fn default() -> Self {
        Self {
            max_states: 1 << 20,
            max_work: 10_000_000,
            deadline: Instant::now().checked_add(Duration::from_secs(1)),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverRowFailure {
    Dimension,
    StateLimit,
    WorkLimit,
    TimeLimit,
    Allocation,
    Arithmetic,
}
impl std::fmt::Display for CoverRowFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cover row audit: {self:?}")
    }
}
impl std::error::Error for CoverRowFailure {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverRowRange {
    pub minimum: i128,
    pub maximum: i128,
}
impl CoverRowRange {
    /// Exact integer comparison for finite binary64 RHS values of magnitude at
    /// most 2^53. False also means this RHS is outside the supported range.
    pub fn satisfies(&self, sense: RowSense, rhs: f64) -> bool {
        if !rhs.is_finite() || rhs.abs() > (1u64 << 53) as f64 {
            return false;
        }
        match sense {
            RowSense::LessEqual => self.maximum <= rhs.floor() as i128,
            RowSense::GreaterEqual => self.minimum >= rhs.ceil() as i128,
            RowSense::Equal => {
                rhs.fract() == 0. && self.minimum == self.maximum && self.minimum == rhs as i128
            }
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CoverRowAudit {
    /// None means no canonical residual partition exists. It is not an LP
    /// infeasibility or native/tolerant-resource certificate.
    pub range: Option<CoverRowRange>,
    pub work_used: usize,
    pub states: usize,
}

/// An exact additive-weight upper proof, independent of separator metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedCoverRowAudit {
    /// None means some path fails the proposed weight envelope.
    pub upper_bound: Option<i128>,
    pub work_used: usize,
}

impl<const D: usize> ConservativeCoverRelaxation<'_, D> {
    /// Check d*a(path) <= sum of customer weights on every auxiliary path.
    /// In an integer residual cover the weighted visit total is fixed, hence
    /// its integer row activity is <= floor(total/d). Signed weights and
    /// coefficients are allowed; source/sink weights must be zero.
    /// This proves an upper bound only, without enumerating partitions.
    pub fn audit_weighted_upper_row(
        &self,
        coefficients: &[i64],
        weights: &[i64],
        denominator: u64,
        mut limits: CoverRowLimits,
    ) -> Result<WeightedCoverRowAudit, CoverRowFailure> {
        let initial = limits.max_work;
        spend(&mut limits, 0)?;
        let graph = &self.problem().graph;
        if denominator == 0
            || weights.len() != graph.vertices.len()
            || coefficients.len() != self.paths().paths().len()
            || weights[graph.source] != 0
            || weights[graph.sink] != 0
        {
            return Err(CoverRowFailure::Dimension);
        }
        let mut total = 0i128;
        for &(vertex, row) in self.customer_rows() {
            spend(&mut limits, 1)?;
            if self.problem().rows[row].rhs == 1. {
                total = total
                    .checked_add(i128::from(weights[vertex]))
                    .ok_or(CoverRowFailure::Arithmetic)?;
            }
        }
        let d = i128::from(denominator);
        for (path, &coefficient) in self.paths().paths().iter().zip(coefficients) {
            spend(&mut limits, 1)?;
            let mut sum = 0i128;
            for &arc in path.arc_positions() {
                spend(&mut limits, 1)?;
                sum = sum
                    .checked_add(i128::from(weights[graph.arcs[arc].head]))
                    .ok_or(CoverRowFailure::Arithmetic)?;
            }
            if i128::from(coefficient)
                .checked_mul(d)
                .ok_or(CoverRowFailure::Arithmetic)?
                > sum
            {
                return Ok(WeightedCoverRowAudit {
                    upper_bound: None,
                    work_used: initial - limits.max_work,
                });
            }
        }
        spend(&mut limits, 0)?;
        Ok(WeightedCoverRowAudit {
            upper_bound: Some(total.div_euclid(d)),
            work_used: initial - limits.max_work,
        })
    }
}
fn spend(limits: &mut CoverRowLimits, work: usize) -> Result<(), CoverRowFailure> {
    if limits.deadline.is_some_and(|end| Instant::now() >= end) {
        return Err(CoverRowFailure::TimeLimit);
    }
    limits.max_work = limits
        .max_work
        .checked_sub(work)
        .ok_or(CoverRowFailure::WorkLimit)?;
    Ok(())
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), CoverRowFailure> {
    v.try_reserve_exact(n)
        .map_err(|_| CoverRowFailure::Allocation)
}
impl<const D: usize> CanonicalCoverRelaxation<'_, D> {
    /// Extremize the exact sum of one integer coefficient per canonical path
    /// over every partition of the remaining customers. Coefficient order must
    /// match `self.paths().paths()`. Direct empty routes and paths using already
    /// fixed customers are excluded from these integer partitions.
    ///
    /// Fleet limits, branch rules and element restrictions are deliberately
    /// relaxed. A row valid on this superset is valid on its subset; a failed
    /// range test does not prove invalidity on a more constrained routing model.
    /// Schedule-dependent coefficients still need a separate projection proof.
    /// Work/deadline exhaustion returns no partial range. State storage and work
    /// are bounded logically; validation/allocation are cooperative checkpoints.
    pub fn audit_cover_row(
        &self,
        coefficients: &[i64],
        limits: CoverRowLimits,
    ) -> Result<CoverRowAudit, CoverRowFailure> {
        audit_cover_row(
            self.problem(),
            self.customer_rows(),
            self.paths().paths().iter().map(|p| p.arc_positions()),
            coefficients,
            limits,
        )
    }
}
impl<const D: usize> ConservativeCoverRelaxation<'_, D> {
    /// Exact row ranges over the conservative residual path relaxation.
    /// Extra resource-infeasible paths enlarge the partition domain. A passing
    /// range test therefore covers exact-real feasible integer partitions too.
    /// Coefficient order must match `self.paths().paths()`; limits and failure
    /// semantics are the same as `CanonicalCoverRelaxation::audit_cover_row`.
    pub fn audit_cover_row(
        &self,
        coefficients: &[i64],
        limits: CoverRowLimits,
    ) -> Result<CoverRowAudit, CoverRowFailure> {
        audit_cover_row(
            self.problem(),
            self.customer_rows(),
            self.paths().paths().iter().map(|p| p.arc_positions()),
            coefficients,
            limits,
        )
    }
}
fn audit_cover_row<'p, const D: usize>(
    p: &RootProblem<D>,
    customer_rows: &[(usize, usize)],
    paths: impl ExactSizeIterator<Item = &'p [usize]>,
    coefficients: &[i64],
    mut limits: CoverRowLimits,
) -> Result<CoverRowAudit, CoverRowFailure> {
    let initial = limits.max_work;
    spend(&mut limits, 0)?;
    if coefficients.len() != paths.len() {
        return Err(CoverRowFailure::Dimension);
    }
    let g = &p.graph;
    spend(&mut limits, g.vertices.len())?;
    let mut bits = Vec::new();
    reserve(&mut bits, g.vertices.len())?;
    bits.resize(g.vertices.len(), None);
    let mut customers = 0;
    for &(vertex, row) in customer_rows {
        spend(&mut limits, 1)?;
        if p.rows[row].rhs == 1. {
            bits[vertex] = Some(customers);
            customers += 1;
        }
    }
    let states = 1usize
        .checked_shl(customers)
        .filter(|&n| n <= limits.max_states)
        .ok_or(CoverRowFailure::StateLimit)?;
    spend(&mut limits, states)?;
    let mut dp: Vec<Option<CoverRowRange>> = Vec::new();
    reserve(&mut dp, states)?;
    dp.resize(states, None);
    dp[0] = Some(CoverRowRange {
        minimum: 0,
        maximum: 0,
    });
    spend(&mut limits, paths.len())?;
    let mut routes = Vec::new();
    reserve(&mut routes, paths.len())?;
    for (path, &coefficient) in paths.zip(coefficients) {
        let mut mask = 0usize;
        let mut admissible = true;
        for &position in path {
            spend(&mut limits, 1)?;
            let head = g.arcs[position].head;
            if head == g.sink {
                continue;
            }
            match bits[head] {
                Some(bit) => mask |= 1usize << bit,
                None => {
                    admissible = false;
                    break;
                }
            }
        }
        if admissible && mask != 0 {
            routes.push((mask, i128::from(coefficient)));
        }
    }
    for mask in 1..states {
        spend(&mut limits, 1)?;
        let first = 1usize << mask.trailing_zeros();
        let mut range: Option<CoverRowRange> = None;
        for &(support, coefficient) in &routes {
            spend(&mut limits, 1)?;
            if support & first == 0 || support & mask != support {
                continue;
            }
            let Some(rest) = dp[mask ^ support] else {
                continue;
            };
            let minimum = rest
                .minimum
                .checked_add(coefficient)
                .ok_or(CoverRowFailure::Arithmetic)?;
            let maximum = rest
                .maximum
                .checked_add(coefficient)
                .ok_or(CoverRowFailure::Arithmetic)?;
            match &mut range {
                Some(old) => {
                    old.minimum = old.minimum.min(minimum);
                    old.maximum = old.maximum.max(maximum);
                }
                None => range = Some(CoverRowRange { minimum, maximum }),
            }
        }
        dp[mask] = range;
    }
    spend(&mut limits, 0)?;
    Ok(CoverRowAudit {
        range: dp[states - 1],
        work_used: initial - limits.max_work,
        states,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/master/root/cover_rows/tests.rs"]
mod tests;
