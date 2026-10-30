//! Ordered machine-front comparison, independent of source pruning proofs.
use super::*;

impl PairedFrontOperations {
    /// Port the complete forward paired-resource dominance wrapper, including
    /// domain containment and the mode-specific ordered front scan. The
    /// result is raw machine behavior, not a source-model pruning proof.
    pub fn machine_forward_dominance(
        left: &PreparedSlopeFunction,
        right: &PreparedSlopeFunction,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let started = Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(remaining_work) = budget.max_operations.checked_sub(4) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let (Some(a0), Some(a1), Some(b0), Some(b1)) = (
            left.pairs().first(),
            left.pairs().last(),
            right.pairs().first(),
            right.pairs().last(),
        ) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward dominance empty front",
            ));
        };
        let starts = match near(a0.1, b0.1) {
            Ok(equal) => equal || a0.1 <= b0.1,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let ends = match near(a1.1, b1.1) {
            Ok(equal) => equal || b1.1 <= a1.1,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if !starts || !ends {
            return FrontierOutcome::Empty;
        }
        let budget = FrontierBudget {
            max_operations: remaining_work,
            max_elapsed: remaining_time,
            ..budget
        };
        match mode {
            4 => match Self::machine_forward_feasibility_dominance(left, right, budget) {
                FrontierOutcome::Complete(()) => FrontierOutcome::Complete(accumulated),
                FrontierOutcome::Empty => FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
            },
            5 | 6 => Self::machine_forward_cost_dominance(left, right, mode, accumulated, budget),
            _ => FrontierOutcome::Unknown(FrontierUnknown::Unsupported("forward dominance mode")),
        }
    }

    /// Ordered feasibility-only machine comparison. Endpoint tolerance and
    /// the interior-start check are literal machine behavior;
    /// acceptance does not establish strict source-domain containment.
    pub fn machine_forward_feasibility_dominance(
        left: &PreparedSlopeFunction,
        right: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        let started = Instant::now();
        let calculate = || -> Result<bool, FrontierUnknown> {
            let (a, b) = (left.pairs(), right.pairs());
            if a.is_empty() || b.is_empty() || a.len() > 12 || b.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "forward feasibility comparison geometry",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            if a.len() + b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if (a.len() + b.len()) * 40 > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let mut av = a[0].0;
            let mut bv = b[0].0;
            let (mut i, mut j) = (1, 1);
            let accepts =
                |x: f64, y: f64| -> Result<bool, FrontierUnknown> { Ok(near(x, y)? || x <= y) };
            if near(a[0].1, b[0].1)? {
                if !accepts(av, bv)? {
                    return Ok(false);
                }
            } else {
                while a
                    .get(i)
                    .ok_or(FrontierUnknown::Unsupported(
                        "forward feasibility exhausted domain",
                    ))?
                    .1
                    < b[0].1
                {
                    av = value(av, a[i].0, finite(a[i].1 - a[i - 1].1)?)?;
                    i += 1;
                }
                if near(a[i].1, b[0].1)? {
                    av = value(av, a[i].0, finite(a[i].1 - a[i - 1].1)?)?;
                    i += 1;
                    if !accepts(av, bv)? {
                        return Ok(false);
                    }
                } else {
                    let interpolated = value(av, a[i].0, finite(b[0].1 - a[i - 1].1)?)?;
                    // The forward scan compares the interpolated left value to the right.
                    if !accepts(interpolated, bv)? {
                        return Ok(false);
                    }
                }
            }
            while j < b.len() {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                let ap = a.get(i).ok_or(FrontierUnknown::Unsupported(
                    "forward feasibility exhausted domain",
                ))?;
                let bp = b[j];
                let (x, y);
                if near(ap.1, bp.1)? {
                    bv = value(bv, bp.0, finite(bp.1 - b[j - 1].1)?)?;
                    av = value(av, ap.0, finite(ap.1 - a[i - 1].1)?)?;
                    (x, y) = (av, bv);
                    i += 1;
                    j += 1;
                } else if ap.1 < bp.1 {
                    y = value(bv, bp.0, finite(ap.1 - b[j - 1].1)?)?;
                    av = value(av, ap.0, finite(ap.1 - a[i - 1].1)?)?;
                    x = av;
                    i += 1;
                } else {
                    x = value(av, ap.0, finite(bp.1 - a[i - 1].1)?)?;
                    bv = value(bv, bp.0, finite(bp.1 - b[j - 1].1)?)?;
                    y = bv;
                    j += 1;
                }
                if !accepts(x, y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match calculate() {
            Ok(_)
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit) =>
            {
                FrontierOutcome::Unknown(FrontierUnknown::Deadline)
            }
            Ok(true) => FrontierOutcome::Complete(()),
            Ok(false) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Compare forward cost fronts and return the updated signed accumulator.
    /// `Empty` means rejection; unsupported geometry/arithmetic returns unknown.
    /// This literal machine operation is not a source feasibility certificate.
    pub fn machine_forward_cost_dominance(
        left: &PreparedSlopeFunction,
        right: &PreparedSlopeFunction,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let started = Instant::now();
        let calculate = || -> Result<Option<f64>, FrontierUnknown> {
            if mode != 5 && mode != 6 {
                return Err(FrontierUnknown::Unsupported("forward cost comparison mode"));
            }
            let a = left.pairs();
            let b = right.pairs();
            if a.is_empty() || b.is_empty() || a.len() > 12 || b.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "forward cost comparison geometry",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            if a.len() + b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if (a.len() + b.len()) * 32 > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let finite = |x: f64| {
                x.is_finite()
                    .then_some(x)
                    .ok_or(FrontierUnknown::Unsupported(
                        "forward cost comparison arithmetic",
                    ))
            };
            finite(accumulated)?;
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut i, mut j) = (1, 1);
            let mut gap;
            if near(a[0].1, b[0].1)? {
                gap = finite(av - bv)?;
            } else {
                while a
                    .get(i)
                    .ok_or(FrontierUnknown::Unsupported(
                        "forward cost comparison exhausted domain",
                    ))?
                    .1
                    < b[0].1
                {
                    av = finite(av + finite(finite(a[i].1 - a[i - 1].1)? * a[i].0)?)?;
                    i += 1;
                }
                if near(a[i].1, b[0].1)? {
                    av = finite(av + finite(finite(a[i].1 - a[i - 1].1)? * a[i].0)?)?;
                    i += 1;
                    gap = finite(av - bv)?;
                } else {
                    gap =
                        finite(finite(finite(finite(b[0].1 - a[i - 1].1)? * a[i].0)? + av)? - bv)?;
                }
            }
            while j < b.len() {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                let ap = a.get(i).ok_or(FrontierUnknown::Unsupported(
                    "forward cost comparison exhausted domain",
                ))?;
                let bp = b[j];
                let candidate;
                if near(ap.1, bp.1)? {
                    av = finite(av + finite(finite(ap.1 - a[i - 1].1)? * ap.0)?)?;
                    bv = finite(bv + finite(finite(bp.1 - b[j - 1].1)? * bp.0)?)?;
                    candidate = finite(av - bv)?;
                    i += 1;
                    j += 1;
                } else if bp.1 <= ap.1 {
                    bv = finite(bv + finite(finite(bp.1 - b[j - 1].1)? * bp.0)?)?;
                    candidate =
                        finite(finite(finite(finite(bp.1 - a[i - 1].1)? * ap.0)? + av)? - bv)?;
                    j += 1;
                } else {
                    av = finite(av + finite(finite(ap.1 - a[i - 1].1)? * ap.0)?)?;
                    candidate =
                        finite(av - finite(finite(finite(ap.1 - b[j - 1].1)? * bp.0)? + bv)?)?;
                    i += 1;
                }
                if candidate > gap {
                    gap = candidate;
                }
            }
            if mode == 6 && !near(gap, 0.)? && gap > 0. {
                return Ok(None);
            }
            Ok(Some(finite(gap + accumulated)?))
        };
        match calculate() {
            Ok(Some(value))
                if !budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit) =>
            {
                FrontierOutcome::Complete(value)
            }
            Ok(Some(_)) => FrontierOutcome::Unknown(FrontierUnknown::Deadline),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
