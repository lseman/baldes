//! Directed slope-function bounds and retained paired-front replay.
use super::*;
use crate::graph::coupled::PreparedSlopeFunction;

pub(super) fn slope_range(
    function: &PreparedSlopeFunction,
    argument: bounds::Range,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bounds::Range> {
    debit(work, function.pairs().len(), deadline)?;
    let (lo, hi) = function.domain();
    let argument = bounds::Range {
        lo: argument.lo.max(lo),
        hi: argument.hi.min(hi),
    };
    if argument.lo > argument.hi {
        return None;
    }
    let (value, mut previous) = function.pairs()[0];
    let mut prefix = bounds::Range::point(value);
    let mut result = (argument.lo == previous).then_some(prefix);
    for &(slope, end) in &function.pairs()[1..] {
        let part = bounds::Range {
            lo: argument.lo.max(previous),
            hi: argument.hi.min(end),
        };
        if part.lo <= part.hi {
            let next = prefix.add(
                part.subtract(bounds::Range::point(previous))
                    .multiply(bounds::Range::point(slope)),
            );
            if !next.finite() {
                return None;
            }
            result = Some(result.map_or(next, |old: bounds::Range| bounds::Range {
                lo: old.lo.min(next.lo),
                hi: old.hi.max(next.hi),
            }));
        }
        if argument.hi <= end {
            break;
        }
        prefix = prefix.add(
            bounds::Range::point(end)
                .subtract(bounds::Range::point(previous))
                .multiply(bounds::Range::point(slope)),
        );
        if !prefix.finite() {
            return None;
        }
        previous = end;
    }
    result
}

impl CoupledKernel {
    pub(super) fn certify_relations(
        &self,
        path: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        debit(work, self.definitions.len(), deadline)?;
        if self.slope_paired.is_empty() {
            return Some((true, [0., 0.]));
        }
        if !self.machine_paired_join_complete() {
            return None;
        }
        let joined = self.joined_machine_path_bounds(path, work, deadline)?;
        if deadline.is_some_and(|end| Instant::now() >= end) {
            return None;
        }
        Some(match joined {
            Some(bounds) => (true, bounds),
            None => (false, [0., 0.]),
        })
    }
}
