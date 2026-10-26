//! Bounded exact row reductions through original shared linear equalities.
use super::*;

pub(super) fn derive(
    targets: &[Constraint],
    equalities: &[Constraint],
    budget: usize,
    deadline: Instant,
) -> (Vec<Constraint>, usize, bool) {
    if equalities.is_empty() {
        return (vec![], 0, true);
    }
    let key = |r: &Constraint| {
        (
            u8::from(r.sense == RowSense::GreaterEqual),
            r.rhs.to_bits(),
            r.entries
                .iter()
                .map(|&(v, a)| (v, a.to_bits()))
                .collect::<Vec<_>>(),
        )
    };
    let mut seen = HashSet::new();
    let mut derived: Vec<Constraint> = vec![];
    let mut work = 0usize;
    let mut cursor = 0;
    while cursor < targets.len() + derived.len() {
        // Only support-reducing combinations are retained. Equalities are
        // original exact rows, not nonlinear lower/upper envelopes.
        let source = if cursor < targets.len() {
            &targets[cursor]
        } else {
            &derived[cursor - targets.len()]
        };
        let copy_work = source.entries.len().saturating_add(1);
        if copy_work > budget - work || Instant::now() >= deadline {
            return (derived, work, false);
        }
        work += copy_work;
        let target = source.clone();
        for equality in equalities {
            for sign in [1., -1.] {
                let cost = target
                    .entries
                    .len()
                    .saturating_add(equality.entries.len())
                    .saturating_add(3);
                if cost > budget - work || Instant::now() >= deadline {
                    return (derived, work, false);
                }
                work += cost;
                let mut rhs = crate::master::numerics::ExactSum::default();
                if rhs.add(target.rhs).is_none() || rhs.add(sign * equality.rhs).is_none() {
                    continue;
                }
                let Some(value) = rhs.upper() else {
                    continue;
                };
                if rhs.add(-value).is_none() || rhs.sign() != 0. {
                    continue;
                }
                let terms = target
                    .entries
                    .iter()
                    .copied()
                    .chain(equality.entries.iter().map(|&(v, a)| (v, sign * a)))
                    .collect();
                let Some(reduced) = super::row(terms, target.sense, value) else {
                    continue;
                };
                if reduced.entries.len() >= target.entries.len()
                    || (reduced.entries.is_empty() && reduced.rhs == 0.)
                {
                    continue;
                }
                if seen.insert(key(&reduced)) {
                    derived.push(reduced);
                }
            }
        }
        cursor += 1;
    }
    (derived, work, true)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/model/preprocessing_flow/links/tests.rs"]
mod tests;
