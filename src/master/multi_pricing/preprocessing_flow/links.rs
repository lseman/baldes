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
                let mut rhs = crate::master::root::ExactSum::default();
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
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn exact_shared_link_elimination_is_bounded_and_rejects_rounded_coefficients() {
        let deadline = Instant::now() + Duration::from_secs(10);
        let target = Constraint {
            entries: vec![(0, 1.), (1, 1.)],
            sense: RowSense::GreaterEqual,
            rhs: 1.5,
        };
        let equality = Constraint {
            entries: vec![(0, -1.), (1, -1.), (2, 1.)],
            sense: RowSense::Equal,
            rhs: 0.,
        };
        let (derived, spent, complete) =
            derive(&[target.clone()], &[equality.clone()], 10_000, deadline);
        assert!(complete && spent < 10_000);
        assert!(derived
            .iter()
            .any(|r| r.entries == [(2, 1.)] && r.rhs == 1.5 && r.sense == RowSense::GreaterEqual));
        assert_eq!(
            derive(&[target.clone()], &[equality.clone()], 0, deadline).1,
            0
        );
        assert!(!derive(&[target.clone()], &[equality.clone()], 0, deadline).2);
        assert!(
            !derive(
                &[target.clone()],
                &[equality.clone()],
                10_000,
                Instant::now()
            )
            .2
        );
        let mut rounded = target.clone();
        rounded.rhs = 1e16;
        let mut shifted = equality.clone();
        shifted.rhs = 1.;
        assert!(derive(&[rounded], &[shifted], 10_000, deadline)
            .0
            .is_empty());
        let mut rounded = target;
        rounded.entries[1].1 = 1e16;
        let mut shifted = equality;
        shifted.entries[1].1 = 1.;
        assert!(derive(&[rounded], &[shifted], 10_000, deadline)
            .0
            .is_empty());
    }
}
