//! Shared nonlinear suffix programs: compose by linking, evaluate at a join.
//! This retains exact ordered operations and avoids exponential growth in
//! piecewise regions when several dependencies are composed.
use std::sync::Arc;

struct Step {
    arc: usize,
    next: Option<Arc<Step>>,
    length: usize,
    fingerprint: u64,
}
impl Drop for Step {
    fn drop(&mut self) {
        let mut next = self.next.take();
        while let Some(step) = next {
            match Arc::try_unwrap(step) {
                Ok(mut step) => next = step.next.take(),
                Err(_) => break,
            }
        }
    }
}
#[derive(Clone, Default)]
pub(super) struct Continuation(Option<Arc<Step>>);
impl Continuation {
    pub(super) fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |step| step.length)
    }
    pub(super) fn prepend(&self, arc: usize) -> Self {
        let length = self
            .0
            .as_ref()
            .map_or(1, |step| step.length.saturating_add(1));
        let tail = self
            .0
            .as_ref()
            .map_or(0x9e3779b97f4a7c15, |step| step.fingerprint);
        let fingerprint = tail.rotate_left(11)
            ^ (arc as u64).wrapping_mul(0x100000001b3)
            ^ (length as u64).rotate_left(29);
        Self(Some(Arc::new(Step {
            arc,
            next: self.0.clone(),
            length,
            fingerprint,
        })))
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut next = self.0.as_deref();
        std::iter::from_fn(move || {
            let step = next?;
            next = step.next.as_deref();
            Some(step.arc)
        })
    }
    /// Preserve operation order and length while allowing the resource owner
    /// to identify equal operators on different physical arcs. Shared tails
    /// need no repeated comparison.
    pub(super) fn equivalent_by(
        &self,
        other: &Self,
        mut same: impl FnMut(usize, usize) -> bool,
    ) -> bool {
        // Operator equivalence may identify different physical arc IDs, so
        // only length can reject here; physical fingerprints cannot.
        if self.0.as_ref().map(|step| step.length) != other.0.as_ref().map(|step| step.length) {
            return false;
        }
        let (mut a, mut b) = (&self.0, &other.0);
        loop {
            match (a, b) {
                (None, None) => return true,
                (Some(left), Some(right)) => {
                    if Arc::ptr_eq(left, right) {
                        return true;
                    }
                    if left.arc != right.arc && !same(left.arc, right.arc) {
                        return false;
                    }
                    a = &left.next;
                    b = &right.next;
                }
                _ => return false,
            }
        }
    }
}
impl PartialEq for Continuation {
    fn eq(&self, other: &Self) -> bool {
        // A mismatch is decisive. Matching fingerprints still require the
        // exact ordered comparison below, including after any hash collision.
        if self.0.as_ref().map(|step| (step.length, step.fingerprint))
            != other.0.as_ref().map(|step| (step.length, step.fingerprint))
        {
            return false;
        }
        let (mut a, mut b) = (&self.0, &other.0);
        loop {
            match (a, b) {
                (None, None) => return true,
                (Some(left), Some(right)) => {
                    if Arc::ptr_eq(left, right) {
                        return true;
                    }
                    if left.arc != right.arc {
                        return false;
                    }
                    a = &left.next;
                    b = &right.next;
                }
                _ => return false,
            }
        }
    }
}
impl std::fmt::Debug for Continuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
