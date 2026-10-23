//! Shared nonlinear suffix programs: compose by linking, evaluate at a join.
//! This retains exact ordered operations and avoids exponential growth in
//! piecewise regions when several dependencies are composed.
use std::sync::Arc;

struct Step {
    arc: usize,
    next: Option<Arc<Step>>,
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
    pub(super) fn prepend(&self, arc: usize) -> Self {
        Self(Some(Arc::new(Step {
            arc,
            next: self.0.clone(),
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
}
impl PartialEq for Continuation {
    fn eq(&self, other: &Self) -> bool {
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
