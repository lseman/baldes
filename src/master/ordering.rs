//! Explicit collection algorithms for tolerant comparisons.
use std::time::Instant;

pub(crate) struct Work {
    deadline: Option<Instant>,
    operations: usize,
}
impl Work {
    pub fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            operations: 0,
        }
    }
    pub fn alive(&self) -> bool {
        self.deadline.is_none_or(|d| Instant::now() < d)
    }
    pub fn step(&mut self) -> Option<()> {
        self.operations = self.operations.wrapping_add(1);
        (self.operations & 63 != 0 || self.alive()).then_some(())
    }
    pub(crate) fn compare(
        &mut self,
        a: usize,
        b: usize,
        before: &impl Fn(usize, usize) -> bool,
    ) -> Option<bool> {
        self.step()?;
        Some(before(a, b))
    }
}

fn insertion(
    order: &mut [usize],
    before: &impl Fn(usize, usize) -> bool,
    work: &mut Work,
) -> Option<()> {
    for i in 1..order.len() {
        let value = order[i];
        if work.compare(value, order[0], before)? {
            order.copy_within(0..i, 1);
            order[0] = value;
        } else {
            let mut j = i;
            while j > 0 && work.compare(value, order[j - 1], before)? {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = value;
        }
    }
    Some(())
}

fn merge(
    left: &[usize],
    right: &[usize],
    before: &impl Fn(usize, usize) -> bool,
    work: &mut Work,
) -> Option<Vec<usize>> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let (mut a, mut b) = (0, 0);
    while a < left.len() && b < right.len() {
        if work.compare(right[b], left[a], before)? {
            result.push(right[b]);
            b += 1;
        } else {
            result.push(left[a]);
            a += 1;
        }
    }
    result.extend_from_slice(&left[a..]);
    result.extend_from_slice(&right[b..]);
    Some(result)
}

fn merge_chunks(
    order: &[usize],
    width: usize,
    before: &impl Fn(usize, usize) -> bool,
    work: &mut Work,
) -> Option<Vec<usize>> {
    let mut output = Vec::with_capacity(order.len());
    for chunk in order.chunks(width.saturating_mul(2)) {
        let middle = width.min(chunk.len());
        output.extend(merge(&chunk[..middle], &chunk[middle..], before, work)?);
    }
    Some(output)
}

fn buffered(
    order: &mut [usize],
    before: &impl Fn(usize, usize) -> bool,
    work: &mut Work,
) -> Option<()> {
    for chunk in order.chunks_mut(7) {
        insertion(chunk, before, work)?;
    }
    let mut width = 7;
    while width < order.len() {
        let buffer = merge_chunks(order, width, before, work)?;
        width = width.saturating_mul(2);
        order.copy_from_slice(&merge_chunks(&buffer, width, before, work)?);
        width = width.saturating_mul(2);
    }
    Some(())
}

/// Full-buffer stable_sort_adaptive: ceil-half split, insertion chunks of seven,
/// buffered merge passes, then forward/backward final merge by shorter side.
/// Comparisons need not satisfy Rust's total-order contract.
pub(crate) fn stable(
    order: &mut [usize],
    before: &impl Fn(usize, usize) -> bool,
    work: &mut Work,
) -> Option<()> {
    if !work.alive() {
        return None;
    }
    let middle = order.len().div_ceil(2);
    let (left, right) = order.split_at_mut(middle);
    buffered(left, before, work)?;
    buffered(right, before, work)?;
    let output = if left.len() <= right.len() {
        merge(left, right, before, work)?
    } else {
        let mut output = vec![0; left.len() + right.len()];
        let (mut a, mut b, mut to) = (left.len(), right.len(), output.len());
        while a > 0 && b > 0 {
            to -= 1;
            if work.compare(right[b - 1], left[a - 1], before)? {
                a -= 1;
                output[to] = left[a];
            } else {
                b -= 1;
                output[to] = right[b];
            }
        }
        output[..a].copy_from_slice(&left[..a]);
        output[a..a + b].copy_from_slice(&right[..b]);
        output
    };
    if !work.alive() {
        return None;
    }
    order.copy_from_slice(&output);
    Some(())
}
