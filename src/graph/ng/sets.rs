//! Temporary ordered element sets used only during NG preparation.
use std::collections::{btree_set, BTreeSet};

pub(super) trait ElementSet: Clone + Default + FromIterator<usize> {
    type Elements<'a>: Iterator<Item = usize>
    where
        Self: 'a;
    fn elements(&self) -> Self::Elements<'_>;
    fn cardinality(&self) -> usize;
    fn extend_set(&mut self, other: &Self);
    fn extend_domain(&mut self, size: usize);
    fn intersect_set(&self, other: &Self) -> Self;
    fn extend_intersection(&mut self, first: &Self, second: &Self);
}

impl ElementSet for BTreeSet<usize> {
    type Elements<'a> = std::iter::Copied<btree_set::Iter<'a, usize>>;
    #[inline]
    fn elements(&self) -> Self::Elements<'_> {
        self.iter().copied()
    }
    #[inline]
    fn cardinality(&self) -> usize {
        self.len()
    }
    #[inline]
    fn extend_set(&mut self, other: &Self) {
        self.extend(other.iter().copied());
    }
    #[inline]
    fn extend_domain(&mut self, size: usize) {
        self.extend(0..size);
    }
    #[inline]
    fn intersect_set(&self, other: &Self) -> Self {
        self.intersection(other).copied().collect()
    }
    #[inline]
    fn extend_intersection(&mut self, first: &Self, second: &Self) {
        self.extend(first.intersection(second).copied());
    }
}

/// The ID domain is bounded by the caller; local NG capacity is independent.
#[derive(Clone, Copy, Default)]
pub(super) struct CompactSet(u128);
impl CompactSet {
    pub(super) const CAPACITY: usize = u128::BITS as usize;
}

impl FromIterator<usize> for CompactSet {
    #[inline]
    fn from_iter<T: IntoIterator<Item = usize>>(values: T) -> Self {
        let mut bits = 0;
        for id in values {
            debug_assert!(id < Self::CAPACITY);
            bits |= 1_u128 << id;
        }
        Self(bits)
    }
}

pub(super) struct Elements(u128);
impl Iterator for Elements {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let id = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(id)
    }
}
impl ElementSet for CompactSet {
    type Elements<'a> = Elements;
    #[inline]
    fn elements(&self) -> Elements {
        Elements(self.0)
    }
    #[inline]
    fn cardinality(&self) -> usize {
        self.0.count_ones() as usize
    }
    #[inline]
    fn extend_set(&mut self, other: &Self) {
        self.0 |= other.0;
    }
    #[inline]
    fn extend_domain(&mut self, size: usize) {
        debug_assert!(size <= Self::CAPACITY);
        self.0 |= if size == Self::CAPACITY {
            u128::MAX
        } else {
            (1_u128 << size) - 1
        };
    }
    #[inline]
    fn intersect_set(&self, other: &Self) -> Self {
        Self(self.0 & other.0)
    }
    #[inline]
    fn extend_intersection(&mut self, first: &Self, second: &Self) {
        self.0 |= first.0 & second.0;
    }
}
