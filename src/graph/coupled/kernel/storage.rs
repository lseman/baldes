//! Small fixed-length state vectors. Wide models retain an owned heap fallback.
//! No uninitialized storage, unsafe code or resource-count restriction is used.
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub(super) enum StateValues<T, const N: usize> {
    Inline { len: usize, values: [T; N] },
    Heap(Vec<T>),
}
impl<T: Copy + Default, const N: usize> From<Vec<T>> for StateValues<T, N> {
    fn from(values: Vec<T>) -> Self {
        if values.len() <= N {
            Self::from_slice(&values)
        } else {
            Self::Heap(values)
        }
    }
}
impl<T: Copy + Default, const N: usize> StateValues<T, N> {
    pub(super) fn from_slice(slice: &[T]) -> Self {
        if slice.len() <= N {
            let mut values = [T::default(); N];
            values[..slice.len()].copy_from_slice(slice);
            Self::Inline {
                len: slice.len(),
                values,
            }
        } else {
            Self::Heap(slice.to_vec())
        }
    }
    pub(super) fn filled(len: usize, value: T) -> Self {
        if len <= N {
            Self::Inline {
                len,
                values: [value; N],
            }
        } else {
            Self::Heap(vec![value; len])
        }
    }
    pub(super) fn as_slice(&self) -> &[T] {
        self
    }
}
impl<T, const N: usize> Deref for StateValues<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Inline { len, values } => &values[..*len],
            Self::Heap(values) => values,
        }
    }
}
impl<T, const N: usize> DerefMut for StateValues<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Inline { len, values } => &mut values[..*len],
            Self::Heap(values) => values,
        }
    }
}
impl<'a, T, const N: usize> IntoIterator for &'a StateValues<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<T: PartialEq, const N: usize> PartialEq for StateValues<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/storage/tests.rs"]
mod tests;
