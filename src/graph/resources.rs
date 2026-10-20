//! Fixed scalar arrays. Dimension is a const parameter, never a heap fallback.
use std::ops::{Deref, DerefMut};
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct ScalarValues<T, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }>(pub [T; D]);
pub type ResourceValues<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> =
    ScalarValues<f64, D>;
impl<T: Clone, const D: usize> ScalarValues<T, D> {
    pub fn filled(dimensions: usize, value: T) -> Self {
        assert!(
            dimensions <= D,
            "resource count exceeds compiled dimensions"
        );
        Self(std::array::from_fn(|_| value.clone()))
    }
}
impl<T: Clone + Default, const D: usize> ScalarValues<T, D> {
    pub fn from_slice(values: &[T]) -> Self {
        assert!(
            values.len() <= D,
            "resource count exceeds compiled dimensions"
        );
        Self(std::array::from_fn(|i| {
            values.get(i).cloned().unwrap_or_default()
        }))
    }
}
impl<T: Default, const D: usize, const N: usize> From<[T; N]> for ScalarValues<T, D> {
    fn from(values: [T; N]) -> Self {
        assert!(N <= D, "resource count exceeds compiled dimensions");
        let mut values = values.into_iter();
        Self(std::array::from_fn(|_| values.next().unwrap_or_default()))
    }
}
impl<T: Default, const D: usize> From<Vec<T>> for ScalarValues<T, D> {
    fn from(values: Vec<T>) -> Self {
        assert!(
            values.len() <= D,
            "resource count exceeds compiled dimensions"
        );
        let mut values = values.into_iter();
        Self(std::array::from_fn(|_| values.next().unwrap_or_default()))
    }
}
impl<T, const D: usize> Deref for ScalarValues<T, D> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}
impl<T, const D: usize> DerefMut for ScalarValues<T, D> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}
impl<T, const D: usize> AsRef<[T]> for ScalarValues<T, D> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}
impl<T: PartialEq, const D: usize, const N: usize> PartialEq<[T; N]> for ScalarValues<T, D> {
    fn eq(&self, other: &[T; N]) -> bool {
        self.as_ref() == other
    }
}
impl<'a, T, const D: usize> IntoIterator for &'a ScalarValues<T, D> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_const_capacity_uses_only_the_fixed_array() {
        const DIMENSIONS: usize = 37;
        assert_eq!(
            std::mem::size_of::<ResourceValues<DIMENSIONS>>(),
            DIMENSIONS * 8
        );
        let mut graph = crate::Graph::<DIMENSIONS>::new(2, DIMENSIONS).unwrap();
        let arc = graph.add_arc(0, 1, -1.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[DIMENSIONS - 1] = 3.;
        let q = crate::labeling::ScalarResourceKernel::<DIMENSIONS>::extend(
            &graph,
            &graph.arcs[0],
            [0.; DIMENSIONS],
            true,
        )
        .unwrap();
        assert_eq!(q[DIMENSIONS - 1], 3.);
        assert_eq!(graph.initial.len(), DIMENSIONS);
        assert_eq!(graph.arcs[0].bounds.len(), DIMENSIONS);
    }
}
