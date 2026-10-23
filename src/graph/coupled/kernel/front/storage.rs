//! Inline common fronts; arbitrary-width functions keep an owned Vec fallback.
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub(super) enum FrontPoints {
    Empty,
    One([f64; 2]),
    Two([[f64; 2]; 2]),
    Heap(Vec<[f64; 2]>),
}
impl FrontPoints {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        if capacity > 4 {
            Self::Heap(Vec::with_capacity(capacity))
        } else {
            Self::Empty
        }
    }
    pub(super) fn push(&mut self, point: [f64; 2]) {
        match self {
            Self::Empty => *self = Self::One(point),
            Self::One(first) => *self = Self::Two([*first, point]),
            Self::Two(pair) => {
                let mut points = Vec::with_capacity(4);
                points.extend_from_slice(pair);
                points.push(point);
                *self = Self::Heap(points);
            }
            Self::Heap(points) => points.push(point),
        }
    }
}
impl From<Vec<[f64; 2]>> for FrontPoints {
    fn from(points: Vec<[f64; 2]>) -> Self {
        match points.as_slice() {
            [] => Self::Empty,
            [one] => Self::One(*one),
            [first, second] => Self::Two([*first, *second]),
            _ => Self::Heap(points),
        }
    }
}
impl FromIterator<[f64; 2]> for FrontPoints {
    fn from_iter<T: IntoIterator<Item = [f64; 2]>>(iter: T) -> Self {
        let mut points = Self::Empty;
        for point in iter {
            points.push(point);
        }
        points
    }
}
impl Deref for FrontPoints {
    type Target = [[f64; 2]];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Empty => &[],
            Self::One(point) => std::slice::from_ref(point),
            Self::Two(points) => points,
            Self::Heap(points) => points,
        }
    }
}
impl DerefMut for FrontPoints {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Empty => &mut [],
            Self::One(point) => std::slice::from_mut(point),
            Self::Two(points) => points,
            Self::Heap(points) => points,
        }
    }
}
impl PartialEq for FrontPoints {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}
impl PartialEq<Vec<[f64; 2]>> for FrontPoints {
    fn eq(&self, other: &Vec<[f64; 2]>) -> bool {
        self[..] == other[..]
    }
}
impl<'a> IntoIterator for &'a FrontPoints {
    type Item = &'a [f64; 2];
    type IntoIter = std::slice::Iter<'a, [f64; 2]>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_points_preserve_exact_values_and_independent_wide_clones() {
        for len in [0, 1, 2, 3, 12, 97] {
            let original: FrontPoints = (0..len).map(|i| [i as f64, -(i as f64)]).collect();
            assert_eq!(matches!(original, FrontPoints::Heap(_)), len > 2);
            assert_eq!(original.len(), len);
            let mut copy = original.clone();
            if len > 0 {
                copy[0][1] = 100.;
                assert_eq!(original[0][1].to_bits(), (-0.0_f64).to_bits());
            }
            let rebuilt = FrontPoints::from(original.to_vec());
            assert_eq!(rebuilt, original);
        }
    }
}
