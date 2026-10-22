//! Circle sector arithmetic on a [0, 65536) modular ring.
//!
//! Angles are stored as u16 in [0, 65535] instead of degrees [0, 359] so that
//! modulo operations become cheap bitwise ANDs.  This matches the original
//! HGS implementation and its interval-overlap tricks.
//!
//! See: <https://web.archive.org/web/20200912191950/https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/>

/// A circle sector on a [0, 65536) modular ring.
#[derive(Clone, Copy, Debug, Default)]
pub struct CircleSector {
    /// Angle where the sector starts (inclusive), in [0, 65535].
    pub start: u16,
    /// Angle where the sector ends (inclusive), in [0, 65535].
    pub end: u16,
}

impl CircleSector {
    /// Positive modulo 65536 via bitwise AND — equivalent to `i % 65536` but faster.
    #[inline]
    pub fn positive_mod(i: i32) -> u16 {
        (i & 0xFFFF) as u16
    }

    /// Initialise a sector from a single point.
    #[inline]
    pub fn from_point(point: u16) -> Self {
        Self {
            start: point,
            end: point,
        }
    }

    /// Re-initialize the sector to a single point.
    #[inline]
    pub fn initialize(&mut self, point: u16) {
        self.start = point;
        self.end = point;
    }

    /// Test whether `point` is enclosed within this sector.
    #[inline]
    pub fn is_enclosed(&self, point: u16) -> bool {
        let start_diff = Self::positive_mod(i32::from(point) - i32::from(self.start)) as i32;
        let end_diff = Self::positive_mod(i32::from(self.end) - i32::from(self.start)) as i32;
        start_diff <= end_diff
    }

    /// Test whether two sectors overlap, each widened by `tolerance` units
    /// (HGS CircleSector::overlap): one start lies within the other sector.
    #[inline]
    pub fn overlaps(sector1: &CircleSector, sector2: &CircleSector, tolerance: u16) -> bool {
        let size1 = i32::from(Self::positive_mod(
            i32::from(sector1.end) - i32::from(sector1.start),
        )) + i32::from(tolerance);
        let size2 = i32::from(Self::positive_mod(
            i32::from(sector2.end) - i32::from(sector2.start),
        )) + i32::from(tolerance);
        i32::from(Self::positive_mod(
            i32::from(sector2.start) - i32::from(sector1.start),
        )) <= size1
            || i32::from(Self::positive_mod(
                i32::from(sector1.start) - i32::from(sector2.start),
            )) <= size2
    }

    /// Extend this sector to include `point`, greedily growing the smaller side.
    pub fn extend(&mut self, point: u16) {
        if !self.is_enclosed(point) {
            let extend_end_diff = Self::positive_mod(i32::from(point) - i32::from(self.end)) as i32;
            let extend_start_diff =
                Self::positive_mod(i32::from(self.start) - i32::from(point)) as i32;
            if extend_end_diff <= extend_start_diff {
                self.end = point;
            } else {
                self.start = point;
            }
        }
    }

    /// Convert a polar angle in degrees [0, 359] to the modular-arithmetic space.
    #[inline]
    pub fn from_degrees(degrees: i32) -> u16 {
        Self::positive_mod(32768 * degrees / 180)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/hgs/circle_sector/tests.rs"]
mod tests;
