use super::*;

#[test]
fn positive_mod_roundtrips() {
    assert_eq!(CircleSector::positive_mod(0), 0);
    assert_eq!(CircleSector::positive_mod(65535), 65535);
    assert_eq!(CircleSector::positive_mod(65536), 0);
    assert_eq!(CircleSector::positive_mod(-1), 65535);
    assert_eq!(CircleSector::positive_mod(-65536), 0);
}

#[test]
fn single_point_enclosed() {
    let s = CircleSector::from_point(100);
    assert!(s.is_enclosed(100));
    assert!(!s.is_enclosed(101));
}

#[test]
fn extended_sector_encloses() {
    let mut s = CircleSector::from_point(100);
    s.extend(150);
    assert!(s.is_enclosed(125));
    assert!(s.is_enclosed(100));
    assert!(s.is_enclosed(150));
}

#[test]
fn wrap_around_sector() {
    let mut s = CircleSector::from_point(65530);
    s.extend(10);
    // Sector now wraps: [65530, 65535] ∪ [0, 10]
    assert!(s.is_enclosed(65535));
    assert!(s.is_enclosed(5));
    assert!(!s.is_enclosed(20));
}

#[test]
fn overlap_basic() {
    let s1 = CircleSector::from_point(0);
    let mut s2 = CircleSector::from_point(100);
    s2.extend(200);
    assert!(!CircleSector::overlaps(&s1, &s2, 0));
}

#[test]
fn overlap_with_tolerance() {
    // s1 = [0, 45], s2 = [50, 100] → gap = 5, tolerance = 5 → should overlap
    let mut s1 = CircleSector::from_point(0);
    s1.extend(45);
    let mut s2 = CircleSector::from_point(50);
    s2.extend(50);
    assert!(CircleSector::overlaps(&s1, &s2, 5));

    // No tolerance → no overlap (gap is 5)
    assert!(!CircleSector::overlaps(&s1, &s2, 0));
}

#[test]
fn overlap_with_tolerance_wrap_around() {
    // s1 = [65530, 5] (wraps), s2 = [10, 20] → gap from s1.end to s2.start = 5
    let mut s1 = CircleSector::from_point(65530);
    s1.extend(5);
    let mut s2 = CircleSector::from_point(10);
    s2.extend(10);
    assert!(CircleSector::overlaps(&s1, &s2, 5));
}

#[test]
fn overlapping_sectors() {
    // s1 = [0, 100] and s2 = [50, 200] share [50, 100].
    let mut s1 = CircleSector::from_point(0);
    s1.extend(100);
    let mut s2 = CircleSector::from_point(50);
    s2.extend(200);
    assert!(CircleSector::overlaps(&s1, &s2, 0));
    assert!(CircleSector::overlaps(&s2, &s1, 0));
}
