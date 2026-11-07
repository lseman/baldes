use super::*;

fn reference(an: &[usize], ap: &[usize], bn: &[usize], bp: &[usize]) -> usize {
    let mut count = 0;
    for j in 0..an.len() {
        if an[j] != bn[j] && an[j] != bp[j] {
            count += 1;
        }
        if ap[j] == 0 && bp[j] != 0 && bn[j] != 0 {
            count += 1;
        }
    }
    count
}

#[test]
fn counts_match_reference_for_depots_reversed_arcs_tails_and_high_bits() {
    // Exhaust all combinations of the four endpoint IDs. Include IDs whose
    // upper 32 bits differ, so narrowing comparisons would fail on x86_64.
    let ids = [0, 1, 2, usize::MAX, usize::MAX / 2];
    let (mut an, mut ap, mut bn, mut bp) = (vec![], vec![], vec![], vec![]);
    for a in ids {
        for b in ids {
            for c in ids {
                for d in ids {
                    an.push(a);
                    ap.push(b);
                    bn.push(c);
                    bp.push(d);
                }
            }
        }
    }
    for len in 0..=an.len() {
        // Offset slices exercise unaligned vector loads as well as every tail.
        let (an, ap, bn, bp) = (
            &an[1..len.max(1)],
            &ap[1..len.max(1)],
            &bn[1..len.max(1)],
            &bp[1..len.max(1)],
        );
        let expected = reference(an, ap, bn, bp);
        assert_eq!(broken_pairs_count_scalar(an, ap, bn, bp), expected);
        assert_eq!(broken_pairs_count(an, ap, bn, bp), expected);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature check; slices have matching lengths.
            assert_eq!(unsafe { broken_pairs_count_avx2(an, ap, bn, bp) }, expected);
        }
    }
}

#[test]
fn identical_and_reversed_routes_have_zero_distance() {
    let next = [2, 3, 0, 5, 0];
    let prev = [0, 1, 2, 0, 4];
    assert_eq!(broken_pairs_count(&next, &prev, &next, &prev), 0);
    assert_eq!(broken_pairs_count(&next, &prev, &prev, &next), 0);
    // Same outgoing arcs, but the route start differs: the second condition
    // must still count independently of the outgoing-arc comparison.
    assert_eq!(broken_pairs_count(&[2], &[0], &[2], &[3]), 1);
}

#[test]
#[ignore = "manual fixed-work kernel benchmark"]
fn benchmark_broken_pairs_count() {
    use std::{hint::black_box, time::Instant};
    for n in [100, 330, 1000] {
        let an: Vec<_> = (0..n).map(|i| (i + 1) % 23).collect();
        let ap: Vec<_> = (0..n).map(|i| i % 23).collect();
        let bn: Vec<_> = (0..n)
            .map(|i| if i % 5 == 0 { 0 } else { (i + 1) % 23 })
            .collect();
        let bp: Vec<_> = (0..n).map(|i| (i + 22) % 23).collect();
        for (name, kernel) in [
            (
                "reference",
                reference as fn(&[usize], &[usize], &[usize], &[usize]) -> usize,
            ),
            ("scalar", broken_pairs_count_scalar),
            ("dispatched", broken_pairs_count),
        ] {
            let start = Instant::now();
            let mut sum = 0;
            for _ in 0..2_000_000 {
                sum += kernel(
                    black_box(&an),
                    black_box(&ap),
                    black_box(&bn),
                    black_box(&bp),
                );
            }
            println!(
                "n={n} kernel={name} seconds={:.6} checksum={}",
                start.elapsed().as_secs_f64(),
                black_box(sum)
            );
        }
    }
}
