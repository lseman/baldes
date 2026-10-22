/// Preserve byte/level order and each floating-point addition/subtraction.
/// Packing cuts commonly have one or two coefficient levels; specialize those
/// cases so bitmap words and lookup-table bounds are checked outside that loop.
#[inline]
pub(super) fn score(
    tables: &[Vec<f64>],
    increased: &[Vec<u64>],
    decreased: &[Vec<u64>],
    nbytes: usize,
    max_rhs: i32,
) -> f64 {
    match max_rhs {
        1 => small::<1>(tables, increased, decreased, nbytes),
        2 => small::<2>(tables, increased, decreased, nbytes),
        _ => general(tables, increased, decreased, nbytes, max_rhs),
    }
}

#[inline]
fn small<const LEVELS: usize>(
    tables: &[Vec<f64>],
    increased: &[Vec<u64>],
    decreased: &[Vec<u64>],
    nbytes: usize,
) -> f64 {
    if nbytes == 0 {
        return 0.;
    }
    let nwords = nbytes / 8 + usize::from(nbytes % 8 != 0);
    let first_in = &increased[1][..nwords];
    let first_out = &decreased[1][..nwords];
    let second_in = if LEVELS == 2 {
        &increased[2][..nwords]
    } else {
        &[]
    };
    let second_out = if LEVELS == 2 {
        &decreased[2][..nwords]
    } else {
        &[]
    };
    let tables = &tables[..nbytes];
    let mut gain = 0.;
    let mut byte = 0;
    for word in 0..nwords {
        let (a, b) = (first_in[word], first_out[word]);
        let (c, d) = if LEVELS == 2 {
            (second_in[word], second_out[word])
        } else {
            (0, 0)
        };
        for j in 0..(nbytes - byte).min(8) {
            // Construction always provides 256 entries. Masked byte indices
            // then need no individual dynamic-length checks at each level.
            let table = &tables[byte][..256];
            let shift = 8 * j;
            let x1 = table[((a >> shift) & 0xff) as usize] * 1.;
            let x2 = 1. * table[((b >> shift) & 0xff) as usize];
            gain += x1;
            gain -= x2;
            if LEVELS == 2 {
                let x1 = table[((c >> shift) & 0xff) as usize] * 2.;
                let x2 = 2. * table[((d >> shift) & 0xff) as usize];
                gain += x1;
                gain -= x2;
            }
            byte += 1;
        }
    }
    gain
}

#[inline]
fn general(
    tables: &[Vec<f64>],
    increased: &[Vec<u64>],
    decreased: &[Vec<u64>],
    nbytes: usize,
    max_rhs: i32,
) -> f64 {
    let nwords = nbytes / 8 + usize::from(nbytes % 8 != 0);
    let mut gain = 0.;
    let mut byte = 0;
    for word in 0..nwords {
        let mut j = 0;
        while j < 8 && byte < nbytes {
            if max_rhs > 0 {
                for level in 1..=max_rhs as usize {
                    let coefficient = level as f64;
                    let x1 = tables[byte][((increased[level][word] >> (8 * j)) & 0xff) as usize]
                        * coefficient;
                    let x2 = coefficient
                        * tables[byte][((decreased[level][word] >> (8 * j)) & 0xff) as usize];
                    gain += x1;
                    gain -= x2;
                }
            }
            byte += 1;
            j += 1;
        }
    }
    gain
}

#[cfg(test)]
#[path = "../../../../tests/unit/cuts/rank_one/local_search/gain_tests.rs"]
mod tests;
