//! Dense LDLᵀ with scalar restricted BK and optional safe matrix-multiply panels.
//! Pivot selection reads the lower triangle. Blocked updates may write the
//! unused upper part of a row tile; it never contributes to lower entries.
use super::*;

#[derive(Debug, Default)]
pub(super) struct DenseScratch {
    pub(super) used_blocked: bool,
    multipliers: Vec<(f64, f64)>,
    #[cfg(feature = "ipm-blocked")]
    left: Vec<f64>,
    #[cfg(feature = "ipm-blocked")]
    right: Vec<f64>,
}

fn swap_lower(f: &mut [f64], n: usize, a: usize, b: usize, perm: &mut [usize]) {
    if a == b {
        return;
    }
    let (a, b) = (a.min(b), a.max(b));
    for i in 0..a {
        f.swap(a * n + i, b * n + i);
    }
    f.swap(a * n + a, b * n + b);
    for i in a + 1..b {
        f.swap(i * n + a, b * n + i);
    }
    for i in b + 1..n {
        f.swap(i * n + a, i * n + b);
    }
    perm.swap(a, b);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bk_into(
    f: &mut [f64],
    n: usize,
    np: usize,
    regularize: bool,
    stats: &mut FactorStats,
    node: bool,
    blocks: &mut Vec<Block>,
    perm: &mut Vec<usize>,
    scratch: &mut DenseScratch,
    allow_blocked: bool,
) -> Result<()> {
    scratch.used_blocked = false;
    blocks.clear();
    blocks.resize(np, Block::Skip);
    perm.clear();
    perm.extend(0..np);
    // Batch trailing updates outside each panel. A 1x1 pivot that needs
    // interchange or regularization flushes pending work and resumes scalar BK.
    #[cfg(feature = "ipm-blocked")]
    let mut k = if allow_blocked && n >= 128 && np >= 32 {
        scratch.used_blocked = true;
        blocked_prefix(f, n, np, stats, node, blocks, scratch)
    } else {
        0
    };
    #[cfg(not(feature = "ipm-blocked"))]
    let mut k = {
        let _ = allow_blocked;
        0
    };
    let multipliers = &mut scratch.multipliers;
    const ALPHA: f64 = (1.0 + 4.123_105_625_617_661) / 8.0;
    while k < np {
        let mut lambda = 0.0;
        let mut pivot = k;
        for i in k + 1..np {
            if f[i * n + k].abs() > lambda {
                lambda = f[i * n + k].abs();
                pivot = i;
            }
        }
        let abs = f[k * n + k].abs();
        let mut two = false;
        let mut interchange = k;
        if lambda > 0.0 && abs < ALPHA * lambda {
            let sigma = (k..np)
                .filter(|&i| i != pivot)
                .map(|i| f[i.max(pivot) * n + i.min(pivot)].abs())
                .fold(0.0_f64, f64::max);
            if abs * sigma < ALPHA * lambda * lambda {
                if f[pivot * n + pivot].abs() >= ALPHA * sigma {
                    interchange = pivot;
                } else if k + 1 < np {
                    two = true;
                }
            }
        }
        if !two {
            swap_lower(f, n, k, interchange, perm);
            let mut d = f[k * n + k];
            if !d.is_finite() {
                return Err(Error::NumericalFailure("non-finite frontal pivot".into()));
            }
            if d.abs() <= 1e-11 && regularize {
                let mut floor = 1e-10_f64;
                for i in k + 1..n {
                    let diag = f[i * n + i].abs();
                    if diag > 0.0 {
                        floor = floor.max(f[i * n + k] * f[i * n + k] / diag);
                    }
                }
                d = if d < 0.0 { -floor } else { floor };
                stats.perturbed_pivots += 1;
            }
            if d == 0.0 {
                return Err(Error::NumericalFailure("zero frontal pivot".into()));
            }
            blocks[k] = Block::One(d);
            stats.min_abs_pivot = stats.min_abs_pivot.min(d.abs());
            for i in k + 1..n {
                f[i * n + k] /= d;
            }
            update_lower(f, n, k, [d, 0., 0.], false, node, multipliers);
            k += 1;
        } else {
            swap_lower(f, n, k + 1, pivot, perm);
            let a = f[k * n + k];
            let b = f[(k + 1) * n + k];
            let c = f[(k + 1) * n + k + 1];
            let det = a * c - b * b;
            // A singular 2x2 pivot is a numerical failure. Replacing only its
            // determinant would make the multipliers inconsistent with stored D.
            if !det.is_finite() || det == 0.0 {
                return Err(Error::NumericalFailure("singular 2x2 pivot".into()));
            }
            blocks[k] = Block::Two(a, b, c);
            stats.num_2x2 += 1;
            let mid = 0.5 * (a + c);
            let rad = (0.5 * (a - c)).hypot(b);
            stats.min_abs_pivot = stats
                .min_abs_pivot
                .min((mid + rad).abs().min((mid - rad).abs()));
            for i in k + 2..n {
                let u = f[i * n + k];
                let v = f[i * n + k + 1];
                f[i * n + k] = (c * u - b * v) / det;
                f[i * n + k + 1] = (a * v - b * u) / det;
            }
            update_lower(f, n, k, [a, b, c], true, node, multipliers);
            f[(k + 1) * n + k] = 0.0;
            k += 2;
        }
    }
    if (0..n).any(|i| !finite(&f[i * n..i * n + i + 1])) {
        return Err(Error::NumericalFailure(
            "frontal factorization overflow".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "ipm-blocked")]
fn blocked_prefix(
    f: &mut [f64],
    n: usize,
    np: usize,
    stats: &mut FactorStats,
    node: bool,
    blocks: &mut [Block],
    scratch: &mut DenseScratch,
) -> usize {
    const WIDTH: usize = 64;
    const ALPHA: f64 = (1.0 + 4.123_105_625_617_661) / 8.0;
    let mut lo = 0;
    while lo < np {
        let end = (lo + WIDTH).min(np);
        let mut k = lo;
        while k < end {
            let d = f[k * n + k];
            let lambda = (k + 1..np)
                .map(|i| f[i * n + k].abs())
                .fold(0.0_f64, f64::max);
            // This sufficient BK acceptance test never changes pivot policy.
            // Pivot swaps, 2x2 blocks and pivot regularization use scalar BK.
            if !d.is_finite() || d.abs() <= 1e-11 || d.abs() < ALPHA * lambda {
                break;
            }
            blocks[k] = Block::One(d);
            stats.min_abs_pivot = stats.min_abs_pivot.min(d.abs());
            for i in k + 1..n {
                f[i * n + k] /= d;
            }
            let start = k + 1;
            scratch.multipliers.clear();
            scratch
                .multipliers
                .extend((start..n).map(|i| (f[i * n + k], 0.)));
            let multipliers = &scratch.multipliers;
            for (offset, row) in f[start * n..].chunks_mut(n).enumerate() {
                let scale = multipliers[offset].0 * d;
                let target = &mut row[start..end.min(start + offset + 1)];
                for (value, &(entry, _)) in target.iter_mut().zip(multipliers) {
                    *value -= scale * entry;
                }
            }
            k += 1;
        }
        panel_update(f, n, lo, k, end, blocks, node, scratch);
        if k != end {
            return k;
        }
        lo = end;
    }
    np
}

#[cfg(feature = "ipm-blocked")]
#[allow(clippy::too_many_arguments)]
fn panel_update(
    f: &mut [f64],
    n: usize,
    lo: usize,
    done: usize,
    start: usize,
    blocks: &[Block],
    node: bool,
    scratch: &mut DenseScratch,
) {
    use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};
    if done == lo || start == n {
        return;
    }
    let rows = n - start;
    let width = done - lo;
    scratch.left.resize(rows * width, 0.);
    scratch.right.resize(rows * width, 0.);
    for k in lo..done {
        let Block::One(d) = blocks[k] else {
            unreachable!()
        };
        for i in start..n {
            scratch.left[(i - start) * width + k - lo] = f[i * n + k] * d;
            scratch.right[(k - lo) * rows + i - start] = f[i * n + k];
        }
    }
    let (left, right) = (&scratch.left, &scratch.right);
    let update = |(tile, chunk): (usize, &mut [f64])| {
        let offset = tile * 64;
        let height = chunk.len() / n;
        let columns = offset + height;
        let a = ArrayView2::from_shape(
            (height, width),
            &left[offset * width..(offset + height) * width],
        )
        .unwrap();
        let b =
            ArrayView2::from_shape((width, columns).strides((rows, 1)), right.as_slice()).unwrap();
        // The rectangle stops at the last row's diagonal. Only the diagonal
        // tile has extra upper entries; GEMM never mixes output cells, so a
        // poisoned unused upper entry cannot affect a required lower entry.
        let mut c =
            ArrayViewMut2::from_shape((height, columns).strides((n, 1)), &mut chunk[start..])
                .unwrap();
        ndarray::linalg::general_mat_mul(-1., &a, &b, 1., &mut c);
    };
    if node && rows >= 128 {
        f[start * n..]
            .par_chunks_mut(n * 64)
            .enumerate()
            .for_each(update);
    } else {
        f[start * n..]
            .chunks_mut(n * 64)
            .enumerate()
            .for_each(update);
    }
}

fn update_lower(
    f: &mut [f64],
    n: usize,
    k: usize,
    d: [f64; 3],
    two: bool,
    node: bool,
    multipliers: &mut Vec<(f64, f64)>,
) {
    let start = k + if two { 2 } else { 1 };
    multipliers.clear();
    multipliers.extend((start..n).map(|i| (f[i * n + k], if two { f[i * n + k + 1] } else { 0. })));
    let multipliers = multipliers.as_slice();
    let update = |(offset, row): (usize, &mut [f64])| {
        let (u, v) = multipliers[offset];
        let target = &mut row[start..=start + offset];
        if two {
            for (entry, &(x, y)) in target.iter_mut().zip(multipliers) {
                *entry -= u * (d[0] * x + d[1] * y) + v * (d[1] * x + d[2] * y);
            }
        } else {
            let value = u * d[0];
            for (entry, &(x, _)) in target.iter_mut().zip(multipliers) {
                *entry -= value * x;
            }
        }
    };
    if node && n - start >= 256 {
        f[start * n..]
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(32)
            .for_each(update);
    } else {
        f[start * n..].chunks_mut(n).enumerate().for_each(update);
    }
}

#[cfg(all(test, feature = "ipm-blocked"))]
#[path = "../../../../tests/unit/extensions/ipm/factor/blocked_tests.rs"]
mod blocked_tests;
