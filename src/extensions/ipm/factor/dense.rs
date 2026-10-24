//! Lower-triangle dense LDLᵀ. Row updates stream contiguous memory; pivot
//! selection reads the symmetric lower entry instead of maintaining a copy of
//! the upper triangle after each pivot. The reference kernel stays in tests.
use super::*;

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
    multipliers: &mut Vec<(f64, f64)>,
) -> Result<()> {
    blocks.clear();
    blocks.resize(np, Block::Skip);
    perm.clear();
    perm.extend(0..np);
    let mut k = 0;
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
