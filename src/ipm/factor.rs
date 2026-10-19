//! Sparse symbolic elimination and multifrontal LDLᵀ. Dense storage is local to a front.
use super::{finite, Error, Result, SparseMatrix};
use std::collections::BTreeSet;

/// Deterministic minimum-degree ordering. `order[new] = old`.
/// Uses explicit graph updates rather than the C++ approximate-degree quotient graph.
pub(crate) fn minimum_degree(mut graph: Vec<BTreeSet<usize>>) -> Vec<usize> {
    let n = graph.len();
    let mut active = vec![true; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let k = (0..n)
            .filter(|&i| active[i])
            .min_by_key(|&i| (graph[i].len(), i))
            .unwrap();
        let neighbors: Vec<_> = graph[k].iter().copied().collect();
        for &i in &neighbors {
            graph[i].remove(&k);
        }
        for (p, &i) in neighbors.iter().enumerate() {
            for &j in &neighbors[p + 1..] {
                graph[i].insert(j);
                graph[j].insert(i);
            }
        }
        graph[k].clear();
        active[k] = false;
        order.push(k);
    }
    order
}

pub(crate) fn block_order(a: &SparseMatrix) -> Vec<usize> {
    let mut pg = vec![BTreeSet::new(); a.cols];
    let mut dg = vec![BTreeSet::new(); a.rows];
    let mut rows = vec![vec![]; a.rows];
    for j in 0..a.cols {
        let col: Vec<_> = a.column(j).map(|(i, _)| i).collect();
        for &i in &col {
            rows[i].push(j);
        }
        for (p, &i) in col.iter().enumerate() {
            for &k in &col[p + 1..] {
                dg[i].insert(k);
                dg[k].insert(i);
            }
        }
    }
    for row in rows {
        for (p, &i) in row.iter().enumerate() {
            for &j in &row[p + 1..] {
                pg[i].insert(j);
                pg[j].insert(i);
            }
        }
    }
    let mut order = minimum_degree(pg);
    order.extend(minimum_degree(dg).into_iter().map(|i| a.cols + i));
    order
}

#[derive(Debug, Clone)]
struct Front {
    lo: usize,
    end: usize,
    updates: Vec<usize>,
    children: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Symbolic {
    order: Vec<usize>,
    inverse: Vec<usize>,
    fronts: Vec<Front>,
}

impl Symbolic {
    pub fn analyze(a: &SparseMatrix, order: Vec<usize>, boundary: usize) -> Self {
        let n = a.cols;
        let mut inverse = vec![0; n];
        for (k, &i) in order.iter().enumerate() {
            inverse[i] = k;
        }
        let mut graph = vec![BTreeSet::new(); n];
        for j in 0..n {
            for (i, _) in a.column(j) {
                if i != j {
                    let p = inverse[i];
                    let q = inverse[j];
                    graph[p].insert(q);
                    graph[q].insert(p);
                }
            }
        }
        let mut pattern = vec![vec![]; n];
        for k in 0..n {
            let nb: Vec<_> = graph[k].iter().copied().filter(|&i| i > k).collect();
            for (p, &i) in nb.iter().enumerate() {
                graph[i].remove(&k);
                for &j in &nb[p + 1..] {
                    graph[i].insert(j);
                    graph[j].insert(i);
                }
            }
            graph[k].clear();
            pattern[k] = nb;
        }
        let mut fronts = vec![];
        let mut col_front = vec![0; n];
        let mut lo = 0;
        while lo < n {
            let mut end = lo + 1;
            while end < n
                && end != boundary
                && pattern[end - 1].first() == Some(&end)
                && pattern[end - 1][1..] == pattern[end]
            {
                end += 1;
            }
            for x in &mut col_front[lo..end] {
                *x = fronts.len();
            }
            fronts.push(Front {
                lo,
                end,
                updates: pattern[end - 1].clone(),
                children: vec![],
            });
            lo = end;
        }
        // Parent columns always exceed child columns; this order is already a
        // topological order of the elimination forest, without recursion.
        for s in 0..fronts.len() {
            if let Some(&p) = fronts[s].updates.first() {
                let parent = col_front[p];
                fronts[parent].children.push(s);
            }
        }
        Self {
            order,
            inverse,
            fronts,
        }
    }

    pub fn factorize(&self, a: &SparseMatrix, regularize: bool) -> Result<Factor> {
        let n = self.order.len();
        let mut columns = vec![vec![]; n];
        for j in 0..n {
            for (i, v) in a.column(j) {
                let r = self.inverse[i];
                let c = self.inverse[j];
                if r >= c {
                    columns[c].push((r, v));
                }
            }
        }
        let mut l: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        let mut blocks = vec![Block::Skip; n];
        let mut order = self.order.clone();
        let mut schur: Vec<Option<Vec<f64>>> = vec![None; self.fronts.len()];
        let mut stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..FactorStats::default()
        };
        for (s, front) in self.fronts.iter().enumerate() {
            let np = front.end - front.lo;
            let nf = np + front.updates.len();
            let mut f = vec![0.0; nf * nf];
            let mut local = vec![usize::MAX; n];
            for k in 0..np {
                local[front.lo + k] = k;
            }
            for (k, &i) in front.updates.iter().enumerate() {
                local[i] = np + k;
            }
            for j in front.lo..front.end {
                for &(i, v) in &columns[j] {
                    let r = local[i];
                    if r != usize::MAX {
                        f[r * nf + j - front.lo] += v;
                    }
                }
            }
            for &child in &front.children {
                let values = schur[child].take().ok_or_else(|| {
                    Error::NumericalFailure("missing child Schur complement".into())
                })?;
                let rows = &self.fronts[child].updates;
                let nc = rows.len();
                for j in 0..nc {
                    for i in j..nc {
                        let r = local[rows[i]];
                        let c = local[rows[j]];
                        f[r * nf + c] += values[i * nc + j];
                    }
                }
            }
            for j in 0..nf {
                for i in j + 1..nf {
                    f[j * nf + i] = f[i * nf + j];
                }
            }
            let (d, perm) = bk(&mut f, nf, np, regularize, &mut stats)?;
            let mut inv = vec![0; np];
            for (k, &old) in perm.iter().enumerate() {
                order[front.lo + k] = self.order[front.lo + old];
                inv[old] = k;
            }
            for col in &mut l[..front.lo] {
                for (i, _) in col {
                    if *i >= front.lo && *i < front.end {
                        *i = front.lo + inv[*i - front.lo];
                    }
                }
            }
            for k in 0..np {
                blocks[front.lo + k] = d[k];
                let first = k + if matches!(d[k], Block::Two(..)) { 2 } else { 1 };
                for i in first..nf {
                    let v = f[i * nf + k];
                    if v != 0.0 {
                        l[front.lo + k].push((
                            if i < np {
                                front.lo + i
                            } else {
                                front.updates[i - np]
                            },
                            v,
                        ));
                    }
                }
            }
            let nu = nf - np;
            let mut update = vec![0.0; nu * nu];
            for i in 0..nu {
                for j in 0..nu {
                    update[i * nu + j] = f[(np + i) * nf + np + j];
                }
            }
            schur[s] = Some(update);
        }
        for col in &mut l {
            col.sort_by_key(|e| e.0);
        }
        if n == 0 {
            stats.min_abs_pivot = 0.0;
        }
        Ok(Factor {
            l,
            blocks,
            order,
            stats,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum Block {
    One(f64),
    Two(f64, f64, f64),
    Skip,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FactorStats {
    pub perturbed_pivots: usize,
    pub num_2x2: usize,
    pub min_abs_pivot: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Factor {
    l: Vec<Vec<(usize, f64)>>,
    blocks: Vec<Block>,
    order: Vec<usize>,
    pub stats: FactorStats,
}

impl Factor {
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let n = self.order.len();
        if rhs.len() != n || !finite(rhs) {
            return Err(Error::InvalidInput("invalid linear solve RHS".into()));
        }
        let mut x: Vec<_> = self.order.iter().map(|&i| rhs[i]).collect();
        for j in 0..n {
            for &(i, v) in &self.l[j] {
                x[i] -= v * x[j];
            }
        }
        for k in 0..n {
            match self.blocks[k] {
                Block::One(d) => x[k] /= d,
                Block::Two(a, b, c) => {
                    let det = a * c - b * b;
                    let u = x[k];
                    let v = x[k + 1];
                    x[k] = (c * u - b * v) / det;
                    x[k + 1] = (a * v - b * u) / det;
                }
                Block::Skip => {}
            }
        }
        for j in (0..n).rev() {
            for &(i, v) in &self.l[j] {
                x[j] -= v * x[i];
            }
        }
        let mut out = vec![0.0; n];
        for (k, &i) in self.order.iter().enumerate() {
            out[i] = x[k];
        }
        if !finite(&out) {
            return Err(Error::NumericalFailure(
                "non-finite triangular solution".into(),
            ));
        }
        Ok(out)
    }
}

fn swap(f: &mut [f64], n: usize, a: usize, b: usize, perm: &mut [usize]) {
    if a == b {
        return;
    }
    for j in 0..n {
        f.swap(a * n + j, b * n + j);
    }
    for i in 0..n {
        f.swap(i * n + a, i * n + b);
    }
    perm.swap(a, b);
}

fn bk(
    f: &mut [f64],
    n: usize,
    np: usize,
    regularize: bool,
    stats: &mut FactorStats,
) -> Result<(Vec<Block>, Vec<usize>)> {
    let mut blocks = vec![Block::Skip; np];
    let mut perm: Vec<_> = (0..np).collect();
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
                .map(|i| f[i * n + pivot].abs())
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
            swap(f, n, k, interchange, &mut perm);
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
            for j in k + 1..n {
                for i in j..n {
                    f[i * n + j] -= f[i * n + k] * d * f[j * n + k];
                    f[j * n + i] = f[i * n + j];
                }
            }
            k += 1;
        } else {
            swap(f, n, k + 1, pivot, &mut perm);
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
            for j in k + 2..n {
                for i in j..n {
                    f[i * n + j] -= f[i * n + k] * (a * f[j * n + k] + b * f[j * n + k + 1])
                        + f[i * n + k + 1] * (b * f[j * n + k] + c * f[j * n + k + 1]);
                    f[j * n + i] = f[i * n + j];
                }
            }
            f[(k + 1) * n + k] = 0.0;
            k += 2;
        }
    }
    if !finite(f) {
        return Err(Error::NumericalFailure(
            "frontal factorization overflow".into(),
        ));
    }
    Ok((blocks, perm))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_by_two_and_pivot_interchange() {
        for values in [
            vec![0., 2., 1., 2., 0., 3., 1., 3., 4.],
            vec![0.01, 1., 0., 1., 10., 2., 0., 2., 3.],
        ] {
            let entries: Vec<_> = (0..3)
                .flat_map(|i| {
                    let v = &values;
                    (0..3).map(move |j| (i, j, v[3 * i + j]))
                })
                .collect();
            let a = SparseMatrix::from_triplets(3, 3, &entries).unwrap();
            let factor = Symbolic::analyze(&a, vec![0, 1, 2], 3)
                .factorize(&a, false)
                .unwrap();
            let rhs = a.mul(&[1., -2., 3.]);
            let x = factor.solve(&rhs).unwrap();
            for (v, w) in x.iter().zip([1., -2., 3.]) {
                assert!((v - w).abs() < 1e-10, "{x:?}");
            }
        }
    }
}
