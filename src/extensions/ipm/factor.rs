//! Sparse symbolic elimination and multifrontal LDLᵀ. Dense storage is local to a front.
use super::{finite, Error, Result, SparseMatrix};
use std::collections::BTreeSet;

mod storage;
use storage::LowerFactor;
mod uplooking;
pub(crate) use uplooking::Uplooking;

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
        // If even the minimum degree reaches every remaining vertex, the
        // remaining graph is complete. Elimination cannot add fill and every
        // degree stays tied, so the existing tie-break visits IDs in order.
        // Dense routing rows otherwise cause cubic redundant tree insertions.
        if graph[k].len() == n - order.len() - 1 {
            order.extend((0..n).filter(|&i| active[i]));
            break;
        }
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

// For dense intersection graphs, packed adjacency avoids allocating a tree
// node per edge and updates an entire word of fill edges at once. Numeric
// matrices and factors remain sparse. Bound this workspace to 32 MiB.
fn column_order(rows: &[Vec<usize>], n: usize) -> Vec<usize> {
    if (64..=16_384).contains(&n) && rows.iter().any(|r| r.len() >= n / 8) {
        let words = (n + 63) / 64;
        let mut graph = vec![vec![0u64; words]; n];
        let mut mask = vec![0u64; words];
        for row in rows {
            mask.fill(0);
            for &j in row {
                mask[j / 64] |= 1 << (j % 64);
            }
            for &j in row {
                for (dst, &src) in graph[j].iter_mut().zip(&mask) {
                    *dst |= src;
                }
            }
        }
        for (j, row) in graph.iter_mut().enumerate() {
            row[j / 64] &= !(1 << (j % 64));
        }
        minimum_degree_bitset(graph)
    } else {
        let mut graph = vec![BTreeSet::new(); n];
        for row in rows {
            for (p, &i) in row.iter().enumerate() {
                for &j in &row[p + 1..] {
                    graph[i].insert(j);
                    graph[j].insert(i);
                }
            }
        }
        minimum_degree(graph)
    }
}

fn minimum_degree_bitset(mut graph: Vec<Vec<u64>>) -> Vec<usize> {
    let n = graph.len();
    let mut active = vec![true; n];
    let mut degree: Vec<usize> = graph
        .iter()
        .map(|row| row.iter().map(|w| w.count_ones() as usize).sum())
        .collect();
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        let k = (0..n)
            .filter(|&i| active[i])
            .min_by_key(|&i| (degree[i], i))
            .unwrap();
        if degree[k] == n - order.len() - 1 {
            order.extend((0..n).filter(|&i| active[i]));
            break;
        }
        let neighbors = graph[k].clone();
        for (word, &bits) in neighbors.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let i = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                degree[i] = 0;
                for (w, (dst, &src)) in graph[i].iter_mut().zip(&neighbors).enumerate() {
                    *dst |= src;
                    if w == i / 64 {
                        *dst &= !(1 << (i % 64));
                    }
                    if w == k / 64 {
                        *dst &= !(1 << (k % 64));
                    }
                    degree[i] += dst.count_ones() as usize;
                }
            }
        }
        graph[k].fill(0);
        active[k] = false;
        order.push(k);
    }
    order
}

pub(crate) fn block_order(a: &SparseMatrix) -> Vec<usize> {
    // The augmented KKT primal block is diagonal: primal vertices have no
    // edges to each other, and eliminating one cannot create such edges.
    // Their order therefore cannot change fill. Do not build/order the
    // potentially huge column-intersection graph of A for this block.
    let columns: Vec<Vec<_>> = (0..a.cols)
        .map(|j| a.column(j).map(|(i, _)| i).collect())
        .collect();
    let mut order: Vec<_> = (0..a.cols).collect();
    // Eliminating the primal block connects rows sharing a column. Preserve
    // exact minimum-degree ordering on that dual Schur-complement graph.
    order.extend(
        column_order(&columns, a.rows)
            .into_iter()
            .map(|i| a.cols + i),
    );
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
    /// Append independent primal vertices when their row cliques are already
    /// covered by the filled dual graph. Keep the old dual order and fronts;
    /// a missing edge requires fresh symbolic analysis.
    pub fn extend_primal(&self, old: &SparseMatrix, a: &SparseMatrix) -> Option<Self> {
        let before = old.cols;
        let added = a.cols.checked_sub(before)?;
        if added == 0
            || a.rows != old.rows
            || a.ptr[..=before] != old.ptr
            || a.idx[..old.nnz()] != old.idx
            || self.order[..before] != (0..before).collect::<Vec<_>>()
            || self.fronts.len() < before
            || self.fronts[..before]
                .iter()
                .enumerate()
                .any(|(i, f)| f.lo != i || f.end != i + 1)
        {
            return None;
        }
        let mut old_front = vec![0; self.order.len()];
        for (s, front) in self.fronts.iter().enumerate() {
            old_front[front.lo..front.end].fill(s);
        }
        let mut appended = Vec::with_capacity(added);
        for j in before..a.cols {
            let mut updates: Vec<_> = a.column(j).map(|(i, _)| self.inverse[before + i]).collect();
            updates.sort_unstable();
            for (k, &i) in updates.iter().enumerate() {
                let front = &self.fronts[old_front[i]];
                for &later in &updates[k + 1..] {
                    if later >= front.end && front.updates.binary_search(&later).is_err() {
                        return None;
                    }
                }
            }
            appended.push(Front {
                lo: j,
                end: j + 1,
                updates: updates.into_iter().map(|i| i + added).collect(),
                children: vec![],
            });
        }
        let mut fronts = Vec::with_capacity(self.fronts.len() + added);
        for front in &self.fronts[..before] {
            let mut front = front.clone();
            for i in &mut front.updates {
                *i += added;
            }
            front.children.clear();
            fronts.push(front);
        }
        fronts.extend(appended);
        for front in &self.fronts[before..] {
            fronts.push(Front {
                lo: front.lo + added,
                end: front.end + added,
                updates: front.updates.iter().map(|i| i + added).collect(),
                children: vec![],
            });
        }
        let mut order: Vec<_> = (0..a.cols).collect();
        order.extend(self.order[before..].iter().map(|i| i + added));
        let mut inverse = vec![0; order.len()];
        let mut col_front = vec![0; order.len()];
        for (k, &i) in order.iter().enumerate() {
            inverse[i] = k;
        }
        for (s, front) in fronts.iter().enumerate() {
            col_front[front.lo..front.end].fill(s);
        }
        for s in 0..fronts.len() {
            if let Some(&p) = fronts[s].updates.first() {
                fronts[col_front[p]].children.push(s);
            }
        }
        Some(Self {
            order,
            inverse,
            fronts,
        })
    }

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
        let mut row_position: Vec<_> = (0..n).collect();
        let mut local = vec![usize::MAX; n];
        let mut schur: Vec<Option<Vec<f64>>> = vec![None; self.fronts.len()];
        let mut stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..FactorStats::default()
        };
        // Fronts are processed sequentially. Reuse their dense assembly
        // workspace instead of allocating one for every eliminated front.
        let mut f = Vec::new();
        for (s, front) in self.fronts.iter().enumerate() {
            let np = front.end - front.lo;
            let nf = np + front.updates.len();
            f.clear();
            f.resize(nf * nf, 0.0);
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
                // Child updates contain only the lower triangle, packed by
                // row. Row-wise assembly keeps both source and destination
                // accesses local; each cell still accumulates children in the
                // same order as before.
                for i in 0..nc {
                    let r = local[rows[i]];
                    let offset = i * (i + 1) / 2;
                    for j in 0..=i {
                        let c = local[rows[j]];
                        f[r * nf + c] += values[offset + j];
                    }
                }
            }
            for j in 0..nf {
                for i in j + 1..nf {
                    f[j * nf + i] = f[i * nf + j];
                }
            }
            let (d, perm) = bk(&mut f, nf, np, regularize, &mut stats)?;
            for (k, &old) in perm.iter().enumerate() {
                order[front.lo + k] = self.order[front.lo + old];
                row_position[front.lo + old] = front.lo + k;
            }
            for k in 0..np {
                blocks[front.lo + k] = d[k];
                let first = k + if matches!(d[k], Block::Two(..)) { 2 } else { 1 };
                for i in first..nf {
                    let v = f[i * nf + k];
                    if v != 0.0 {
                        l[front.lo + k].push((
                            if i < np {
                                front.lo + perm[i]
                            } else {
                                front.updates[i - np]
                            },
                            v,
                        ));
                    }
                }
            }
            let nu = nf - np;
            // Parents only consume the lower triangle. Avoid allocating and
            // copying the unused upper half of every child Schur complement.
            let mut update = Vec::with_capacity(nu * (nu + 1) / 2);
            for i in 0..nu {
                let start = (np + i) * nf + np;
                update.extend_from_slice(&f[start..start + i + 1]);
            }
            schur[s] = Some(update);
            for i in front.lo..front.end {
                local[i] = usize::MAX;
            }
            for &i in &front.updates {
                local[i] = usize::MAX;
            }
        }
        for col in &mut l {
            // Rows remain in symbolic coordinates during assembly. Apply all
            // local pivot permutations once, instead of rescanning the entire
            // existing factor after every front (quadratic on sparse systems).
            for (i, _) in col.iter_mut() {
                *i = row_position[*i];
            }
            col.sort_by_key(|e| e.0);
        }
        if n == 0 {
            stats.min_abs_pivot = 0.0;
        }
        Ok(Factor {
            l: LowerFactor::Columns(l),
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
    l: LowerFactor,
    blocks: Vec<Block>,
    order: Vec<usize>,
    pub stats: FactorStats,
}

impl Factor {
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        let mut work = Vec::new();
        self.solve_into(rhs, &mut out, &mut work)?;
        Ok(out)
    }

    pub fn solve_into(&self, rhs: &[f64], out: &mut Vec<f64>, x: &mut Vec<f64>) -> Result<()> {
        let n = self.order.len();
        if rhs.len() != n || !finite(rhs) {
            return Err(Error::InvalidInput("invalid linear solve RHS".into()));
        }
        x.resize(n, 0.);
        for (k, &i) in self.order.iter().enumerate() {
            x[k] = rhs[i];
        }
        self.l.forward(x);
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
        self.l.backward(x);
        out.resize(n, 0.);
        for (k, &i) in self.order.iter().enumerate() {
            out[i] = x[k];
        }
        if !finite(out) {
            return Err(Error::NumericalFailure(
                "non-finite triangular solution".into(),
            ));
        }
        Ok(())
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
    fn independent_primal_order_preserves_fill_and_kkt_solution() {
        let (m, n) = (9, 80);
        let mut entries = Vec::new();
        for j in 0..n {
            for i in 0..m {
                if j == 0 || i == j % m || i == (3 * j + 1) % m {
                    entries.push((i, j, 0.25 + ((i + j) % 5) as f64 * 0.2));
                }
            }
        }
        let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
        let mut rows = vec![vec![]; m];
        for &(i, j, _) in &entries {
            rows[i].push(j);
        }
        let order = block_order(&a);
        assert_eq!(order[..n], (0..n).collect::<Vec<_>>());
        let mut previous = column_order(&rows, n);
        assert_ne!(previous, order[..n]);
        previous.extend_from_slice(&order[n..]);
        let mut kkt: Vec<_> = (0..n).map(|j| (j, j, -1. - j as f64 / n as f64)).collect();
        kkt.extend((0..m).map(|i| (n + i, n + i, 1.)));
        for &(i, j, value) in &entries {
            kkt.push((n + i, j, value));
            kkt.push((j, n + i, value));
        }
        let kkt = SparseMatrix::from_triplets(n + m, n + m, &kkt).unwrap();
        let current = Symbolic::analyze(&kkt, order, n);
        let previous = Symbolic::analyze(&kkt, previous, n);
        let fill = |symbolic: &Symbolic| {
            symbolic
                .fronts
                .iter()
                .map(|f| {
                    let np = f.end - f.lo;
                    np * (np - 1) / 2 + np * f.updates.len()
                })
                .sum::<usize>()
        };
        assert_eq!(fill(&current), fill(&previous));
        let expected: Vec<_> = (0..n + m).map(|i| (i % 11) as f64 / 11. - 0.25).collect();
        let rhs = kkt.mul(&expected);
        for symbolic in [current, previous] {
            let sparse = Uplooking::new(&symbolic, n).unwrap();
            for factor in [
                symbolic.factorize(&kkt, true).unwrap(),
                sparse.factorize(&symbolic, &kkt).unwrap(),
            ] {
                let solution = factor.solve(&rhs).unwrap();
                for (x, wanted) in solution.iter().zip(&expected) {
                    assert!((x - wanted).abs() < 1e-10);
                }
                for (value, wanted) in kkt.mul(&solution).iter().zip(&rhs) {
                    assert!((value - wanted).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn minimum_degree_matches_explicit_elimination() {
        // Exhaust all undirected five-vertex graphs, including graphs that
        // become complete only after fill, degree ties and isolated vertices.
        fn reference(mut graph: Vec<BTreeSet<usize>>) -> Vec<usize> {
            let n = graph.len();
            let mut active = vec![true; n];
            let mut order = Vec::new();
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
        for n in 0..=5 {
            let pairs: Vec<_> = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .collect();
            for mask in 0..1usize << pairs.len() {
                let mut graph = vec![BTreeSet::new(); n];
                for (bit, &(i, j)) in pairs.iter().enumerate() {
                    if mask & (1 << bit) != 0 {
                        graph[i].insert(j);
                        graph[j].insert(i);
                    }
                }
                let mut packed = vec![vec![0u64; (n + 63) / 64]; n];
                for (i, row) in graph.iter().enumerate() {
                    for &j in row {
                        packed[i][j / 64] |= 1 << (j % 64);
                    }
                }
                assert_eq!(
                    minimum_degree_bitset(packed),
                    reference(graph.clone()),
                    "packed n={n} mask={mask}"
                );
                assert_eq!(
                    minimum_degree(graph.clone()),
                    reference(graph),
                    "n={n} mask={mask}"
                );
            }
        }
    }

    #[test]
    fn column_order_matches_explicit_graph_across_word_boundaries() {
        for n in [63, 64, 65, 127, 129, 257] {
            // Dense overlapping rows and disconnected columns exercise
            // fill, complete tails, and masks spanning multiple words.
            for stride in [2, 3, 7] {
                let rows: Vec<Vec<usize>> = (0..9)
                    .map(|r| {
                        (0..n)
                            .filter(|&j| (j + r * 13) % (stride + r % 3) == 0 && j % 17 != 1)
                            .collect()
                    })
                    .collect();
                let mut graph = vec![BTreeSet::new(); n];
                for row in &rows {
                    for &i in row {
                        for &j in row {
                            if i != j {
                                graph[i].insert(j);
                            }
                        }
                    }
                }
                assert_eq!(
                    column_order(&rows, n),
                    minimum_degree(graph),
                    "n={n} stride={stride}"
                );
            }
        }
    }

    #[test]
    fn pivot_permutations_relabel_rows_from_earlier_fronts() {
        // Column 0 is a separate front with an update into the pivoted front.
        // A disconnected component also exercises reuse of the local row map.
        for (diagonal, coupling) in [(0.21, 1.), (0.2, 2.)] {
            let a = SparseMatrix::from_triplets(
                6,
                6,
                &[
                    (0, 0, 5.),
                    (1, 1, diagonal),
                    (2, 2, 10.),
                    (3, 3, 3.),
                    (0, 1, 1.),
                    (1, 0, 1.),
                    (1, 2, coupling),
                    (2, 1, coupling),
                    (1, 3, 0.2),
                    (3, 1, 0.2),
                    (2, 3, 2.),
                    (3, 2, 2.),
                    (4, 4, 2.),
                    (5, 5, -3.),
                    (4, 5, 0.5),
                    (5, 4, 0.5),
                ],
            )
            .unwrap();
            let symbolic = Symbolic::analyze(&a, (0..6).collect(), 3);
            assert_eq!(symbolic.fronts[0].end, 1);
            let factor = symbolic.factorize(&a, false).unwrap();
            assert_ne!(
                factor.order, symbolic.order,
                "fixture must interchange pivots"
            );
            let expected = [1., -2., 3., -4., 5., -6.];
            let rhs = a.mul(&expected);
            let result = factor.solve(&rhs).unwrap();
            for (x, e) in result.iter().zip(expected) {
                assert!((x - e).abs() < 1e-10, "{result:?}");
            }
        }
    }

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
