//! Sparse symbolic elimination and multifrontal LDLᵀ. Dense storage is local to a front.
use super::parallel::Execution;
use super::{finite, Error, Result, SparseMatrix};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::sync::Arc;

mod workspace;
use workspace::Assembly;
pub(crate) use workspace::FactorWorkspace;
mod dense;
use dense::bk_into;
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
    levels: Vec<Vec<usize>>,
    assembly: Arc<Vec<Assembly>>,
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
            levels: Self::tree_levels(&fronts),
            fronts,
            assembly: Arc::new(Vec::new()),
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
        let mut symbolic = Self {
            order,
            inverse,
            levels: Self::tree_levels(&fronts),
            fronts,
            assembly: Arc::new(Vec::new()),
        };
        symbolic.prepare_assembly(a);
        symbolic
    }

    #[cfg(test)]
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

    fn tree_levels(fronts: &[Front]) -> Vec<Vec<usize>> {
        let mut depth = vec![0; fronts.len()];
        let mut levels: Vec<Vec<usize>> = Vec::new();
        for (s, front) in fronts.iter().enumerate() {
            depth[s] = front
                .children
                .iter()
                .map(|&c| depth[c] + 1)
                .max()
                .unwrap_or(0);
            if levels.len() <= depth[s] {
                levels.resize_with(depth[s] + 1, Vec::new);
            }
            levels[depth[s]].push(s);
        }
        levels
    }

    pub fn assembly_is_missing(&self) -> bool {
        self.assembly.len() != self.fronts.len()
    }

    pub fn parallel_levels(&self, execution: &Execution) -> (bool, bool) {
        // Tiny leaf fronts are cheaper on the up-looking path. Require enough
        // independent arithmetic to amortize assembly and pool scheduling.
        let substantial = |&s: &usize| {
            let f = &self.fronts[s];
            let size = f.end - f.lo + f.updates.len();
            size.saturating_mul(size).saturating_mul(f.end - f.lo) >= 4096
        };
        execution.levels(
            self.levels
                .iter()
                .any(|level| level.iter().filter(|s| substantial(s)).count() >= 4),
            self.fronts
                .iter()
                .any(|f| f.end - f.lo >= 32 && f.end - f.lo + f.updates.len() >= 512),
        )
    }

    pub fn work_estimate(&self) -> usize {
        self.fronts.iter().fold(0usize, |sum, f| {
            let size = f.end - f.lo + f.updates.len();
            sum.saturating_add(size.saturating_mul(size).saturating_mul(f.end - f.lo))
        })
    }

    #[cfg(test)]
    pub fn factorize_with(
        &self,
        a: &SparseMatrix,
        regularize: bool,
        execution: &Execution,
    ) -> Result<Factor> {
        self.factorize_cached(
            a,
            regularize,
            execution,
            &mut FactorWorkspace::default(),
            None,
        )
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

#[cfg(test)]
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

#[cfg(test)]
fn bk(
    f: &mut [f64],
    n: usize,
    np: usize,
    regularize: bool,
    stats: &mut FactorStats,
) -> Result<(Vec<Block>, Vec<usize>)> {
    bk_with_parallel(f, n, np, regularize, stats, false)
}

#[cfg(test)]
fn bk_with_parallel(
    f: &mut [f64],
    n: usize,
    np: usize,
    regularize: bool,
    stats: &mut FactorStats,
    node: bool,
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
            if node && n - k >= 128 {
                frontal_update(f, n, k, [d, 0., 0.], false, true);
            } else {
                for j in k + 1..n {
                    for i in j..n {
                        f[i * n + j] -= f[i * n + k] * d * f[j * n + k];
                        f[j * n + i] = f[i * n + j];
                    }
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
            if node && n - k >= 128 {
                frontal_update(f, n, k, [a, b, c], true, true);
            } else {
                for j in k + 2..n {
                    for i in j..n {
                        f[i * n + j] -= f[i * n + k] * (a * f[j * n + k] + b * f[j * n + k + 1])
                            + f[i * n + k + 1] * (b * f[j * n + k] + c * f[j * n + k + 1]);
                        f[j * n + i] = f[i * n + j];
                    }
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

// Pivot choice stays serial. Disjoint trailing rows share immutable multipliers.
#[cfg(test)]
fn frontal_update(f: &mut [f64], n: usize, k: usize, d: [f64; 3], two: bool, node: bool) {
    let start = k + if two { 2 } else { 1 };
    let multipliers: Vec<_> = (start..n)
        .map(|i| (f[i * n + k], if two { f[i * n + k + 1] } else { 0. }))
        .collect();
    let update = |(offset, row): (usize, &mut [f64])| {
        let i = start + offset;
        let (u, v) = multipliers[offset];
        for j in start..=i {
            let (x, y) = multipliers[j - start];
            row[j] -= if two {
                u * (d[0] * x + d[1] * y) + v * (d[1] * x + d[2] * y)
            } else {
                u * d[0] * x
            };
        }
    };
    if node && n - start >= 128 {
        f[start * n..]
            .par_chunks_mut(n)
            .enumerate()
            .for_each(update);
    } else {
        f[start * n..].chunks_mut(n).enumerate().for_each(update);
    }
    for i in start..n {
        for j in start..i {
            f[j * n + i] = f[i * n + j];
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/ipm/factor/tests.rs"]
mod tests;
