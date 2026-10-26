//! Assembly positions belong to the symbolic plan; only numeric buffers change
//! between Newton steps. No workspace or mutable factor is shared by clones.
use super::*;
use std::ops::Range;

mod schedule;

#[derive(Debug, Clone)]
pub(super) struct Assembly {
    /// (offset in the dense front, position in the original CSC values).
    entries: Vec<(usize, usize)>,
    /// Rows of each child's update in the parent front's coordinates.
    children: Vec<Vec<usize>>,
}

impl Symbolic {
    pub(crate) fn dependency_schedule(&self) -> bool {
        let heavy = |&s: &usize| {
            let front = &self.fronts[s];
            let width = front.end - front.lo;
            let size = width + front.updates.len();
            width >= 16 && size.saturating_mul(size).saturating_mul(width) >= 100_000
        };
        // Large numbers of cheap primal leaves make per-front ready-task
        // bookkeeping more expensive than level batching.
        let heavy_count = (0..self.fronts.len()).filter(|s| heavy(s)).count();
        if self.fronts.len() > heavy_count.saturating_mul(8) {
            return false;
        }
        self.levels
            .iter()
            .filter(|level| level.iter().any(heavy))
            .count()
            >= 2
            && self
                .levels
                .iter()
                .any(|level| level.iter().filter(|s| heavy(s)).count() >= 2)
    }

    pub(crate) fn prepare_assembly(&mut self, a: &SparseMatrix) {
        let mut owner = vec![0; self.order.len()];
        for (s, front) in self.fronts.iter().enumerate() {
            owner[front.lo..front.end].fill(s);
        }
        let mut assembly: Vec<_> = self
            .fronts
            .iter()
            .map(|front| {
                let np = front.end - front.lo;
                let position = |i: usize| {
                    if i < front.end {
                        i - front.lo
                    } else {
                        np + front
                            .updates
                            .binary_search(&i)
                            .expect("filled front contains child rows")
                    }
                };
                Assembly {
                    entries: Vec::new(),
                    children: front
                        .children
                        .iter()
                        .map(|&child| {
                            self.fronts[child]
                                .updates
                                .iter()
                                .map(|&i| position(i))
                                .collect()
                        })
                        .collect(),
                }
            })
            .collect();
        for j in 0..a.cols {
            let c = self.inverse[j];
            let s = owner[c];
            let front = &self.fronts[s];
            let np = front.end - front.lo;
            let nf = np + front.updates.len();
            for p in a.ptr[j]..a.ptr[j + 1] {
                let r = self.inverse[a.idx[p]];
                if r < c {
                    continue;
                }
                let row = if r < front.end {
                    r - front.lo
                } else {
                    np + front
                        .updates
                        .binary_search(&r)
                        .expect("filled front contains matrix entries")
                };
                assembly[s].entries.push((row * nf + c - front.lo, p));
            }
        }
        self.assembly = Arc::new(assembly);
    }

    /// Factor using cached maps and numeric scratch. The previous factor is
    /// consumed for its allocations; a failure cannot expose stale factors.
    pub fn factorize_cached(
        &self,
        a: &SparseMatrix,
        regularize: bool,
        execution: &Execution,
        work: &mut FactorWorkspace,
        previous: Option<Factor>,
    ) -> Result<Factor> {
        work.recycle(previous);
        let (tree, node) = self.parallel_levels(execution);
        let workers = if tree {
            execution
                .pool
                .as_ref()
                .map_or(1, |p| p.current_num_threads())
        } else {
            1
        };
        // Ready tasks amortize their state locks only when substantial pivot
        // blocks can overlap across levels. Keep cheap leaves and one heavy
        // root level on the retained, batched schedule.
        if tree && self.dependency_schedule() {
            return self.factorize_dependencies(a, regularize, execution, work, node, workers);
        }
        work.prepare(self, tree, workers);
        let FactorWorkspace {
            tasks,
            levels,
            dense,
            schur,
            valid,
            l,
            blocks,
            order,
            row_position,
            spare,
            ..
        } = work;
        for values in schur.iter_mut() {
            spare.put(std::mem::take(values));
        }
        valid.fill(false);
        order.clone_from(&self.order);
        let mut stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..FactorStats::default()
        };
        let mut run = || -> Result<()> {
            for level in levels.iter() {
                let tasks = &mut tasks[level.clone()];
                if tree && tasks.len() > 1 {
                    for task in tasks.iter_mut() {
                        self.prepare_front(task, schur, l, spare);
                    }
                    // One retained scratch buffer per batch. Node tasks may
                    // use the same pool without holding locks or borrowing
                    // another batch's workspace.
                    let batch = (tasks.len() + workers - 1) / workers;
                    tasks
                        .par_chunks_mut(batch)
                        .zip(dense.par_iter_mut())
                        .try_for_each(|(tasks, scratch)| {
                            for task in tasks {
                                self.compute_front(
                                    a, regularize, node, schur, valid, scratch, task,
                                )?;
                            }
                            Ok::<_, Error>(())
                        })?;
                    for task in tasks {
                        self.commit_front(
                            task,
                            schur,
                            valid,
                            l,
                            blocks,
                            order,
                            row_position,
                            spare,
                            &mut stats,
                        );
                    }
                } else {
                    for task in tasks {
                        self.prepare_front(task, schur, l, spare);
                        self.compute_front(a, regularize, node, schur, valid, &mut dense[0], task)?;
                        self.commit_front(
                            task,
                            schur,
                            valid,
                            l,
                            blocks,
                            order,
                            row_position,
                            spare,
                            &mut stats,
                        );
                    }
                }
            }
            Ok(())
        };
        if let Some(pool) = &execution.pool {
            pool.install(run)?;
        } else {
            run()?;
        }
        for column in l.iter_mut() {
            for (i, _) in column.iter_mut() {
                *i = row_position[*i];
            }
            column.sort_by_key(|e| e.0);
        }
        if self.order.is_empty() {
            stats.min_abs_pivot = 0.;
        }
        Ok(Factor {
            l: LowerFactor::Columns(std::mem::take(l)),
            blocks: std::mem::take(blocks),
            order: std::mem::take(order),
            stats,
        })
    }

    fn prepare_front(
        &self,
        task: &mut FrontWork,
        schur: &mut [Vec<f64>],
        l: &mut [Vec<(usize, f64)>],
        spare: &mut SchurPool,
    ) {
        let front = &self.fronts[task.s];
        spare.put(std::mem::take(&mut task.update));
        // Normally empty; reclaim partially computed values after a failed call.
        spare.put(std::mem::take(&mut schur[task.s]));
        let nu = front.updates.len();
        task.update = spare.take(nu * (nu + 1) / 2);
        for (column, old) in task.l.iter_mut().zip(&mut l[front.lo..front.end]) {
            std::mem::swap(column, old);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_front(
        &self,
        a: &SparseMatrix,
        regularize: bool,
        node: bool,
        schur: &[Vec<f64>],
        valid: &[bool],
        scratch: &mut DenseWorkspace,
        task: &mut FrontWork,
    ) -> Result<()> {
        let front = &self.fronts[task.s];
        self.compute_front_values(
            a,
            regularize,
            node,
            |i| {
                let child = front.children[i];
                valid[child].then_some(schur[child].as_slice())
            },
            scratch,
            task,
        )
    }

    fn compute_front_values<'a>(
        &self,
        a: &SparseMatrix,
        regularize: bool,
        node: bool,
        mut child_values: impl FnMut(usize) -> Option<&'a [f64]>,
        scratch: &mut DenseWorkspace,
        task: &mut FrontWork,
    ) -> Result<()> {
        let front = &self.fronts[task.s];
        let assembly = &self.assembly[task.s];
        let np = front.end - front.lo;
        let nf = np + front.updates.len();
        let f = &mut scratch.matrix;
        let mut assemble = |f: &mut Vec<f64>| -> Result<()> {
            f.resize(nf * nf, 0.);
            for i in 0..nf {
                f[i * nf..i * nf + i + 1].fill(0.);
            }
            for &(offset, p) in &assembly.entries {
                f[offset] += a.val[p];
            }
            for (child, rows) in assembly.children.iter().enumerate() {
                let values = child_values(child).ok_or_else(|| {
                    Error::NumericalFailure("missing child Schur complement".into())
                })?;
                for (i, &r) in rows.iter().enumerate() {
                    let offset = i * (i + 1) / 2;
                    let row = &mut f[r * nf..(r + 1) * nf];
                    for (j, &c) in rows[..=i].iter().enumerate() {
                        row[c] += values[offset + j];
                    }
                }
            }
            Ok(())
        };
        assemble(f)?;
        task.stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..FactorStats::default()
        };
        let result = bk_into(
            f,
            nf,
            np,
            regularize,
            &mut task.stats,
            node,
            &mut task.blocks,
            &mut task.perm,
            &mut scratch.multipliers,
            true,
        );
        if result.is_err() && scratch.multipliers.used_blocked {
            // A batched sum can overflow even when sequential subtractions
            // stay finite. Reassemble from immutable sources and try the
            // original scalar kernel before changing Newton regularization.
            assemble(f)?;
            task.stats = FactorStats {
                min_abs_pivot: f64::INFINITY,
                ..Default::default()
            };
            bk_into(
                f,
                nf,
                np,
                regularize,
                &mut task.stats,
                node,
                &mut task.blocks,
                &mut task.perm,
                &mut scratch.multipliers,
                false,
            )?;
        } else {
            result?;
        }
        for k in 0..np {
            let column = &mut task.l[k];
            column.clear();
            let first = k + if matches!(task.blocks[k], Block::Two(..)) {
                2
            } else {
                1
            };
            for i in first..nf {
                let value = f[i * nf + k];
                if value != 0. {
                    let row = if i < np {
                        front.lo + task.perm[i]
                    } else {
                        front.updates[i - np]
                    };
                    column.push((row, value));
                }
            }
        }
        task.update.clear();
        for i in 0..front.updates.len() {
            let start = (np + i) * nf + np;
            task.update.extend_from_slice(&f[start..start + i + 1]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_front(
        &self,
        task: &mut FrontWork,
        schur: &mut [Vec<f64>],
        valid: &mut [bool],
        l: &mut [Vec<(usize, f64)>],
        blocks: &mut [Block],
        order: &mut [usize],
        row_position: &mut [usize],
        spare: &mut SchurPool,
        stats: &mut FactorStats,
    ) {
        let front = &self.fronts[task.s];
        for &child in &front.children {
            valid[child] = false;
            spare.put(std::mem::take(&mut schur[child]));
        }
        std::mem::swap(&mut schur[task.s], &mut task.update);
        valid[task.s] = true;
        for (column, out) in task.l.iter_mut().zip(&mut l[front.lo..front.end]) {
            std::mem::swap(column, out);
        }
        blocks[front.lo..front.end].copy_from_slice(&task.blocks);
        for (k, &old) in task.perm.iter().enumerate() {
            order[front.lo + k] = self.order[front.lo + old];
            row_position[front.lo + old] = front.lo + k;
        }
        stats.perturbed_pivots += task.stats.perturbed_pivots;
        stats.num_2x2 += task.stats.num_2x2;
        stats.min_abs_pivot = stats.min_abs_pivot.min(task.stats.min_abs_pivot);
    }
}

#[derive(Debug, Default)]
pub(crate) struct FactorWorkspace {
    tasks: Vec<FrontWork>,
    levels: Vec<Range<usize>>,
    dense: Vec<DenseWorkspace>,
    schur: Vec<Vec<f64>>,
    valid: Vec<bool>,
    l: Vec<Vec<(usize, f64)>>,
    blocks: Vec<Block>,
    order: Vec<usize>,
    row_position: Vec<usize>,
    tree: bool,
    spare: SchurPool,
    assembly: Option<Arc<Vec<Assembly>>>,
}

impl Clone for FactorWorkspace {
    fn clone(&self) -> Self {
        // Scratch is disposable; cloning a solver copies its factor, not the
        // high-water allocations or partially computed temporary values.
        Self::default()
    }
}

impl FactorWorkspace {
    #[cfg(test)]
    pub(super) fn retained_buffers(&self) -> Vec<(*const f64, usize)> {
        self.dense
            .iter()
            .map(|d| (d.matrix.as_ptr(), d.matrix.capacity()))
            .collect()
    }
    fn recycle(&mut self, previous: Option<Factor>) {
        if let Some(Factor {
            l: LowerFactor::Columns(l),
            blocks,
            order,
            ..
        }) = previous
        {
            self.l = l;
            self.blocks = blocks;
            self.order = order;
        }
    }
    fn prepare(&mut self, symbolic: &Symbolic, tree: bool, workers: usize) {
        let n = symbolic.order.len();
        self.l.resize_with(n, Vec::new);
        self.blocks.resize(n, Block::Skip);
        self.row_position.resize(n, 0);
        let changed = self
            .assembly
            .as_ref()
            .map_or(true, |plan| !Arc::ptr_eq(plan, &symbolic.assembly));
        if changed || self.tasks.len() != symbolic.fronts.len() || self.tree != tree {
            self.assembly = Some(Arc::clone(&symbolic.assembly));
            self.tree = tree;
            self.tasks.clear();
            self.levels.clear();
            if tree {
                for level in &symbolic.levels {
                    let start = self.tasks.len();
                    self.tasks.extend(
                        level
                            .iter()
                            .map(|&s| FrontWork::new(s, &symbolic.fronts[s])),
                    );
                    self.levels.push(start..self.tasks.len());
                }
            } else {
                self.tasks.extend(
                    symbolic
                        .fronts
                        .iter()
                        .enumerate()
                        .map(|(s, f)| FrontWork::new(s, f)),
                );
                self.levels.push(0..self.tasks.len());
            }
            self.schur.resize_with(symbolic.fronts.len(), Vec::new);
            self.valid.resize(symbolic.fronts.len(), false);
        }
        self.dense.resize_with(workers, DenseWorkspace::default);
    }
}

#[derive(Debug, Default)]
struct DenseWorkspace {
    matrix: Vec<f64>,
    multipliers: DenseScratch,
}

#[derive(Debug, Default)]
struct FrontWork {
    s: usize,
    l: Vec<Vec<(usize, f64)>>,
    blocks: Vec<Block>,
    perm: Vec<usize>,
    update: Vec<f64>,
    stats: FactorStats,
}
impl FrontWork {
    fn empty(s: usize) -> Self {
        Self {
            s,
            ..Self::default()
        }
    }
    fn new(s: usize, front: &Front) -> Self {
        Self {
            s,
            l: vec![Vec::new(); front.end - front.lo],
            blocks: Vec::new(),
            perm: Vec::new(),
            update: Vec::new(),
            stats: FactorStats::default(),
        }
    }
}

// Retain a bounded amount of consumed Schur storage, rather than retaining
// every front's clique. On deep trees the latter can use cubic memory even
// when the live factorization needs only quadratic storage.
const SCHUR_CACHE_BYTES: usize = 8 * 1024 * 1024;
#[derive(Debug, Default)]
struct SchurPool {
    bins: Vec<Vec<Vec<f64>>>,
    bytes: usize,
}
impl SchurPool {
    fn take(&mut self, entries: usize) -> Vec<f64> {
        if entries == 0 {
            return Vec::new();
        }
        let bin = (usize::BITS - (entries - 1).leading_zeros()) as usize;
        for candidates in self.bins.iter_mut().skip(bin) {
            if let Some(mut buffer) = candidates.pop() {
                self.bytes -= buffer.capacity() * std::mem::size_of::<f64>();
                buffer.clear();
                return buffer;
            }
        }
        Vec::new()
    }
    fn put(&mut self, mut buffer: Vec<f64>) {
        let capacity = buffer.capacity();
        let bytes = capacity.saturating_mul(std::mem::size_of::<f64>());
        if capacity == 0 || bytes > SCHUR_CACHE_BYTES - self.bytes {
            return;
        }
        let bin = (usize::BITS - 1 - capacity.leading_zeros()) as usize;
        self.bins
            .resize_with(self.bins.len().max(bin + 1), Vec::new);
        buffer.clear();
        self.bins[bin].push(buffer);
        self.bytes += bytes;
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/ipm/factor/workspace_tests.rs"]
mod tests;
