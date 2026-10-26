//! Dependency-ready tree work. Dense work runs outside the shared state lock.
use super::*;
use std::collections::VecDeque;
use std::sync::Mutex;

struct State<'a> {
    work: &'a mut FactorWorkspace,
    slots: Vec<usize>,
    pending: Vec<usize>,
    parent: Vec<Option<usize>>,
    ready: VecDeque<usize>,
    workers: usize,
    error: Option<Error>,
    stats: FactorStats,
}

#[cfg(test)]
#[path = "../../../../../tests/unit/extensions/ipm/factor/schedule_tests.rs"]
mod tests;

impl Symbolic {
    pub(super) fn factorize_dependencies(
        &self,
        a: &SparseMatrix,
        regularize: bool,
        execution: &Execution,
        work: &mut FactorWorkspace,
        node: bool,
        workers: usize,
    ) -> Result<Factor> {
        work.prepare(self, true, workers);
        for values in &mut work.schur {
            work.spare.put(std::mem::take(values));
        }
        work.valid.fill(false);
        work.order.clone_from(&self.order);
        let mut slots = vec![0; self.fronts.len()];
        for (slot, task) in work.tasks.iter().enumerate() {
            slots[task.s] = slot;
        }
        let mut parent = vec![None; self.fronts.len()];
        for (s, front) in self.fronts.iter().enumerate() {
            for &child in &front.children {
                parent[child] = Some(s);
            }
        }
        let state = Mutex::new(State {
            work,
            slots,
            parent,
            ready: self
                .fronts
                .iter()
                .enumerate()
                .filter_map(|(s, f)| f.children.is_empty().then_some(s))
                .collect(),
            pending: self.fronts.iter().map(|f| f.children.len()).collect(),
            workers,
            error: None,
            stats: FactorStats {
                min_abs_pivot: f64::INFINITY,
                ..Default::default()
            },
        });
        let run = || {
            rayon::scope(|scope| {
                // At most one in-flight front per retained dense workspace.
                // Completion releases its buffer before scheduling replacement
                // work, including when a worker runs nested Rayon node jobs.
                for _ in 0..workers {
                    let next = state.lock().unwrap().ready.pop_front();
                    if let Some(s) = next {
                        let state = &state;
                        scope.spawn(move |scope| {
                            process(scope, self, a, regularize, node, state, s)
                        });
                    }
                }
            })
        };
        execution
            .pool
            .as_ref()
            .expect("tree work has a pool")
            .install(run);
        let State {
            work, error, stats, ..
        } = state.into_inner().unwrap();
        if let Some(error) = error {
            return Err(error);
        }
        for column in &mut work.l {
            for (i, _) in column.iter_mut() {
                *i = work.row_position[*i];
            }
            column.sort_by_key(|e| e.0);
        }
        Ok(Factor {
            l: LowerFactor::Columns(std::mem::take(&mut work.l)),
            blocks: std::mem::take(&mut work.blocks),
            order: std::mem::take(&mut work.order),
            stats,
        })
    }
}

fn process<'scope, 'work: 'scope>(
    scope: &rayon::Scope<'scope>,
    symbolic: &'scope Symbolic,
    a: &'scope SparseMatrix,
    regularize: bool,
    node: bool,
    state: &'scope Mutex<State<'work>>,
    s: usize,
) {
    let (mut task, mut scratch, children, slot) = {
        let mut state = state.lock().unwrap();
        if state.error.is_some() {
            return;
        }
        if symbolic.fronts[s]
            .children
            .iter()
            .any(|&c| !state.work.valid[c])
        {
            state.error = Some(Error::NumericalFailure("missing ready child".into()));
            return;
        }
        let slot = state.slots[s];
        let work = &mut state.work;
        let mut task = std::mem::replace(&mut work.tasks[slot], FrontWork::empty(s));
        symbolic.prepare_front(&mut task, &mut work.schur, &mut work.l, &mut work.spare);
        let children: Vec<_> = symbolic.fronts[s]
            .children
            .iter()
            .map(|&c| {
                work.valid[c] = false;
                std::mem::take(&mut work.schur[c])
            })
            .collect();
        let scratch = work.dense.pop().expect("one workspace per in-flight front");
        (task, scratch, children, slot)
    };
    // No state/scratch lock may span nested Rayon node work.
    let result = symbolic.compute_front_values(
        a,
        regularize,
        node,
        |i| Some(children[i].as_slice()),
        &mut scratch,
        &mut task,
    );
    let next = {
        let mut state = state.lock().unwrap();
        for child in children {
            state.work.spare.put(child);
        }
        debug_assert!(state.work.dense.len() < state.workers);
        state.work.dense.push(scratch);
        if let Err(error) = result {
            if state.error.is_none() {
                state.error = Some(error);
            }
        }
        if state.error.is_some() {
            state.work.tasks[slot] = task;
            return;
        }
        let State { work, stats, .. } = &mut *state;
        symbolic.commit_front(
            &mut task,
            &mut work.schur,
            &mut work.valid,
            &mut work.l,
            &mut work.blocks,
            &mut work.order,
            &mut work.row_position,
            &mut work.spare,
            stats,
        );
        work.tasks[slot] = task;
        if let Some(parent) = state.parent[s] {
            state.pending[parent] -= 1;
            if state.pending[parent] == 0 {
                state.ready.push_front(parent);
            }
        }
        state.ready.pop_front()
    };
    if let Some(parent) = next {
        scope.spawn(move |scope| process(scope, symbolic, a, regularize, node, state, parent));
    }
}
