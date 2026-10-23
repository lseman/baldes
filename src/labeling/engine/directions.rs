use super::sweep::Sweep;
use super::*;

impl<const D: usize> LabelingEngine<D> {
    pub(super) fn ctx(&self) -> Context<'_, D> {
        Context {
            grid: &self.grid,
            ng: &self.ng,
            branching: &self.branching,
            forbidden_arcs: &self.forbidden_arcs,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
            symmetric_costs: &self.symmetric_costs,
            coupled_costs: self.coupled_costs,
            coupled_completion: &self.coupled_completion,
            resource_error: &self.resource_error,
        }
    }

    pub(super) fn vertex_step(&self, label: &mut StoredLabel<D>) {
        self.ctx().vertex_step(label)
    }

    pub(super) fn root_cut_state(&self, vertex: usize) -> ([u32; LIMITED_WORDS], f64) {
        self.ctx().root_cut_state(vertex)
    }

    pub fn join_cost(
        &self,
        g: &Graph<D>,
        vertex: usize,
        f: &StoredLabel<D>,
        b: &StoredLabel<D>,
    ) -> Option<f64> {
        self.ctx().join_cost(g, vertex, f.into(), b.into())
    }

    pub(super) fn extend(
        &self,
        g: &Graph<D>,
        forward: bool,
        arc: &BucketArc,
        from: LabelRef<'_, D>,
    ) -> Option<StoredLabel<D>> {
        self.ctx().extend(g, forward, arc, from)
    }

    /// A sweep of one direction on this engine's own counters; `other` holds
    /// the opposite direction's label count for the combined label limit.
    pub(super) fn sweep<'a>(
        &'a mut self,
        forward: bool,
        terminals: &'a mut Vec<(usize, f64, usize)>,
        counts: &'a (AtomicUsize, AtomicUsize),
    ) -> Sweep<'a, D> {
        let ctx = Context {
            grid: &self.grid,
            ng: &self.ng,
            branching: &self.branching,
            forbidden_arcs: &self.forbidden_arcs,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
            symmetric_costs: &self.symmetric_costs,
            coupled_costs: self.coupled_costs,
            coupled_completion: &self.coupled_completion,
            resource_error: &self.resource_error,
        };
        let (storage, other) = if forward {
            (&mut self.forward, &self.backward)
        } else {
            (&mut self.backward, &self.forward)
        };
        counts.1.store(other.labels.len(), Ordering::Relaxed);
        Sweep {
            ctx,
            storage,
            statistics: &mut self.statistics,
            extension_checks: &mut self.extension_checks,
            filter_checks: &mut self.filter_checks,
            filter_scratch: Default::default(),
            coupled_workspace: Default::default(),
            terminals,
            labels: (&counts.0, &counts.1),
        }
    }

    pub(super) fn insert(
        &mut self,
        g: &Graph<D>,
        forward: bool,
        label: &StoredLabel<D>,
        hot: HotLabel<D>,
        controls: &Controls,
    ) -> bool {
        let mut terminals = vec![];
        let counts = Default::default();
        self.sweep(forward, &mut terminals, &counts)
            .insert::<true, StoredLabel<D>, _>(g, forward, label, hot, controls)
    }

    pub(super) fn run_direction(
        &mut self,
        g: &Graph<D>,
        forward: bool,
        controls: &Controls,
        started: Instant,
        pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    ) -> Result<()> {
        let mut terminals = vec![];
        let counts = Default::default();
        let result = self
            .sweep(forward, &mut terminals, &counts)
            .run_direction(g, forward, controls, started);
        insert_terminals(pool, forward, &terminals);
        if result.is_ok() {
            self.update_knapsack_bounds(g, forward);
        }
        result
    }

    /// The forward and backward sweeps of `solve` on two threads. The
    /// backward sweep counts into scratch statistics merged afterwards, and
    /// the pool receives the forward candidates, then the backward ones, as
    /// in sequential order. Only the combined label limit can trip at a
    /// different point; it ends pricing either way. Returns the sweep times.
    pub(super) fn run_both_directions(
        &mut self,
        g: &Graph<D>,
        controls: &Controls,
        started: Instant,
        pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    ) -> Result<(f64, f64)> {
        let ctx = Context {
            grid: &self.grid,
            ng: &self.ng,
            branching: &self.branching,
            forbidden_arcs: &self.forbidden_arcs,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
            symmetric_costs: &self.symmetric_costs,
            coupled_costs: self.coupled_costs,
            coupled_completion: &self.coupled_completion,
            resource_error: &self.resource_error,
        };
        let vertices = self.extension_checks.len();
        let (mut statistics, mut extension_checks, mut filter_checks) =
            (Statistics::default(), vec![0; vertices], vec![0; vertices]);
        let (mut forward_terminals, mut backward_terminals) = (vec![], vec![]);
        let forward_count = AtomicUsize::new(self.forward.labels.len());
        let backward_count = AtomicUsize::new(self.backward.labels.len());
        let mut forward = Sweep {
            ctx,
            storage: &mut self.forward,
            statistics: &mut self.statistics,
            extension_checks: &mut self.extension_checks,
            filter_checks: &mut self.filter_checks,
            filter_scratch: Default::default(),
            coupled_workspace: Default::default(),
            terminals: &mut forward_terminals,
            labels: (&forward_count, &backward_count),
        };
        let mut backward = Sweep {
            ctx,
            storage: &mut self.backward,
            statistics: &mut statistics,
            extension_checks: &mut extension_checks,
            filter_checks: &mut filter_checks,
            filter_scratch: Default::default(),
            coupled_workspace: Default::default(),
            terminals: &mut backward_terminals,
            labels: (&backward_count, &forward_count),
        };
        let timed = |sweep: &mut Sweep<D>, forward: bool| {
            let begin = Instant::now();
            let result = sweep.run_direction(g, forward, controls, started);
            (result, begin.elapsed().as_secs_f64())
        };
        let ((forward_result, forward_time), (backward_result, backward_time)) =
            std::thread::scope(|scope| {
                let handle = scope.spawn(|| timed(&mut backward, false));
                let forward_result = timed(&mut forward, true);
                (
                    forward_result,
                    handle.join().expect("backward labeling sweep panicked"),
                )
            });
        self.statistics.absorb(&statistics);
        for (total, part) in self.extension_checks.iter_mut().zip(&extension_checks) {
            *total = total.wrapping_add(*part);
        }
        for (total, part) in self.filter_checks.iter_mut().zip(&filter_checks) {
            *total = total.wrapping_add(*part);
        }
        forward_result?;
        insert_terminals(pool, true, &forward_terminals);
        backward_result?;
        insert_terminals(pool, false, &backward_terminals);
        self.update_knapsack_bounds(g, true);
        self.update_knapsack_bounds(g, false);
        Ok((forward_time, backward_time))
    }
}
