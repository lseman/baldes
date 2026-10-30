//! Completed route storage and equivalence classes.
use super::*;
use std::{collections::HashMap, sync::Arc};
// Native retained-path export uses this hash domain; dominance storage can
// choose a smaller partition count independently.
const EXPORT_PARTITIONS: u32 = 2_000_000;
#[derive(Clone, Debug)]
struct MachineRetainedJoin {
    contribution: Option<f64>,
    paired_resource: Option<usize>,
    forward: Box<crate::graph::coupled::CoupledLabelState>,
    backward: Box<crate::graph::coupled::CoupledLabelState>,
}
#[derive(Clone, Debug)]
pub struct Path {
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub arcs: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct PathStorage {
    pub partitions: Vec<Vec<Path>>,
    pub len: usize,
    pub checks: u64,
    /// Arc-flow branching distinguishes orderings of the same customer set.
    pub(super) preserve_arc_paths: bool,
    export_partitions: Vec<u32>,
    export_ranks: std::collections::HashMap<u32, usize>,
    // Preserve the native partition's actual insertion order. Reconstructing
    // it from surviving insertion times is not valid with tolerant ordering.
    export_rows: Arc<Vec<Vec<(Arc<[usize]>, f64)>>>,
    // One index supplies both deduplication and canonical insertion order.
    // Forks share its immutable keys until a child changes membership.
    path_order: Arc<HashMap<Arc<[usize]>, usize>>,
    // Raw machine join values are diagnostic metadata, not source costs.
    machine_contributions: Arc<HashMap<Arc<[usize]>, MachineRetainedJoin>>,
    machine_identity: Option<Arc<()>>,
    next_path: usize,
}
impl PathStorage {
    pub fn new(partitions: usize) -> Result<Self> {
        if partitions == 0 {
            return Err(invalid("positive enumeration path partitions required"));
        }
        Ok(Self {
            partitions: vec![vec![]; partitions],
            len: 0,
            checks: 0,
            preserve_arc_paths: false,
            export_partitions: vec![],
            export_ranks: std::collections::HashMap::new(),
            export_rows: Arc::default(),
            path_order: Arc::default(),
            machine_contributions: Arc::default(),
            machine_identity: None,
            next_path: 0,
        })
    }
    pub fn insert(&mut self, path: Path) -> bool {
        let export_partition = path.element_hash % EXPORT_PARTITIONS;
        if !self.export_ranks.contains_key(&export_partition) {
            self.export_ranks
                .insert(export_partition, self.export_partitions.len());
            self.export_partitions.push(export_partition);
            Arc::make_mut(&mut self.export_rows).push(Vec::new());
        }
        let partition = path.element_hash as usize % self.partitions.len();
        if self.preserve_arc_paths {
            if self.path_order.contains_key(path.arcs.as_slice()) {
                return false;
            }
            Arc::make_mut(&mut self.path_order)
                .insert(Arc::from(path.arcs.as_slice()), self.next_path);
            self.next_path += 1;
            self.record_export_path(
                export_partition,
                Arc::from(path.arcs.as_slice()),
                path.original_cost,
                &[],
            );
            self.partitions[partition].push(path);
            self.len += 1;
            return true;
        }
        let row = &mut self.partitions[partition];
        // Dominance storage can coalesce several native export partitions.
        // Only candidates in this path's export partition may stop the native
        // cost scan or reject an equivalent candidate.
        let equivalent: Vec<&[usize]> = row
            .iter()
            .filter(|previous| {
                previous.element_hash % EXPORT_PARTITIONS == export_partition
                    && previous.visited == path.visited
            })
            .map(|previous| previous.arcs.as_slice())
            .collect();
        for (arcs, cost) in &self.export_rows[self.export_ranks[&export_partition]] {
            if *cost > path.original_cost + EPSILON {
                break;
            }
            self.checks += 1;
            if equivalent.iter().any(|old| *old == arcs.as_ref()) {
                return false;
            }
        }
        let mut at = row.len();
        for (index, previous) in row.iter().enumerate() {
            if previous.original_cost > path.original_cost + EPSILON {
                at = index;
                break;
            }
        }
        let before = row.len();
        let mut removed = Vec::new();
        let mut index = 0;
        let mut removed_before = 0;
        row.retain(|p| {
            let current = index;
            index += 1;
            self.checks += 1;
            if p.element_hash % EXPORT_PARTITIONS == export_partition && p.visited == path.visited {
                if current < at {
                    removed_before += 1;
                }
                Arc::make_mut(&mut self.path_order).remove(p.arcs.as_slice());
                Arc::make_mut(&mut self.machine_contributions).remove(p.arcs.as_slice());
                removed.push((p.element_hash % EXPORT_PARTITIONS, p.arcs.clone()));
                false
            } else {
                true
            }
        });
        self.len -= before - row.len();
        // Keep the physical row's insertion position after compaction. The
        // native export position is maintained independently below.
        Arc::make_mut(&mut self.path_order).insert(Arc::from(path.arcs.as_slice()), self.next_path);
        self.next_path += 1;
        let export_arcs = Arc::from(path.arcs.as_slice());
        let export_cost = path.original_cost;
        row.insert(at - removed_before, path);
        self.record_export_path(export_partition, export_arcs, export_cost, &removed);
        self.len += 1;
        true
    }
    fn record_export_path(
        &mut self,
        partition: u32,
        arcs: Arc<[usize]>,
        cost: f64,
        removed: &[(u32, Vec<usize>)],
    ) {
        let rank = self.export_ranks[&partition];
        let rows = Arc::make_mut(&mut self.export_rows);
        for (old_partition, old_arcs) in removed {
            if let Some(&index) = self.export_ranks.get(old_partition) {
                if index != rank {
                    rows[index].retain(|(arcs, _)| old_arcs.as_slice() != arcs.as_ref());
                }
            }
        }
        let row = &mut rows[rank];
        let at = row
            .iter()
            .position(|(_, previous)| *previous > cost + EPSILON)
            .unwrap_or(row.len());
        let removed_before = row[..at]
            .iter()
            .filter(|(arcs, _)| {
                removed
                    .iter()
                    .any(|(_, old)| old.as_slice() == arcs.as_ref())
            })
            .count();
        row.retain(|(arcs, _)| {
            !removed
                .iter()
                .any(|(_, old)| old.as_slice() == arcs.as_ref())
        });
        row.insert(at - removed_before, (arcs, cost));
    }
    /// Raw machine resource contribution retained for this path, indexed by
    /// physical arc-vector indices, when its join carried installed machine
    /// state. This is not a source-objective or feasibility certificate.
    pub fn machine_contribution(&self, arcs: &[usize]) -> Option<f64> {
        self.machine_contributions
            .get(arcs)
            .and_then(|join| join.contribution)
    }

    /// Return raw retained resource scalars in enumeration export order.
    /// Missing or stale contributions cannot describe a complete universe.
    pub fn machine_contributions<const D: usize>(
        &self,
        g: &Graph<D>,
    ) -> Result<Option<Vec<(Vec<usize>, f64)>>> {
        let Some(kernel) = g.coupled_kernel() else {
            return Ok(None);
        };
        kernel.validate_for(g)?;
        if !kernel.has_complete_machine_inspection_contribution() {
            return Ok(None);
        }
        if self.len == 0 {
            return Ok(Some(Vec::new()));
        }
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return Ok(None);
        }
        Ok(self
            .ordered_paths()
            .into_iter()
            .map(|path| {
                self.machine_contribution(&path.arcs)
                    .map(|value| (path.arcs.clone(), value))
            })
            .collect())
    }

    /// Return one retained paired-resource machine selection without running
    /// the path recorder. The optional index uses coupled-descriptor order.
    pub fn machine_paired_selection<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
        resource: Option<usize>,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<
        crate::graph::coupled::descriptor::MachineConcatenation,
    > {
        use crate::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(kernel) = g.coupled_kernel() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired front program unavailable",
            ));
        };
        let validation = g
            .arcs
            .len()
            .saturating_add(g.vertices.len())
            .saturating_add(kernel.definitions().len())
            .saturating_add(arcs.len());
        let Some(remaining_work) = budget.max_operations.checked_sub(validation) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if kernel.validate_for(g).is_err() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired front program graph changed",
            ));
        }
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained join is stale",
            ));
        }
        let Some(join) = self.machine_contributions.get(arcs) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained join unavailable",
            ));
        };
        let Some(resource) = resource.or(join.paired_resource) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained resource unavailable",
            ));
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        g.paired_machine_selection_for_path(
            resource,
            &join.forward,
            &join.backward,
            arcs,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Reconstruct the raw paired machine choices retained with an enumerated
    /// path. These values are not source-model controls or objective bounds.
    pub fn reconstruct_front_program_path<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<
        crate::graph::coupled::descriptor::MachineJoinedPathReconstruction,
    > {
        self.reconstruct_front_program_path_with_debug(g, arcs, 0, budget)
    }

    /// Reconstruct with an explicit native recorder diagnostic level.
    pub fn reconstruct_front_program_path_with_debug<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
        debug_level: i32,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<
        crate::graph::coupled::descriptor::MachineJoinedPathReconstruction,
    > {
        self.reconstruct_front_program_resource_with_debug(g, arcs, None, debug_level, budget)
    }

    /// Reconstruct one paired resource by its coupled-resource index.
    pub fn reconstruct_front_program_resource_with_debug<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
        resource: Option<usize>,
        debug_level: i32,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<
        crate::graph::coupled::descriptor::MachineJoinedPathReconstruction,
    > {
        use crate::graph::coupled::descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = g.coupled_kernel() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired front program unavailable",
            ));
        };
        let validation = g
            .arcs
            .len()
            .saturating_add(g.vertices.len())
            .saturating_add(kernel.definitions().len())
            .saturating_add(arcs.len());
        let Some(remaining_work) = budget.max_operations.checked_sub(validation) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if kernel.validate_for(g).is_err() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired front program graph changed",
            ));
        }
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained join is stale",
            ));
        }
        let Some(join) = self.machine_contributions.get(arcs) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained join unavailable",
            ));
        };
        let Some(resource) = resource.or(join.paired_resource) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired retained resource unavailable",
            ));
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        g.reconstruct_paired_machine_join_for_path(
            resource,
            &join.forward,
            &join.backward,
            arcs,
            debug_level,
            crate::graph::coupled::descriptor::FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Record a retained custom-resource path using its saved label states.
    /// The result is raw machine output, not a source-model cost certificate.
    pub fn record_custom_machine_path<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
    ) -> Result<Option<f64>> {
        let Some(kernel) = g.coupled_kernel() else {
            return Ok(None);
        };
        kernel.validate_for(g)?;
        if !kernel.has_single_custom_machine_resource() {
            return Ok(None);
        }
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return Ok(None);
        }
        let Some(join) = self.machine_contributions.get(arcs) else {
            return Ok(None);
        };
        kernel.record_custom_machine_path(arcs, &join.forward, &join.backward)
    }

    /// Record a selected custom resource using the retained coupled labels.
    pub fn record_custom_machine_resource_path<const D: usize>(
        &self,
        g: &Graph<D>,
        arcs: &[usize],
        resource: usize,
    ) -> Result<Option<f64>> {
        let Some(kernel) = g.coupled_kernel() else {
            return Ok(None);
        };
        kernel.validate_for(g)?;
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return Ok(None);
        }
        let Some(join) = self.machine_contributions.get(arcs) else {
            return Ok(None);
        };
        kernel.record_custom_machine_resource_path(resource, arcs, &join.forward, &join.backward)
    }

    pub(super) fn insert_with_machine_contribution(
        &mut self,
        path: Path,
        contribution: Option<(
            Arc<()>,
            Option<f64>,
            Option<usize>,
            Box<crate::graph::coupled::CoupledLabelState>,
            Box<crate::graph::coupled::CoupledLabelState>,
        )>,
    ) -> bool {
        let key = contribution
            .as_ref()
            .map(|_| Arc::<[usize]>::from(path.arcs.as_slice()));
        if !self.insert(path) {
            return false;
        }
        if let (Some(key), Some((identity, value, paired_resource, forward, backward))) =
            (key, contribution)
        {
            if self.machine_contributions.is_empty() {
                self.machine_identity = Some(identity.clone());
            }
            if self
                .machine_identity
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &identity))
            {
                Arc::make_mut(&mut self.machine_contributions).insert(
                    key,
                    MachineRetainedJoin {
                        contribution: value,
                        paired_resource,
                        forward,
                        backward,
                    },
                );
            }
        }
        true
    }

    /// Reprice retained machine paths in the binary's accumulation order.
    /// Returns `None` if a path lacks a contribution from the current machine
    /// resource. These are raw machine values, not source certificates.
    pub fn machine_inspection_costs<const D: usize>(
        &self,
        g: &Graph<D>,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<Option<Vec<(Vec<usize>, f64)>>> {
        self.machine_inspection_costs_until(g, cuts, kpath, None, false)
    }

    fn machine_inspection_costs_until<const D: usize>(
        &self,
        g: &Graph<D>,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        timer: Option<(std::time::Instant, std::time::Duration)>,
        normalize_paired: bool,
    ) -> Result<Option<Vec<(Vec<usize>, f64)>>> {
        let expired = || timer.is_some_and(|(start, limit)| start.elapsed() > limit);
        if expired() {
            return Err(crate::incomplete("enumeration time limit exceeded"));
        }
        let Some(kernel) = g.coupled_kernel() else {
            return Ok(None);
        };
        kernel.validate_for(g)?;
        if !kernel.has_complete_machine_inspection_contribution() {
            return Ok(None);
        }
        if self.len == 0 {
            return Ok(Some(Vec::new()));
        }
        if !self
            .machine_identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
        {
            return Ok(None);
        }
        let mut result = Vec::with_capacity(self.len);
        let mut vertices = Vec::new();
        let mut ids = Vec::new();
        for path in self.ordered_paths() {
            if expired() {
                return Err(crate::incomplete("enumeration time limit exceeded"));
            }
            let Some(mut cost) = self.machine_contribution(&path.arcs) else {
                return Ok(None);
            };
            if normalize_paired {
                cost = kernel.paired_retained_cost(cost)?;
            }
            for &index in &path.arcs {
                if expired() {
                    return Err(crate::incomplete("enumeration time limit exceeded"));
                }
                let arc = g
                    .arcs
                    .get(index)
                    .ok_or_else(|| invalid("unknown enumerated arc"))?;
                cost += arc.reduced_cost;
            }
            if !cuts.is_empty() || !kpath.is_empty() {
                vertices.clear();
                vertices.push(g.source);
                vertices.extend(path.arcs.iter().map(|&index| g.arcs[index].head));
                ids.clear();
                ids.extend(path.arcs.iter().map(|&index| g.arcs[index].id));
                cost = state::enumerated_path_cut_cost(cuts, kpath, &vertices, &ids, cost);
            }
            result.push((path.arcs.clone(), cost));
        }
        Ok(Some(result))
    }

    /// Return raw machine inspection prices ordered by cost, preserving the
    /// retained-path order on ties. Unordered costs cannot be represented as
    /// a stable numeric ordering and return `None`.
    pub fn machine_sorted_inspection_costs<const D: usize>(
        &self,
        g: &Graph<D>,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<Option<Vec<(Vec<usize>, f64)>>> {
        let Some(mut paths) = self.machine_inspection_costs(g, cuts, kpath)? else {
            return Ok(None);
        };
        if paths.iter().any(|(_, cost)| cost.is_nan()) {
            return Ok(None);
        }
        paths.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(Some(paths))
    }

    /// Compute the machine inspection cutoff flags in retained-path order.
    /// `true` marks a path for elimination by the cost check. The optional
    /// reference-side consistency check reports recorded paths against these
    /// flags; it does not add a second branch-restriction filter. This raw
    /// view does not change the source-model universe.
    pub fn machine_inspection_cutoff_flags<const D: usize>(
        &self,
        g: &Graph<D>,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        cutoff: f64,
    ) -> Result<Option<Vec<(Vec<usize>, bool)>>> {
        let Some(paths) = self.machine_inspection_costs(g, cuts, kpath)? else {
            return Ok(None);
        };
        Ok(Some(
            paths
                .into_iter()
                .map(|(arcs, cost)| {
                    // UCOMISD's unordered case takes the same branch as `<`.
                    let eliminate = !cost.is_nan() && !cutoff.is_nan() && cost >= cutoff;
                    (arcs, eliminate)
                })
                .collect(),
        ))
    }

    /// Eliminate retained paths using the installed front program's raw
    /// inspection costs. Returns `None` without changing the retained paths
    /// when the program did not provide a contribution for every path.
    ///
    /// The resulting costs are machine-front values; they do not certify the
    /// source model's objective. The retained paths keep their source costs.
    pub fn eliminate_by_front_program_cost<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        cutoff: f64,
    ) -> Result<Option<usize>> {
        if !g.front_program_domain() {
            return Ok(None);
        }
        let Some(flags) = self.machine_inspection_cutoff_flags(g, cuts, kpath, cutoff)? else {
            return Ok(None);
        };
        let before = self.len;
        let eliminated: std::collections::HashSet<Vec<usize>> = flags
            .into_iter()
            .filter_map(|(arcs, eliminate)| eliminate.then_some(arcs))
            .collect();
        self.retain(|path| !eliminated.contains(&path.arcs));
        Ok(Some(before - self.len))
    }
    /// Path repricing: physical arc costs, then rank-one costs from
    /// physical incidence (no memory).
    pub fn inspect<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<usize> {
        self.inspect_with_knapsack(g, cutoff, cuts, kpath, &Default::default())
    }
    pub fn inspect_with_knapsack<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
    ) -> Result<usize> {
        self.inspect_with_presence(g, cutoff, cuts, kpath, knapsack, &Default::default())
    }

    pub fn inspect_with_presence<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
        presence: &crate::cuts::soft_conflicts::PresencePricing,
    ) -> Result<usize> {
        self.inspect_with_presence_until(g, cutoff, cuts, kpath, knapsack, presence, None)
    }

    pub(crate) fn inspect_with_presence_until<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
        presence: &crate::cuts::soft_conflicts::PresencePricing,
        timer: Option<(std::time::Instant, std::time::Duration)>,
    ) -> Result<usize> {
        let expired = || timer.is_some_and(|(start, limit)| start.elapsed() > limit);
        if expired() {
            return Err(crate::incomplete("enumeration time limit exceeded"));
        }
        if g.coupled_kernel()
            .is_some_and(|kernel| kernel.has_negative_paired_objectives())
        {
            return Err(crate::incomplete(
                "negative paired objectives require source-control master pricing",
            ));
        }
        if cutoff.is_nan() {
            return Err(invalid("invalid enumeration inspection cutoff"));
        }
        let paired_costs = if g.coupled_kernel().is_some_and(|kernel| {
            kernel.machine_paired_join_complete()
                && kernel.has_complete_machine_inspection_contribution()
        }) {
            let paths = self
                .machine_inspection_costs_until(g, cuts, kpath, timer, true)?
                .ok_or_else(|| {
                    crate::incomplete("paired front-program inspection contributions unavailable")
                })?;
            Some(paths.into_iter().collect::<HashMap<_, _>>())
        } else {
            None
        };
        let before = self.len;
        let mut vertices = Vec::new();
        let mut ids = Vec::new();
        // Compute the complete inspection cost vector before changing the
        // retained universe. A failed resource or cut evaluation must not
        // leave earlier partitions repriced or partially eliminated.
        let mut costs = Vec::with_capacity(self.len);
        for row in &self.partitions {
            for path in row {
                if expired() {
                    return Err(crate::incomplete("enumeration time limit exceeded"));
                }
                let mut cost = if let Some(costs) = &paired_costs {
                    *costs.get(&path.arcs).ok_or_else(|| {
                        crate::incomplete("paired front-program inspection path unavailable")
                    })?
                } else {
                    // Match f64::sum's signed-zero identity and order.
                    let mut cost = -0.0f64;
                    for &arc in &path.arcs {
                        if expired() {
                            return Err(crate::incomplete("enumeration time limit exceeded"));
                        }
                        cost += g.arcs[arc].reduced_cost;
                    }
                    cost += g
                        .coupled_pricing_path_cost(&path.arcs)?
                        .ok_or_else(|| invalid("enumerated path violates coupled resources"))?;
                    cost
                };
                if paired_costs.is_none() && (!cuts.is_empty() || !kpath.is_empty()) {
                    vertices.clear();
                    vertices.push(g.source);
                    vertices.extend(path.arcs.iter().map(|&a| g.arcs[a].head));
                    ids.clear();
                    ids.extend(path.arcs.iter().map(|&a| g.arcs[a].id));
                    cost = state::enumerated_path_cut_cost(cuts, kpath, &vertices, &ids, cost);
                }
                if presence.active() {
                    cost += presence.route_cost(&path.arcs)?;
                }
                if knapsack.active {
                    let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
                    for r in 0..g.resources.len() {
                        q[r] = g
                            .root_resource(r, true)
                            .ok_or_else(|| invalid("initial resource outside source windows"))?;
                    }
                    for &arc in &path.arcs {
                        if expired() {
                            return Err(crate::incomplete("enumeration time limit exceeded"));
                        }
                        q = extend_resources(g, &g.arcs[arc], &q, true)
                            .ok_or_else(|| invalid("enumerated knapsack resource infeasible"))?;
                    }
                    cost += knapsack.cost(&q);
                }
                if !cost.is_finite() {
                    return Err(crate::incomplete(
                        "enumeration inspection cost is nonfinite",
                    ));
                }
                costs.push(cost);
            }
        }
        if expired() {
            return Err(crate::incomplete("enumeration time limit exceeded"));
        }
        let mut costs = costs.into_iter();
        for row in &mut self.partitions {
            for path in row.iter_mut() {
                path.reduced_cost = costs.next().expect("one inspection cost per path");
            }
            row.retain(|p| p.reduced_cost < cutoff);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        if self.len != before {
            self.retain_path_order();
        }
        Ok(before - self.len)
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&Path) -> bool) {
        let before = self.len;
        for partition in &mut self.partitions {
            partition.retain(&mut keep);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        if self.len != before {
            self.retain_path_order();
        }
    }
    pub fn routes<const D: usize>(&self, g: &Graph<D>) -> Result<Vec<Route>> {
        self.routes_until(g, None)
    }
    pub(crate) fn routes_until<const D: usize>(
        &self,
        g: &Graph<D>,
        timer: Option<(std::time::Instant, std::time::Duration)>,
    ) -> Result<Vec<Route>> {
        let expired = || timer.is_some_and(|(start, limit)| start.elapsed() > limit);
        if expired() {
            return Err(crate::incomplete("enumeration time limit exceeded"));
        }
        let mut routes = Vec::with_capacity(self.len);
        for path in self.ordered_paths() {
            if expired() {
                return Err(crate::incomplete("enumeration time limit exceeded"));
            }
            routes.push(self.route(g, path)?);
        }
        if expired() {
            return Err(crate::incomplete("enumeration time limit exceeded"));
        }
        Ok(routes)
    }
    /// Rank retained paths without replaying every resource schedule. The
    /// minimum describes the retained domain independently of output limits.
    pub(crate) fn priced_routes<const D: usize>(
        &self,
        g: &Graph<D>,
        maximum: usize,
        negative_only: bool,
        zero_threshold: f64,
    ) -> Result<(Vec<Route>, f64)> {
        let mut paths = self.ordered_paths();
        // Stable sorting preserves canonical export order for equal costs,
        // including the two signed zeros.
        paths.sort_by(|a, b| {
            a.reduced_cost
                .partial_cmp(&b.reduced_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let minimum = paths
            .first()
            .map_or(crate::labeling::NO_ROUTE_COST, |path| path.reduced_cost);
        let routes = paths
            .into_iter()
            .filter(|path| !negative_only || path.reduced_cost < zero_threshold - EPSILON)
            .take(maximum)
            .map(|path| self.route(g, path))
            .collect::<Result<_>>()?;
        Ok((routes, minimum))
    }
    fn ordered_paths(&self) -> Vec<&Path> {
        // Native export visits hash partitions in activation order, then
        // their candidates in original-cost insertion order. Dominance may
        // use fewer physical partitions without changing this path index.
        let paths: HashMap<&[usize], &Path> = self
            .partitions
            .iter()
            .flatten()
            .map(|path| (path.arcs.as_slice(), path))
            .collect();
        self.export_rows
            .iter()
            .flatten()
            .map(|(arcs, _)| paths[arcs.as_ref()])
            .collect()
    }
    fn route<const D: usize>(&self, g: &Graph<D>, path: &Path) -> Result<Route> {
        let retained_selection = if g
            .coupled_kernel()
            .is_some_and(|kernel| kernel.machine_paired_join_complete())
        {
            use crate::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome};
            let resource = self
                .machine_contributions
                .get(path.arcs.as_slice())
                .and_then(|join| join.paired_resource)
                .ok_or_else(|| crate::incomplete("paired retained resource unavailable"))?;
            match self.machine_paired_selection(
                g,
                &path.arcs,
                Some(resource),
                FrontierBudget::default(),
            ) {
                FrontierOutcome::Complete(selection) => Some((resource, selection.dependent)),
                FrontierOutcome::Empty => {
                    return Err(invalid("enumerated paired join is infeasible"))
                }
                FrontierOutcome::Unknown(reason) => {
                    return Err(crate::incomplete(format!(
                        "paired retained join unavailable: {reason:?}"
                    )))
                }
            }
        } else {
            None
        };
        let machine_feasible = if let Some(kernel) = g
            .coupled_kernel()
            .filter(|kernel| kernel.uses_zero_paired_inspection_cost())
        {
            kernel.validate_for(g)?;
            if !self
                .machine_identity
                .as_ref()
                .is_some_and(|identity| Arc::ptr_eq(identity, kernel.machine_identity()))
                || self.machine_contribution(&path.arcs).is_none()
            {
                return Err(crate::incomplete(
                    "paired front-program route contribution unavailable",
                ));
            }
            true
        } else {
            false
        };
        if retained_selection.is_none() {
            g.coupled_route_cost(&path.arcs)?
                .ok_or_else(|| invalid("enumerated route violates coupled resources"))?;
        }
        let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
        for r in 0..g.resources.len() {
            q[r] = g
                .root_resource(r, true)
                .ok_or_else(|| invalid("initial resource outside source windows"))?;
        }
        let mut nodes = vec![g.source as i32];
        let mut resource_profile = vec![q[..g.resources.len()].to_vec()];
        for &id in &path.arcs {
            let a = &g.arcs[id];
            q = extend_resources(g, a, &q, true)
                .ok_or_else(|| invalid("enumerated route is no longer resource feasible"))?;
            nodes.push(a.head as i32);
            resource_profile.push(q[..g.resources.len()].to_vec());
        }
        Ok(Route {
            coupled_values: if let Some((resource, value)) = retained_selection {
                value
                    .map(|value| vec![(resource, value)])
                    .unwrap_or_default()
            } else if machine_feasible {
                Vec::new()
            } else {
                g.coupled_paired_values(&path.arcs)?
            },
            nodes,
            arcs: path.arcs.iter().map(|&a| g.arcs[a].id).collect(),
            cost: path.reduced_cost,
            original_cost: path.original_cost,
            resources: q[..g.resources.len()].to_vec(),
            resource_profile,
        })
    }
    fn retain_path_order(&mut self) {
        let live: HashSet<_> = self
            .partitions
            .iter()
            .flatten()
            .map(|path| path.arcs.as_slice())
            .collect();
        Arc::make_mut(&mut self.path_order).retain(|arcs, _| live.contains(arcs.as_ref()));
        for row in Arc::make_mut(&mut self.export_rows) {
            row.retain(|(arcs, _)| live.contains(arcs.as_ref()));
        }
        Arc::make_mut(&mut self.machine_contributions)
            .retain(|arcs, _| live.contains(arcs.as_ref()));
        if self.machine_contributions.is_empty() {
            self.machine_identity = None;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/enumeration/paths/order_tests.rs"]
mod order_tests;
