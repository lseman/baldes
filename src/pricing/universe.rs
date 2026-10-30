//! Elementary enumeration, inspection, and reduced-cost fixing.
use super::*;

pub(super) fn is_enumeration_interruption(message: &str) -> bool {
    matches!(message,
        "enumeration time limit exceeded"
        | "enumeration concatenation time limit exceeded"
        | "enumeration label budget exceeded"
        | "enumeration pending-label budget exceeded"
        | "enumeration path budget exceeded"
    )
}

impl<const D: usize> Solver<D> {
    /// A partial store cannot replace the original pricing domain. Rebuild
    /// cost-dependent working state rather than retaining an enumeration's
    /// possibly restricted bucket graph or completion bounds.
    pub(super) fn discard_incomplete_enumeration(&mut self) {
        if self
            .enumeration
            .as_ref()
            .is_some_and(|universe| !universe.completed)
        {
            self.enumeration = None;
            self.enumeration_snapshot = None;
            self.engine = None;
            self.prepared = None;
        }
    }
    pub(super) fn check_presence_universe(&self) -> Result<()> {
        if self
            .engine
            .as_ref()
            .is_some_and(|e| e.cuts.presence.inputs() != self.presence_pricing.inputs())
        {
            return Err(incomplete(
                "unowned presence pricing rows require enumeration state integration",
            ));
        }
        Ok(())
    }

    pub fn enumerate(&mut self, cutoff: f64) -> Result<Vec<crate::Route>> {
        self.enumerate_with_concatenation_limit(cutoff, None)
    }

    pub(super) fn enumerate_with_concatenation_limit(
        &mut self,
        cutoff: f64,
        concatenation_limit: Option<Duration>,
    ) -> Result<Vec<crate::Route>> {
        self.check_presence_universe()?;
        if self.engine.is_none() {
            self.price_phase(0, 1, false)?;
        }
        self.complete_labels(true)?;
        let controls = enumeration::Controls {
            cutoff,
            label_limit: self
                .options
                .get("BALDESmaxNumOfLabelsInEnumeration")?
                .max(1.) as usize,
            path_limit: self
                .options
                .get("BALDESmaxNumOfEnumeratedSolutions")?
                .max(1.) as usize,
            time_limit: self.controls.time_limit,
            max_route_arcs: self.controls.max_route_arcs,
            preserve_arc_paths: self.enumeration_preserve_arc_paths
                || !self.ryan_foster.is_empty()
                || self.presence_pricing.active(),
            // Dominance needs only one working partition. Path export tracks
            // native hash activation and candidate order independently.
            partitions: 1,
        };
        let scan = self.options.get("BALDESuseExactComplBoundsInEnumeration")? != 0.;
        // Enumeration traverses bucket arcs directly rather than the pricing
        // extension routine. Filter only after fallible control lookup, so a
        // setup error cannot leave a partially modified working grid.
        if let Some(engine) = &mut self.engine {
            for direction in [&mut engine.grid.forward, &mut engine.grid.backward] {
                for bucket in direction.buckets.iter_mut().flatten() {
                    bucket.arcs.retain(|a| {
                        !self
                            .forbidden_arcs
                            .get(a.physical)
                            .copied()
                            .unwrap_or(false)
                    });
                }
            }
        }
        let enumerated_kpath = self.enumerated_kpath_inputs();
        let has_cuts = self.has_cuts();
        if has_cuts {
            let engine = self.engine.as_mut().unwrap();
            let mut full = state::FullPreparation::default();
            // Completion can materialize backward buckets even when pricing
            // started in forward-only mode. Enumeration uses those buckets.
            let backward = !engine.consumption_joins && !engine.grid.backward.buckets.is_empty();
            if !state::prepare_full_memory(
                &mut engine.cuts.info,
                &mut full,
                &self.graph,
                &self.applied_cuts.2,
                &self.applied_cuts.3,
                backward,
            ) {
                // Fixed-arc filtering above has already modified the working
                // bucket graph. Discard it on preparation failure just as on
                // an interrupted enumeration run.
                self.restore_arcs();
                return Err(incomplete(
                    "enumeration cut memory exceeds the native label words",
                ));
            }
            full.limited_to_field = self.cut_input_source.as_ref()
                .and_then(|source| source.projection_fields(&full));
            engine.cuts.full = Some(full);
        }
        let engine = self.engine.as_ref().unwrap();
        let tables =
            engine
                .cuts
                .full
                .as_ref()
                .filter(|_| has_cuts)
                .map(|full| enumeration::CutTables {
                    info: &engine.cuts.info,
                    full,
                    knapsack: &engine.cuts.knapsack,
                    presence: engine
                        .cuts
                        .presence
                        .active()
                        .then_some(&engine.cuts.presence),
                });
        let enumeration_started = std::time::Instant::now();
        let result = Enumeration::run_with_concatenation_limit(
            &self.graph,
            &engine.grid,
            &controls,
            if engine.consumption_joins {
                &engine.reverse
            } else {
                &[]
            },
            tables,
            Some(&engine.branching),
            |forward, label| {
                engine.enumeration_completion_with_timer(
                    &self.graph, forward, label, cutoff, scan,
                    Some((enumeration_started, controls.time_limit)),
                )
            },
            concatenation_limit,
        );
        if engine.completion_resource_failed() {
            self.restore_arcs();
            return Err(incomplete(
                "coupled resource completion arithmetic failure during enumeration",
            ));
        }
        let mut universe = match result {
            Ok(universe) => universe,
            Err(error) => {
                // A budget interruption leaves the fixed bucket graph and
                // completed pricing labels usable. Enumeration's local
                // labels and paths are not published until success.
                if !is_enumeration_interruption(&error.message) {
                    self.restore_arcs();
                }
                return Err(error);
            }
        };
        if enumeration_started.elapsed() > controls.time_limit {
            return Err(incomplete("enumeration time limit exceeded"));
        }
        let g = &self.graph;
        universe.paths.retain(|p| {
            self.ryan_foster.iter().all(|c| {
                c.satisfied_by(
                    std::iter::once(g.source).chain(p.arcs.iter().map(|&a| g.arcs[a].head)),
                )
            })
        });
        if self.presence_pricing.active() {
            if let Err(error) = universe.paths.inspect_with_presence(
                &self.graph,
                cutoff,
                &self.applied_cuts.0,
                &enumerated_kpath,
                &self.knapsack_pricing,
                &self.presence_pricing,
            ) {
                self.restore_arcs();
                return Err(error);
            }
        }
        let routes = match universe.paths.routes(&self.graph) {
            Ok(routes) => routes,
            Err(error) => {
                self.restore_arcs();
                return Err(error);
            }
        };
        self.enumeration_snapshot = Some(self.topology());
        self.enumeration = Some(universe);
        Ok(routes)
    }
    pub fn inspect_enumeration(&mut self, cutoff: f64) -> Result<Option<Vec<crate::Route>>> {
        self.discard_incomplete_enumeration();
        self.check_presence_universe()?;
        self.refresh_cuts()?;
        let enumerated_kpath = self.enumerated_kpath_inputs();
        if let Some(universe) = &mut self.enumeration {
            universe.paths.inspect_with_presence(
                &self.graph,
                cutoff,
                &self.applied_cuts.0,
                &enumerated_kpath,
                &self.knapsack_pricing,
                &self.presence_pricing,
            )?;
            return Ok(Some(universe.paths.routes(&self.graph)?));
        }
        Ok(None)
    }

    /// Return retained raw machine inspection prices using public arc IDs.
    /// These are not source-model objective certificates. Unsupported active
    /// cut families or missing machine contributions return `None`.
    pub fn machine_enumeration_inspection_costs(
        &mut self,
        sorted: bool,
    ) -> Result<Option<Vec<(Vec<i32>, f64)>>> {
        self.discard_incomplete_enumeration();
        self.check_presence_universe()?;
        self.refresh_cuts()?;
        if self.presence_pricing.active() || self.knapsack_pricing.active {
            return Ok(None);
        }
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return Ok(None);
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return Ok(None);
        }
        let kpath = self.enumerated_kpath_inputs();
        let prices = if sorted {
            universe.paths.machine_sorted_inspection_costs(&self.graph, &self.applied_cuts.0, &kpath)?
        } else {
            universe.paths.machine_inspection_costs(&self.graph, &self.applied_cuts.0, &kpath)?
        };
        Ok(prices.map(|prices| prices.into_iter().map(|(arcs, cost)| {
            (arcs.into_iter().map(|index| self.graph.arcs[index].id).collect(), cost)
        }).collect()))
    }

    /// Return retained raw machine resource scalars using public arc IDs.
    /// The values exclude later arc and cut pricing and are not certificates.
    pub fn machine_enumeration_contributions(&mut self) -> Result<Option<Vec<(Vec<i32>, f64)>>> {
        self.discard_incomplete_enumeration();
        self.refresh_cuts()?;
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return Ok(None);
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return Ok(None);
        }
        let contributions = universe.paths.machine_contributions(&self.graph)?;
        Ok(contributions.map(|values| values.into_iter().map(|(arcs, value)| {
            (arcs.into_iter().map(|index| self.graph.arcs[index].id).collect(), value)
        }).collect()))
    }

    /// Return the machine inspection cutoff decisions without changing the
    /// retained source universe. Unordered comparisons retain the path.
    pub fn machine_enumeration_inspection_flags(
        &mut self,
        cutoff: f64,
    ) -> Result<Option<Vec<(Vec<i32>, bool)>>> {
        let Some(prices) = self.machine_enumeration_inspection_costs(false)? else {
            return Ok(None);
        };
        Ok(Some(prices.into_iter().map(|(arcs, cost)| {
            (arcs, !cost.is_nan() && !cutoff.is_nan() && cost >= cutoff)
        }).collect()))
    }

    /// Return the raw paired selection retained for an enumerated route.
    /// Resource indices follow coupled-descriptor order; `None` selects the
    /// first installed paired program.
    pub fn machine_enumeration_paired_selection(
        &self,
        arcs: &[i32],
        resource: Option<usize>,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<crate::graph::coupled::descriptor::MachineConcatenation> {
        use crate::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(remaining_work) = arcs.len().checked_mul(self.graph.arcs.len())
            .and_then(|work| budget.max_operations.checked_sub(work)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("completed enumeration unavailable"));
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated graph changed"));
        }
        let Some(indices) = arcs.iter().map(|id| self.graph.arcs.iter().position(|arc| arc.id == *id)).collect::<Option<Vec<_>>>() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated arc unavailable"));
        };
        let remaining_time = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        universe.paths.machine_paired_selection(&self.graph, &indices, resource, FrontierBudget {
            max_operations: remaining_work,
            max_elapsed: remaining_time,
            ..budget
        })
    }

    /// Reconstruct raw paired-program values for a retained enumerated route.
    /// Arc IDs use the same representation as [`crate::Route::arcs`]. The
    /// result is a machine record, not a source-control certificate.
    pub fn reconstruct_enumerated_front_program(
        &self,
        arcs: &[i32],
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<crate::graph::coupled::descriptor::MachineJoinedPathReconstruction> {
        self.reconstruct_enumerated_front_program_with_debug(arcs, 0, budget)
    }

    /// Reconstruct a retained route with an explicit recorder diagnostic level.
    pub fn reconstruct_enumerated_front_program_with_debug(
        &self,
        arcs: &[i32],
        debug_level: i32,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<crate::graph::coupled::descriptor::MachineJoinedPathReconstruction> {
        self.reconstruct_enumerated_front_resource_with_debug(arcs, None, debug_level, budget)
    }

    /// Reconstruct one paired resource by its coupled-resource index.
    /// Passing `None` selects the first retained paired program.
    pub fn reconstruct_enumerated_front_resource_with_debug(
        &self,
        arcs: &[i32],
        resource: Option<usize>,
        debug_level: i32,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<crate::graph::coupled::descriptor::MachineJoinedPathReconstruction> {
        use crate::graph::coupled::descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(remaining_work) = arcs.len().checked_mul(self.graph.arcs.len())
            .and_then(|work| budget.max_operations.checked_sub(work)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("completed enumeration unavailable"));
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated graph changed"));
        }
        let Some(indices) = arcs.iter().map(|id| self.graph.arcs.iter().position(|arc| arc.id == *id)).collect::<Option<Vec<_>>>() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated arc unavailable"));
        };
        let remaining_time = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        universe.paths.reconstruct_front_program_resource_with_debug(&self.graph, &indices, resource, debug_level, crate::graph::coupled::descriptor::FrontierBudget {
            max_operations: remaining_work,
            max_elapsed: remaining_time,
            ..budget
        })
    }

    /// Return a source-certified paired control recovered from a retained
    /// machine trajectory. Arc IDs are the public route IDs. The result is a
    /// certificate for this control, not a proof of its optimality.
    pub fn certify_enumerated_paired_control(
        &self,
        arcs: &[i32],
        resource: Option<usize>,
        budget: crate::graph::coupled::descriptor::FrontierBudget,
    ) -> crate::graph::coupled::descriptor::FrontierOutcome<(crate::graph::coupled::PairedControl, f64)> {
        use crate::graph::coupled::descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(remaining_work) = arcs.len().checked_mul(self.graph.arcs.len())
            .and_then(|work| budget.max_operations.checked_sub(work)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("completed enumeration unavailable"));
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated graph changed"));
        }
        let Some(indices) = arcs.iter().map(|id| self.graph.arcs.iter().position(|arc| arc.id == *id)).collect::<Option<Vec<_>>>() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("enumerated arc unavailable"));
        };
        let remaining_time = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        universe.paths.certify_reconstructed_paired_control(&self.graph, &indices, resource, crate::graph::coupled::descriptor::FrontierBudget {
            max_operations: remaining_work,
            max_elapsed: remaining_time,
            ..budget
        })
    }

    /// Replay the raw custom-resource recorder for a retained enumerated
    /// route, using the public arc IDs in [`crate::Route::arcs`].
    pub fn record_enumerated_custom_machine_path(&self, arcs: &[i32]) -> Result<Option<f64>> {
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return Ok(None);
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return Ok(None);
        }
        let indices = arcs.iter().map(|id| self.graph.arcs.iter().position(|arc| arc.id == *id)
            .ok_or_else(|| invalid("enumerated arc unavailable"))).collect::<Result<Vec<_>>>()?;
        universe.paths.record_custom_machine_path(&self.graph, &indices)
    }

    /// Replay a selected custom resource for a retained enumerated route.
    pub fn record_enumerated_custom_machine_resource_path(&self, arcs: &[i32], resource: usize) -> Result<Option<f64>> {
        let Some(universe) = self.enumeration.as_ref().filter(|universe| universe.completed) else {
            return Ok(None);
        };
        if !self.enumeration_snapshot.as_ref().is_some_and(|snapshot| snapshot.same_topology(&self.graph)) {
            return Ok(None);
        }
        let indices = arcs.iter().map(|id| self.graph.arcs.iter().position(|arc| arc.id == *id)
            .ok_or_else(|| invalid("enumerated arc unavailable"))).collect::<Result<Vec<_>>>()?;
        universe.paths.record_custom_machine_resource_path(&self.graph, &indices, resource)
    }
    pub fn fix(&mut self, threshold: f64) -> Result<fixing::FixingReport> {
        self.complete_labels(true)?;
        let has_full_cuts = !self.applied_cuts.2.is_empty() || !self.applied_cuts.3.is_empty();
        self.engine
            .as_mut()
            .ok_or_else(|| invalid("pricing must precede fixing"))?
            .fix_bucket_arcs_with_full_cut_state(&self.graph, threshold, has_full_cuts)
    }
}
