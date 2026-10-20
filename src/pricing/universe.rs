//! Elementary enumeration, inspection, and reduced-cost fixing.
use super::*;
impl<const D: usize> Solver<D> {
    pub fn enumerate(&mut self, cutoff: f64) -> Result<Vec<crate::Route>> {
        if self.engine.is_none() {
            self.price_phase(0, 1, false)?;
        }
        self.complete_labels(true)?;
        // Enumeration traverses bucket arcs directly rather than the pricing
        // extension routine. Remove fixed arcs after completion/jump rebuilds.
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
            preserve_arc_paths: self.enumeration_preserve_arc_paths || !self.ryan_foster.is_empty(),
            // Dominance needs only one working partition. Path export tracks
            // native hash activation and candidate order independently.
            partitions: 1,
        };
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
                return Err(incomplete(
                    "enumeration cut memory exceeds the native label words",
                ));
            }
            engine.cuts.full = Some(full);
        }
        let engine = self.engine.as_ref().unwrap();
        let scan = self.options.get("BALDESuseExactComplBoundsInEnumeration")? != 0.;
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
                });
        let mut universe = Enumeration::run_with_symmetry(
            &self.graph,
            &engine.grid,
            &controls,
            if engine.consumption_joins {
                &engine.reverse
            } else {
                &[]
            },
            tables,
            |forward, label| {
                engine.enumeration_completion(&self.graph, forward, label, cutoff, scan)
            },
        )?;
        let g = &self.graph;
        universe.paths.retain(|p| {
            self.ryan_foster.iter().all(|c| {
                c.satisfied_by(
                    std::iter::once(g.source).chain(p.arcs.iter().map(|&a| g.arcs[a].head)),
                )
            })
        });
        let routes = universe.paths.routes(&self.graph)?;
        self.enumeration_snapshot = Some(self.topology());
        self.enumeration = Some(universe);
        Ok(routes)
    }
    pub fn inspect_enumeration(&mut self, cutoff: f64) -> Result<Option<Vec<crate::Route>>> {
        self.refresh_cuts()?;
        let enumerated_kpath = self.enumerated_kpath_inputs();
        if let Some(universe) = &mut self.enumeration {
            universe.paths.inspect_with_knapsack(
                &self.graph,
                cutoff,
                &self.applied_cuts.0,
                &enumerated_kpath,
                &self.knapsack_pricing,
            )?;
            return Ok(Some(universe.paths.routes(&self.graph)?));
        }
        Ok(None)
    }
    pub fn fix(&mut self, threshold: f64) -> Result<fixing::FixingReport> {
        self.complete_labels(true)?;
        self.engine
            .as_mut()
            .ok_or_else(|| invalid("pricing must precede fixing"))?
            .fix_bucket_arcs(&self.graph, threshold)
    }
}
