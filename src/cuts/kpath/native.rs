use super::*;

/// Original set IDs and graph-specific physical vertex memories. This is the
/// generic native cut, before adaptation to the standalone customer API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeKPathCut {
    pub id: i32,
    pub rhs: i32,
    pub set_ids: Vec<usize>,
    pub memory: BTreeMap<i32, BTreeSet<usize>>,
    pub arc_memory: bool,
}

/// General StrongKPathCutSeparation: uses original graph IDs and preprocessed
/// arc incidence, including covering sets and sparse physical vertex IDs.
#[derive(Clone, Debug)]
pub struct NativeKPathSeparator {
    pub parameters: KPathParameters,
    pub preparation: KPathPreparation,
    pub next_cut_id: i32,
    pub num_separate_calls: usize,
}
impl NativeKPathSeparator {
    pub fn new(use_packing_sets: bool) -> Self {
        Self {
            parameters: KPathParameters::default(),
            preparation: KPathPreparation::new(use_packing_sets, 10),
            next_cut_id: 0,
            num_separate_calls: 0,
        }
    }
    pub fn prepare(
        &mut self,
        graphs: &[KPathGraph],
        demands: &[f64],
        capacity: f64,
        two_path_resource: i32,
    ) -> Result<()> {
        self.preparation.pulse.num_buckets = self.parameters.pulse_buckets;
        self.preparation
            .prepare(graphs, demands, capacity, two_path_resource)
    }
    pub fn separate(
        &mut self,
        solutions: &[crate::cuts::rank_one::separation::Solution],
        values: &[f64],
    ) -> Result<Vec<NativeKPathCut>> {
        let prep = &self.preparation;
        let n = prep.num_customers;
        if n == 0 {
            return Ok(vec![]);
        }
        if solutions.len() != values.len() {
            return Err(invalid("k-path path/value dimensions"));
        }
        self.num_separate_calls += 1;
        let p = &self.parameters;
        let params = Params {
            max_num_per_round: p.max_cuts_per_round as i32,
            max_kohl_set_size: p.max_two_path_set_size,
            max_kohl_states: p.max_two_path_states,
            num_pulse_buckets: p.pulse_buckets,
            violation_tolerance: p.violation_tolerance,
            max_shrink_removals: p.candidate_limit as i32,
        };
        let capacity = super::super::shared::rhs_rounding_capacity(prep.capacity, n);
        let mut state = Separation::prepare(&prep.demands[..n], capacity, params, None)?;
        // The prepared pulse graph is restored on both success and rejected
        // column input. Its mutable search scratch already persists between
        // successful calls, so transfer ownership instead of copying matrices.
        std::mem::swap(&mut state.pulse, &mut self.preparation.pulse);
        let result = self.separate_prepared(solutions, values, &mut state);
        std::mem::swap(&mut state.pulse, &mut self.preparation.pulse);
        result
    }

    fn separate_prepared(
        &mut self,
        solutions: &[crate::cuts::rank_one::separation::Solution],
        values: &[f64],
        state: &mut Separation,
    ) -> Result<Vec<NativeKPathCut>> {
        let prep = &self.preparation;
        let n = prep.num_customers;
        state.two_path = prep.two_path_resource >= 0;
        state.num_columns = solutions.len();
        for (i, (path, &value)) in solutions.iter().zip(values).enumerate() {
            let info = usize::try_from(path.graph_id)
                .ok()
                .and_then(|g| prep.preprocessed.get(g))
                .and_then(Option::as_ref)
                .ok_or_else(|| invalid("k-path solution graph is not preprocessed"))?;
            if path.arc_ids.is_empty() {
                // The original retains input indices after skipping empty
                // paths. A hole before a later column can index past columns.
                if solutions[i + 1..].iter().any(|p| !p.arc_ids.is_empty()) {
                    return Err(invalid("native k-path empty-column index gap"));
                }
                continue;
            }
            let sets = if prep.use_packing_sets {
                &info.arc_packing_sets
            } else {
                &info.arc_covering_sets
            };
            let mut customers = vec![n];
            let mut vertices = vec![];
            for (j, &arc) in path.arc_ids.iter().enumerate() {
                let arc = usize::try_from(arc).map_err(|_| invalid("negative k-path arc ID"))?;
                let ids = sets
                    .get(arc)
                    .ok_or_else(|| invalid("k-path arc incidence"))?;
                let e = ids
                    .first()
                    .copied()
                    .filter(|&e| e >= 0 && (e as usize) < n)
                    .map_or(n, |e| e as usize);
                customers.push(e);
                if j == 0 {
                    vertices.push(
                        *info
                            .arc_tail
                            .get(arc)
                            .ok_or_else(|| invalid("k-path arc tail"))?
                            as usize,
                    );
                }
                vertices.push(
                    *info
                        .arc_head
                        .get(arc)
                        .ok_or_else(|| invalid("k-path arc head"))? as usize,
                );
            }
            state.columns.push(ColumnInfo {
                column: i,
                value: (value * 1e8 + 0.5).floor() / 1e8,
                flag: path.enumerated,
                customers,
                vertices,
            });
        }
        let candidates = state.generate_candidates();
        let mut cuts = vec![];
        for c in candidates {
            let mut memory: BTreeMap<i32, BTreeSet<usize>> = prep
                .preprocessed
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_some())
                .map(|(g, _)| (g as i32, BTreeSet::new()))
                .collect();
            for col in &state.columns {
                if col.flag {
                    continue;
                }
                let mut pending = BTreeSet::new();
                let mut inside = false;
                for (&element, &vertex) in col.customers.iter().zip(&col.vertices) {
                    if element != n && c.test(element) {
                        memory
                            .entry(solutions[col.column].graph_id)
                            .or_default()
                            .append(&mut pending);
                        inside = true;
                    } else if inside {
                        pending.insert(vertex);
                    }
                }
            }
            cuts.push(NativeKPathCut {
                id: self.next_cut_id,
                rhs: c.k,
                set_ids: (0..n).filter(|&e| c.test(e)).collect(),
                memory,
                arc_memory: false,
            });
            self.next_cut_id += 1;
        }
        Ok(cuts)
    }
    pub fn coefficient(
        &self,
        path: &crate::cuts::rank_one::separation::Solution,
        cut: &NativeKPathCut,
    ) -> Result<i32> {
        let memory = cut.memory.get(&path.graph_id);
        if path.arc_ids.is_empty() || (memory.is_none() && !path.enumerated) {
            return Ok(0);
        }
        let info = usize::try_from(path.graph_id)
            .ok()
            .and_then(|g| self.preparation.preprocessed.get(g))
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid("k-path solution graph is not preprocessed"))?;
        let sets = if self.preparation.use_packing_sets {
            &info.arc_packing_sets
        } else {
            &info.arc_covering_sets
        };
        let mut inside = false;
        let mut coefficient = 0;
        for &arc in &path.arc_ids {
            let a = usize::try_from(arc).map_err(|_| invalid("negative k-path arc ID"))?;
            let ids = sets.get(a).ok_or_else(|| invalid("k-path arc incidence"))?;
            let hit = ids.first().is_some_and(|&e| {
                e >= 0
                    && (e as usize) < self.preparation.num_customers
                    && cut.set_ids.binary_search(&(e as usize)).is_ok()
            });
            if hit {
                if inside {
                    continue;
                }
                if path.enumerated {
                    return Ok(1);
                }
                coefficient += 1;
                inside = true;
            } else if inside {
                let key = if cut.arc_memory {
                    a
                } else {
                    *info
                        .arc_head
                        .get(a)
                        .ok_or_else(|| invalid("k-path arc head"))? as usize
                };
                inside = memory.is_some_and(|m| m.contains(&key));
            }
        }
        Ok(coefficient)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/kpath/native_rounding_tests.rs"]
mod rounding_tests;
