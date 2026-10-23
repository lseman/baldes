use super::*;

impl CapacitySeparator {
    pub fn statistics(&self) -> CapacityStatistics {
        self.state
            .as_ref()
            .map_or_else(CapacityStatistics::default, |(state, _, _)| {
                CapacityStatistics {
                    columns: state.num_columns,
                    global_subsets: state.global_subsets.len(),
                    ..state.statistics
                }
            })
    }

    /// `demands` has n + 1 entries; index 0 (the depot) is ignored.
    pub fn new(mut demands: Vec<f64>, capacity: f64) -> Result<Self> {
        if demands.len() < 2 || demands.len() > 1024 || !capacity.is_finite() || capacity <= 0. {
            return Err(invalid("capacity dimensions or capacity"));
        }
        if demands.iter().any(|d| !d.is_finite() || *d < 0.) {
            return Err(invalid("capacity demand"));
        }
        demands[0] = 0.;
        Ok(Self {
            parameters: CapacityParameters::default(),
            demands,
            capacity,
            windows: None,
            state: None,
            feasibility: None,
        })
    }

    pub fn num_customers(&self) -> usize {
        self.demands.len() - 1
    }

    pub fn set_time_windows(&mut self, windows: TimeWindows) -> Result<()> {
        let n = self.demands.len();
        if windows.ready.len() != n
            || windows.due.len() != n
            || windows.service.len() != n
            || windows.travel.len() != n
        {
            return Err(invalid("time window dimensions"));
        }
        self.windows = Some(windows);
        self.state = None;
        self.feasibility = None;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    pub(super) fn build(&self, use_windows: bool) -> Result<Separation> {
        let p = &self.parameters;
        if p.max_cuts_per_round == 0
            || p.max_cuts_per_round > i32::MAX as usize
            || p.candidate_limit > i32::MAX as usize
            || p.max_two_path_set_size < 1
            || p.pulse_buckets < 1
            || !p.violation_tolerance.is_finite()
            || p.violation_tolerance < 0.
            || !p.shrink_threshold.is_finite()
        {
            return Err(invalid("capacity parameters"));
        }
        let params = Params {
            max_num_per_round: p.max_cuts_per_round as i32,
            max_kohl_set_size: p.max_two_path_set_size,
            complement_form: p.complement_form,
            violation_tolerance: p.violation_tolerance,
            max_contracted_group_size: 1_000_000,
            contraction_selection: 1,
            greedy_selection: 0,
            shrink_threshold: p.shrink_threshold,
            use_global_pool: p.use_global_pool,
            max_component_removals: p.candidate_limit as i32,
        };
        let windows = if use_windows {
            self.windows.as_ref()
        } else {
            None
        };
        Separation::prepare(
            &self.demands[1..],
            super::super::shared::rhs_rounding_capacity(self.capacity, self.num_customers()),
            params,
            p.pulse_buckets,
            windows,
        )
    }

    /// The separation state, rebuilt (pool and numbering lost) when the
    /// parameters or the time-window use changed.
    pub(super) fn prepared(&mut self) -> Result<&mut Separation> {
        let use_windows = self.windows.is_some() && self.parameters.use_two_path;
        let fresh =
            !matches!(&self.state, Some((_, p, w)) if *p == self.parameters && *w == use_windows);
        if fresh {
            let state = self.build(use_windows)?;
            self.state = Some((state, self.parameters.clone(), use_windows));
        }
        Ok(&mut self.state.as_mut().unwrap().0)
    }

    /// Whether one route can serve the customers within the time windows.
    pub fn time_feasible(&mut self, customers: &[usize]) -> Result<bool> {
        if self.windows.is_none() {
            return Ok(true);
        }
        let n = self.num_customers();
        let subset: Vec<i32> = customers
            .iter()
            .map(|&v| {
                if v < 1 || v > n {
                    Err(invalid("capacity customer"))
                } else {
                    Ok(v as i32 - 1)
                }
            })
            .collect::<Result<_>>()?;
        if let Some((state, _, true)) = &mut self.state {
            return Ok(state.pulse.is_subset_time_feasible(&subset));
        }
        if self.feasibility.is_none() {
            self.feasibility = Some(self.build(true)?);
        }
        Ok(self
            .feasibility
            .as_mut()
            .unwrap()
            .pulse
            .is_subset_time_feasible(&subset))
    }

    /// Separates on fractional columns (customer sequences, depot implicit).
    pub fn separate(&mut self, columns: &[(Vec<usize>, f64)]) -> Result<Vec<CapacityCut>> {
        if !self.parameters.enabled {
            return Ok(vec![]);
        }
        let n = self.num_customers();
        let (sink, width) = (n + 1, n + 2);
        let mut routes = vec![];
        for (customers, value) in columns {
            if !value.is_finite() || *value < 0. {
                return Err(invalid("column value"));
            }
            let mut arcs = vec![];
            let mut previous = 0;
            for &v in customers {
                if v < 1 || v > n || v == previous {
                    return Err(invalid("capacity route customer"));
                }
                arcs.push((previous * width + v) as i32);
                previous = v;
            }
            arcs.push((previous * width + sink) as i32);
            routes.push((arcs, *value));
        }
        let capacity =
            super::super::shared::rhs_rounding_capacity(self.capacity, self.num_customers());
        let state = self.prepared()?;
        let first_id = state.statistics.cuts_created;
        let generated = state.separate(&routes);
        let positive = state.num_positive_demand as i32;
        let demands = &self.demands;
        let mut result = vec![];
        for (offset, (rhs, kind, sets, weights)) in generated.into_iter().enumerate() {
            let mut cut = CapacityCut {
                id: first_id + offset,
                customers: sets.iter().map(|s| s + 1).collect(),
                other: vec![],
                kind: CapacityKind::Capacity,
                rhs,
                k: 0,
                demand: 0.,
                violation: 0.,
                edges: BTreeMap::new(),
            };
            for &c in &cut.customers {
                cut.demand += demands[c];
            }
            for (variable, weight) in weights {
                let tail = variable as usize / width;
                let mut head = variable as usize % width;
                if head == sink {
                    head = 0;
                }
                cut.edges.insert((tail.min(head), tail.max(head)), weight);
            }
            let rounded_k = (cut.demand / capacity).ceil() as i32;
            if kind == 0 {
                cut.k = rhs / 2;
                cut.kind = if cut.k > rounded_k {
                    CapacityKind::TwoPath
                } else {
                    CapacityKind::Capacity
                };
            } else {
                cut.kind = CapacityKind::Complement;
                let size = cut.customers.len() as i32;
                if size > positive / 2 {
                    for v in 1..=n {
                        if demands[v] > 1e-6 && cut.customers.binary_search(&v).is_err() {
                            cut.other.push(v);
                        }
                    }
                    cut.k = positive - size - rhs;
                } else {
                    cut.other = cut.customers.clone();
                    cut.k = size - rhs;
                }
            }
            let mut lhs = 0.;
            for (customers, value) in columns {
                lhs += (value * 1e8 + 0.5).floor() / 1e8 * cut.coefficient(customers) as f64;
            }
            cut.violation = if cut.greater_equal() {
                cut.rhs as f64 - lhs
            } else {
                lhs - cut.rhs as f64
            };
            result.push(cut);
        }
        Ok(result)
    }
}
