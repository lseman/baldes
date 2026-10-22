//! Strong k-path cut separation, ported from the C++ reference
//! `native/separation/StrongKPathSeparation.h`, its standalone adapter
//! `native/adapters/StrongKPathApiAdapter.h`, and `api/CutOptions.h`.
pub mod preparation;
pub use preparation::{KPathGraph, KPathPreparation};

use std::collections::{BTreeMap, BTreeSet};

use super::capacity::TimeWindows;
use super::shared::bits::Bits1024;
use super::shared::mincut::MinCutDigraph;
use super::shared::pulse::PulseDigraph;
use super::shared::stdhash::UnorderedSet;
use super::shared::stdsort;
use crate::core::options::Options;
use crate::{invalid, Result};

/// Separator parameters for strong k-path cuts.
#[derive(Clone, Debug, PartialEq)]
pub struct KPathParameters {
    pub max_cuts_per_round: usize,
    pub violation_tolerance: f64,
    pub max_two_path_set_size: i32,
    /// Maximum unique Kohl search states per separation call; zero skips it.
    pub max_two_path_states: usize,
    pub pulse_buckets: i32,
    pub candidate_limit: usize,
    pub use_two_path: bool,
}

impl Default for KPathParameters {
    fn default() -> Self {
        Self {
            max_cuts_per_round: 100,
            violation_tolerance: 1e-6,
            max_two_path_set_size: 15,
            max_two_path_states: 10_000,
            pulse_buckets: 10,
            candidate_limit: 1000,
            use_two_path: true,
        }
    }
}

/// Configure cuts per round from BALDESrankOneCutsMaxNumPerRound.
pub fn kpath_parameters(options: &Options, mut base: KPathParameters) -> Result<KPathParameters> {
    let per_round = options.get("BALDESrankOneCutsMaxNumPerRound")?;
    if per_round < 1.0 {
        return Err(invalid("BALDESrankOneCutsMaxNumPerRound must be positive"));
    }
    base.max_cuts_per_round = per_round as usize;
    Ok(base)
}

/// A separated strong k-path cut with depot 0 and customers 1..n.
#[derive(Clone, Debug, PartialEq)]
pub struct StrongKPathCut {
    pub customers: Vec<usize>,
    pub memory: Vec<usize>,
    pub k: i32,
    pub rhs: i32,
    pub two_path: bool,
    pub demand: f64,
    pub weight: f64,
    pub violation: f64,
    pub id: i32,
    customer_count: usize,
}

impl StrongKPathCut {
    /// Route coefficient for an elementary customer sequence (1..=n).
    pub fn coefficient(&self, route: &[usize]) -> Result<i32> {
        // The standalone adapter validates the whole sequence before invoking
        // native coefficient(), even if an earlier customer already hits S.
        if route.iter().any(|&v| v < 1 || v > self.customer_count) {
            return Err(invalid("k-path route customer"));
        }
        Ok(route
            .iter()
            .any(|v| self.customers.binary_search(v).is_ok()) as i32)
    }

    /// Limited-memory coefficient on a physical vertex sequence, optionally including terminals.
    pub fn coefficient_with_memory(&self, route: &[usize], enumerated: bool) -> Result<i32> {
        if route.iter().any(|&v| v > self.customer_count + 1) {
            return Err(invalid("k-path route customer"));
        }
        let mut inside = false;
        let mut count = 0i32;
        for &v in route {
            if self.customers.binary_search(&v).is_ok() {
                if inside {
                    continue;
                }
                if enumerated {
                    return Ok(1);
                }
                count += 1;
                inside = true;
            } else if inside {
                inside = self.memory.binary_search(&v).is_ok();
            }
        }
        Ok(count)
    }
}

#[derive(Clone, Debug)]
struct Params {
    max_num_per_round: i32,
    max_kohl_set_size: i32,
    max_kohl_states: usize,
    num_pulse_buckets: i32,
    violation_tolerance: f64,
    max_shrink_removals: i32,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Candidate {
    heuristic: i32,
    subset: Vec<bool>,
    violation: f64,
    lhs: f64,
    k: i32,
}

impl Candidate {
    fn new(heuristic: i32, n: usize) -> Self {
        Self {
            heuristic,
            subset: vec![false; n],
            violation: 0.0,
            lhs: 0.0,
            k: 0,
        }
    }

    fn clear(&mut self) {
        self.subset.iter_mut().for_each(|b| *b = false);
        self.violation = 0.0;
        self.lhs = 0.0;
        self.k = 0;
    }

    fn test(&self, i: usize) -> bool {
        self.subset.get(i).copied().unwrap_or(false)
    }

    fn set(&mut self, i: usize) {
        if let Some(b) = self.subset.get_mut(i) {
            *b = true;
        }
    }

    fn reset(&mut self, i: usize) {
        if let Some(b) = self.subset.get_mut(i) {
            *b = false;
        }
    }
}

/// KPathCandidateSet: ordered by subset; an equal subset is not replaced.
#[derive(Default)]
struct CandidateSet(BTreeMap<Vec<bool>, Candidate>);

impl CandidateSet {
    fn insert(&mut self, c: &Candidate) {
        self.0.entry(c.subset.clone()).or_insert_with(|| c.clone());
    }
}

#[derive(Clone, Debug)]
struct ColumnInfo {
    column: usize,
    value: f64,
    flag: bool,
    customers: Vec<usize>,
    vertices: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RouteColumn {
    id: usize,
    value: f64,
    customers: Vec<usize>,
    key: f64,
}

#[derive(Clone, Debug, Default)]
struct GreedyCandidate {
    id: usize,
    violation: f64,
    delta: f64,
    explored: bool,
}

struct TwoPathCandidate {
    size: i32,
    cut_weight: f64,
    demand: f64,
    column_touched: Vec<bool>,
    touched_columns: Vec<usize>,
    customers: Bits1024,
}

fn greedy_less(a: &GreedyCandidate, b: &GreedyCandidate) -> bool {
    const EPS: f64 = 1e-6;
    if a.explored != b.explored {
        return a.explored;
    }
    if b.violation - EPS > a.violation {
        return true;
    }
    if a.violation > b.violation + EPS {
        return false;
    }
    if b.delta - EPS > a.delta {
        return true;
    }
    if a.delta > b.delta + EPS {
        return false;
    }
    a.id < b.id
}

#[derive(Clone)]
struct Separation {
    params: Params,
    n: usize,
    demands: Vec<f64>,
    capacity: f64,
    two_path: bool,
    next_cut_id: i32,
    columns: Vec<ColumnInfo>,
    customer_columns: Vec<Vec<usize>>,
    num_columns: usize,
    two_path_subsets: UnorderedSet,
    pulse: PulseDigraph,
}

/// Standalone strong k-path cut separator: depot 0, customers 1..n.
#[derive(Clone, Debug)]
pub struct KPathSeparator {
    pub parameters: KPathParameters,
    demands: Vec<f64>,
    capacity: f64,
    windows: Option<TimeWindows>,
    next_id: i32,
}

impl KPathSeparator {
    /// `demands` has n + 1 entries; index 0 (the depot) is set to 0.
    pub fn new(mut demands: Vec<f64>, capacity: f64) -> Result<Self> {
        if demands.len() < 2 || demands.len() > 1024 || !capacity.is_finite() || capacity <= 0.0 {
            return Err(invalid("native k-path dimensions or capacity"));
        }
        if demands.iter().any(|&d| !d.is_finite() || d < 0.0) {
            return Err(invalid("native k-path demand"));
        }
        demands[0] = 0.0;
        Ok(Self {
            parameters: KPathParameters::default(),
            demands,
            capacity,
            windows: None,
            next_id: 0,
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
            || windows.travel.iter().any(|row| row.len() != n)
        {
            return Err(invalid("time window dimensions"));
        }
        for v in 0..n {
            if !windows.ready[v].is_finite()
                || !windows.due[v].is_finite()
                || windows.ready[v] > windows.due[v]
                || !windows.service[v].is_finite()
                || windows.service[v] < 0.0
            {
                return Err(invalid("time window bounds"));
            }
            for &travel in &windows.travel[v] {
                if !travel.is_finite() || travel < 0.0 {
                    return Err(invalid("travel time"));
                }
            }
        }
        self.windows = Some(windows);
        Ok(())
    }

    pub fn has_time_windows(&self) -> bool {
        self.windows.is_some()
    }

    pub fn reset(&mut self) {
        self.next_id = 0;
    }

    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(invalid("native k-path parameter must be finite"));
        }
        match name {
            "max_cuts_per_round" => {
                if value < 1.0 || value > (i32::MAX as f64) {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.max_cuts_per_round = value as usize;
            }
            "violation_tolerance" => {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.violation_tolerance = value;
            }
            "max_two_path_set_size" => {
                if value < 1.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.max_two_path_set_size = value as i32;
            }
            "max_two_path_states" => {
                if !value.is_finite()
                    || value < 0.0
                    || value.fract() != 0.0
                    || value > i32::MAX as f64
                {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.max_two_path_states = value as usize;
            }
            "pulse_buckets" => {
                if value < 1.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.pulse_buckets = value as i32;
            }
            "candidate_limit" => {
                if value < 0.0 || value > (i32::MAX as f64) {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.candidate_limit = value as usize;
            }
            "use_two_path" => {
                self.parameters.use_two_path = value != 0.0;
            }
            _ => return Err(invalid(format!("unknown k-path parameter: {name}"))),
        }
        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Result<f64> {
        match name {
            "max_cuts_per_round" => Ok(self.parameters.max_cuts_per_round as f64),
            "violation_tolerance" => Ok(self.parameters.violation_tolerance),
            "max_two_path_set_size" => Ok(self.parameters.max_two_path_set_size as f64),
            "max_two_path_states" => Ok(self.parameters.max_two_path_states as f64),
            "pulse_buckets" => Ok(self.parameters.pulse_buckets as f64),
            "candidate_limit" => Ok(self.parameters.candidate_limit as f64),
            "use_two_path" => Ok(if self.parameters.use_two_path {
                1.0
            } else {
                0.0
            }),
            _ => Err(invalid(format!("unknown k-path parameter: {name}"))),
        }
    }

    pub fn configure(&mut self, options: &Options) -> Result<()> {
        self.parameters = kpath_parameters(options, self.parameters.clone())?;
        Ok(())
    }

    pub fn separate(&mut self, columns: &[(Vec<usize>, f64)]) -> Result<Vec<StrongKPathCut>> {
        self.separate_impl(columns, &vec![false; columns.len()])
    }

    /// Native master-solution adaptation: flag_04 excludes enumerated columns
    /// from memory construction, but not from the separation score.
    pub fn separate_fractional(
        &mut self,
        columns: &[super::FractionalColumn],
    ) -> Result<Vec<StrongKPathCut>> {
        let routes: Vec<_> = columns
            .iter()
            .map(|c| (c.vertices.clone(), c.value))
            .collect();
        let flags: Vec<_> = columns.iter().map(|c| c.enumerated).collect();
        self.separate_impl(&routes, &flags)
    }

    fn separate_impl(
        &mut self,
        columns: &[(Vec<usize>, f64)],
        flags: &[bool],
    ) -> Result<Vec<StrongKPathCut>> {
        let p = &self.parameters;
        if p.max_cuts_per_round == 0
            || p.max_cuts_per_round > (i32::MAX as usize)
            || p.candidate_limit > (i32::MAX as usize)
            || p.max_two_path_states > i32::MAX as usize
            || p.max_two_path_set_size < 1
            || p.pulse_buckets < 1
            || !p.violation_tolerance.is_finite()
            || p.violation_tolerance < 0.0
        {
            return Err(invalid("native k-path parameters"));
        }

        let count = self.num_customers();
        let sink = count + 1;
        let width = count + 2;

        let params = Params {
            max_num_per_round: p.max_cuts_per_round as i32,
            max_kohl_set_size: p.max_two_path_set_size,
            max_kohl_states: p.max_two_path_states,
            num_pulse_buckets: p.pulse_buckets,
            violation_tolerance: p.violation_tolerance,
            max_shrink_removals: p.candidate_limit as i32,
        };

        let use_windows = if self.windows.is_some() && p.use_two_path {
            self.windows.as_ref()
        } else {
            None
        };

        let mut state =
            Separation::prepare(&self.demands[1..], self.capacity, params, use_windows)?;
        state.next_cut_id = self.next_id;

        let mut routes = Vec::with_capacity(columns.len());
        for (column, (customers, value)) in columns.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 {
                return Err(invalid("column value"));
            }
            let mut arcs = Vec::with_capacity(customers.len() + 1);
            let mut previous = 0usize;
            for &v in customers {
                if v < 1 || v > count || v == previous {
                    return Err(invalid("native k-path route customer"));
                }
                arcs.push((previous * width + v) as i32);
                previous = v;
            }
            arcs.push((previous * width + sink) as i32);
            routes.push((arcs, *value, flags[column]));
        }

        let generated = state.separate(&routes);
        self.next_id = state.next_cut_id;

        let mut result = Vec::with_capacity(generated.len());
        for (id, k, set_ids, memory_set) in generated {
            let mut customers = Vec::with_capacity(set_ids.len());
            let mut demand = 0.0;
            for row in set_ids {
                customers.push(row + 1);
                demand += self.demands[row + 1];
            }
            let two_path = (k as f64) > (demand / self.capacity).ceil();
            let memory: Vec<usize> = memory_set.into_iter().collect();
            let mut cut = StrongKPathCut {
                customers,
                memory,
                k,
                rhs: k,
                two_path,
                demand,
                weight: 0.0,
                violation: 0.0,
                id,
                customer_count: count,
            };
            for (route_customers, raw_val) in columns {
                let val = (raw_val * 1e8 + 0.5).floor() / 1e8;
                let coeff = cut.coefficient(route_customers)? as f64;
                cut.weight += val * coeff;
            }
            cut.violation = (k as f64) - cut.weight;
            result.push(cut);
        }
        Ok(result)
    }
}

mod components;
mod exact;
mod greedy;
mod separation;
mod setup;
mod two_path;

mod native;
pub use native::{NativeKPathCut, NativeKPathSeparator};
