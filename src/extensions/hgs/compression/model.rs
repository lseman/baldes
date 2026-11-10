use super::super::{local_search::TimeWindowData, Client, Individual, Params};

pub(crate) type Routes = Vec<Vec<usize>>;

/// Intersection across all feasible observations in one evolutionary epoch.
pub(crate) struct Consensus {
    reference: Option<(Vec<usize>, Vec<usize>)>,
    stable_next: Vec<bool>,
    stable_previous: Vec<bool>,
}
impl Consensus {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            reference: None,
            stable_next: vec![true; n + 1],
            stable_previous: vec![true; n + 1],
        }
    }
    pub(crate) fn started(&self) -> bool {
        self.reference.is_some()
    }
    pub(crate) fn observe(&mut self, individual: &Individual<'_>) {
        if let Some((next, previous)) = &self.reference {
            for i in 1..next.len() {
                self.stable_next[i] &= next[i] == individual.successors[i];
                self.stable_previous[i] &= previous[i] == individual.predecessors[i];
            }
        } else {
            self.reference = Some((
                individual.successors.clone(),
                individual.predecessors.clone(),
            ));
        }
    }
    fn groups(&self) -> Option<(Routes, Routes)> {
        let (next, previous) = self.reference.as_ref()?;
        let linked = |i: usize, j: usize| {
            j != 0
                && self.stable_next[i]
                && self.stable_previous[j]
                && next[i] == j
                && previous[j] == i
        };
        let mut seen = vec![false; next.len()];
        let mut active = Vec::new();
        let mut fixed = Vec::new();
        for first in 1..next.len() {
            if previous[first] != 0 && linked(previous[first], first) {
                continue;
            }
            let mut chain = vec![first];
            seen[first] = true;
            let mut last = first;
            while linked(last, next[last]) && !seen[next[last]] {
                last = next[last];
                seen[last] = true;
                chain.push(last);
            }
            if self.stable_previous[first]
                && previous[first] == 0
                && self.stable_next[last]
                && next[last] == 0
            {
                fixed.push(chain);
            } else {
                active.push(chain);
            }
        }
        // Cycles cannot occur in a valid routed individual; never contract them.
        for (i, &visited) in seen.iter().enumerate().skip(1) {
            if !visited {
                active.push(vec![i]);
            }
        }
        Some((active, fixed))
    }
    pub(crate) fn plan(&self, params: &Params) -> Option<Plan> {
        let (active, fixed) = self.groups()?;
        Plan::build(params, active, fixed)
    }
}

pub(crate) struct Plan {
    pub(crate) chains: Routes,
    pub(crate) fixed: Routes,
    pub(crate) frozen_distance: f64,
    starts: Vec<Option<(bool, usize)>>,
}
impl Plan {
    pub(super) fn build(params: &Params, chains: Routes, fixed: Routes) -> Option<Self> {
        if params.penalty_wait_time != 0.
            || params.is_duration_constrained
            || params.duals_ptr.is_some()
            || params.forbidden_edges_set.is_some()
        {
            return None;
        }
        let mut active = Vec::new();
        for chain in chains {
            let seq = sequence(params, &chain);
            if chain.iter().any(|&v| params.cli[v].n_tw != 0)
                || seq.time_warp != 0.
                || !seq.duration.is_finite()
                || seq.earliest_arrival > seq.latest_arrival
            {
                active.extend(chain.into_iter().map(|v| vec![v]));
            } else {
                active.push(chain);
            }
        }
        if active.len() >= params.nb_clients as usize {
            return None;
        }
        if !active.is_empty() && fixed.len() >= params.nb_vehicles as usize {
            return None;
        }
        let mut starts = vec![None; params.nb_clients as usize + 1];
        for (fixed_flag, groups) in [(false, &active), (true, &fixed)] {
            for (i, chain) in groups.iter().enumerate() {
                starts[chain[0]] = Some((fixed_flag, i));
            }
        }
        let mut replay = Individual::new_empty(params);
        let mut frozen_distance = 0.;
        for route in &fixed {
            replay.chrom_r[0].clone_from(route);
            replay.evaluate_complete_cost();
            if !replay.is_feasible || replay.my_cost_sol.time_warp != 0. {
                return None;
            }
            frozen_distance += replay.my_cost_sol.distance;
        }
        Some(Self {
            chains: active,
            fixed,
            frozen_distance,
            starts,
        })
    }

    pub(crate) fn project(&self, routes: &[Vec<usize>]) -> Option<Routes> {
        let mut result = Vec::new();
        for route in routes.iter().filter(|r| !r.is_empty()) {
            let (fixed, id) = self.starts.get(route[0]).copied().flatten()?;
            if fixed {
                if *route != self.fixed[id] {
                    return None;
                }
                continue;
            }
            let mut projected = Vec::new();
            let mut position = 0;
            while position < route.len() {
                let (fixed, id) = self.starts.get(route[position]).copied().flatten()?;
                if fixed {
                    return None;
                }
                let chain = &self.chains[id];
                if route.get(position..position + chain.len())? != chain {
                    return None;
                }
                projected.push(id + 1);
                position += chain.len();
            }
            result.push(projected);
        }
        Some(result)
    }

    pub(crate) fn parameters(&self, source: &Params) -> Option<Params> {
        if self.chains.is_empty() {
            return None;
        }
        let mut clients = vec![source.cli[0].clone()];
        let mut entry_cost = vec![0.];
        for (index, chain) in self.chains.iter().enumerate() {
            if chain.len() == 1 {
                let mut client = source.cli[chain[0]].clone();
                client.cust_num = (index + 1) as i32;
                clients.push(client);
                entry_cost.push(0.);
                continue;
            }
            let seq = sequence(source, chain);
            let internal: f64 = chain
                .windows(2)
                .map(|a| source.time_cost.get(a[0], a[1]))
                .sum();
            let demand: i64 = chain.iter().map(|&v| i64::from(source.cli[v].demand)).sum();
            let length = chain.len() as f64;
            // Charge internal distance on entry, shift the macro window by the
            // same amount, and subtract it from service. One matrix then retains
            // BOTH objective cost and scheduling, without free travel shortcuts.
            clients.push(Client {
                cust_num: (index + 1) as i32,
                coord_x: chain.iter().map(|&v| source.cli[v].coord_x / length).sum(),
                coord_y: chain.iter().map(|&v| source.cli[v].coord_y / length).sum(),
                demand: demand.try_into().ok()?,
                service_duration: (seq.duration - internal).max(0.),
                earliest_arrival: seq.earliest_arrival + internal,
                latest_arrival: seq.latest_arrival + internal,
                release_time: seq.latest_release_time,
                ..Client::default()
            });
            entry_cost.push(internal);
        }
        let n = clients.len();
        let first = |v: usize| if v == 0 { 0 } else { self.chains[v - 1][0] };
        let last = |v: usize| {
            if v == 0 {
                0
            } else {
                *self.chains[v - 1].last().unwrap()
            }
        };
        let distances = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            0.
                        } else {
                            source.time_cost.get(last(i), first(j)) + entry_cost[j]
                        }
                    })
                    .collect()
            })
            .collect();
        let fleet = source.nb_vehicles - self.fixed.len() as i32;
        let mut params = Params::build(
            clients,
            fleet,
            source.vehicle_capacity,
            source.problem_type,
            Some(&distances),
        );
        params.inherit_search_runtime(source);
        Some(params)
    }
}
fn sequence(params: &Params, chain: &[usize]) -> TimeWindowData {
    let mut seq = TimeWindowData::single(chain[0] as i32, params);
    for &v in &chain[1..] {
        seq = TimeWindowData::merge(&seq, &TimeWindowData::single(v as i32, params), params);
    }
    seq
}

pub(crate) struct Expansion {
    pub(crate) chains: Routes,
    pub(crate) fixed: Routes,
}
impl Expansion {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            chains: (0..=n)
                .map(|v| if v == 0 { Vec::new() } else { vec![v] })
                .collect(),
            fixed: Vec::new(),
        }
    }
    pub(crate) fn expand(&self, routes: &[Vec<usize>]) -> Routes {
        routes
            .iter()
            .filter(|r| !r.is_empty())
            .map(|r| {
                r.iter()
                    .flat_map(|&v| self.chains[v].iter().copied())
                    .collect()
            })
            .chain(self.fixed.iter().cloned())
            .collect()
    }
    pub(crate) fn contract(&self, plan: &Plan) -> Self {
        let mut fixed = self.fixed.clone();
        fixed.extend(plan.fixed.iter().map(|r| {
            r.iter()
                .flat_map(|&v| self.chains[v].iter().copied())
                .collect()
        }));
        let mut chains = vec![Vec::new()];
        chains.extend(plan.chains.iter().map(|r| {
            r.iter()
                .flat_map(|&v| self.chains[v].iter().copied())
                .collect()
        }));
        Self { chains, fixed }
    }
}
