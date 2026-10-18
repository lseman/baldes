use super::super::{
    pulse::PulseDigraph,
    separation::{GraphData, Incidence},
};
use crate::{invalid, Result};
use std::collections::BTreeMap;

/// Original graph and its preprocessing, with resource values keyed by the
/// original vertex/arc and resource IDs. Missing preprocessing is significant.
#[derive(Clone, Debug, Default)]
pub struct KPathGraph {
    pub graph: GraphData,
    pub incidence: Option<Incidence>,
    pub vertex_lower: BTreeMap<i32, BTreeMap<i32, f64>>,
    pub vertex_upper: BTreeMap<i32, BTreeMap<i32, f64>>,
    pub arc_consumption: BTreeMap<i32, BTreeMap<i32, f64>>,
}

#[derive(Clone, Debug)]
pub struct KPathPreparation {
    pub use_packing_sets: bool,
    pub max_graph_id: i32,
    pub preprocessed: Vec<Option<Incidence>>,
    pub num_customers: usize,
    pub num_positive_demand: usize,
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub two_path_resource: i32,
    pub pulse: PulseDigraph,
}
impl KPathPreparation {
    pub fn new(use_packing_sets: bool, pulse_buckets: i32) -> Self {
        Self {
            use_packing_sets,
            max_graph_id: -1,
            preprocessed: vec![],
            num_customers: 0,
            num_positive_demand: 0,
            demands: vec![],
            capacity: 0.,
            two_path_resource: -1,
            pulse: PulseDigraph::new(pulse_buckets),
        }
    }
    /// Preserves native mutation order and partial state on a rejected input.
    /// Native out-of-range accesses are reported as errors in this safe API.
    pub fn prepare(
        &mut self,
        graphs: &[KPathGraph],
        demands: &[f64],
        capacity: f64,
        two_path_resource: i32,
    ) -> Result<()> {
        if demands.iter().any(|&d| -1e-6 > d) {
            return Err(invalid("demands should be non-negative"));
        }
        if 1e-6 > capacity {
            return Err(invalid("capacity should be positive"));
        }
        self.capacity = capacity;
        self.two_path_resource = two_path_resource;
        self.demands = demands.to_vec();
        self.num_customers = demands.len();
        for input in graphs {
            let g = &input.graph;
            if g.id < 0 {
                return Err(invalid("negative k-path graph ID"));
            }
            self.max_graph_id = self.max_graph_id.max(g.id);
            let count = if self.use_packing_sets {
                g.packing_set_count
            } else {
                g.covering_set_count
            };
            if count >= 0 {
                self.num_customers = self.num_customers.max(count as usize);
            }
        }
        if two_path_resource >= 0 && self.max_graph_id > 0 {
            return Err(invalid("2-path cuts require graph ID zero"));
        }
        let n = self.num_customers;
        self.num_positive_demand = self.demands.iter().take(n).filter(|&&d| d > 0.).count();
        self.demands.resize(n + 1, 0.);
        self.preprocessed
            .resize((self.max_graph_id + 1) as usize, None);
        for input in graphs {
            self.preprocessed[input.graph.id as usize] = input.incidence.clone();
            if input.incidence.is_none() {
                return Err(invalid("k-path graph is not preprocessed"));
            }
        }
        let mut ready = if two_path_resource >= 0 {
            vec![1e12; n + 2]
        } else {
            vec![]
        };
        let mut due = if two_path_resource >= 0 {
            vec![-1e12; n + 2]
        } else {
            vec![]
        };
        let mut distance = vec![vec![1e12; n + 2]; n + 2];
        for input in graphs {
            let g = &input.graph;
            let info = self.preprocessed[g.id as usize].as_ref().unwrap();
            let mut elements = vec![-1; (info.max_vertex_id + 1).max(0) as usize];
            for v in &g.vertices {
                let ids = if self.use_packing_sets {
                    &v.packing_sets
                } else {
                    &v.covering_sets
                };
                if ids.len() > 1 {
                    return Err(invalid("k-path vertex belongs to multiple sets"));
                }
                let e = ids
                    .first()
                    .copied()
                    .filter(|&e| e >= 0 && (e as usize) < n && !(1e-6 > self.demands[e as usize]))
                    .map_or(n, |e| e as usize);
                if let Some(slot) = usize::try_from(v.id).ok().and_then(|v| elements.get_mut(v)) {
                    *slot = e as i32;
                }
                if e < n && (v.id == g.source || v.id == g.sink) {
                    return Err(invalid("k-path terminal belongs to an element set"));
                }
                if two_path_resource < 0 {
                    continue;
                }
                let lower = input
                    .vertex_lower
                    .get(&v.id)
                    .ok_or_else(|| invalid("missing k-path vertex lower bounds"))?;
                let upper = input
                    .vertex_upper
                    .get(&v.id)
                    .ok_or_else(|| invalid("missing k-path vertex upper bounds"))?;
                ready[e] = *lower
                    .get(&0)
                    .ok_or_else(|| invalid("missing k-path resource zero lower bound"))?;
                due[e] = *upper
                    .get(&0)
                    .ok_or_else(|| invalid("missing k-path resource zero upper bound"))?;
                let slot = if e < n {
                    Some(e)
                } else if v.id == g.source {
                    Some(n)
                } else if v.id == g.sink {
                    Some(n + 1)
                } else {
                    None
                };
                if let Some(slot) = slot {
                    let lb = *lower
                        .get(&two_path_resource)
                        .ok_or_else(|| invalid("missing k-path time lower bound"))?;
                    let ub = *upper
                        .get(&two_path_resource)
                        .ok_or_else(|| invalid("missing k-path time upper bound"))?;
                    if ready[slot] > lb {
                        ready[slot] = lb;
                    }
                    if ub > due[slot] {
                        due[slot] = ub;
                    }
                }
            }
            for a in &g.arcs {
                if !a.packing_sets.is_empty() || !a.covering_sets.is_empty() {
                    return Err(invalid("k-path arc holds packing or covering sets"));
                }
                if two_path_resource < 0 {
                    continue;
                }
                let weight = *input
                    .arc_consumption
                    .get(&a.id)
                    .and_then(|r| r.get(&two_path_resource))
                    .ok_or_else(|| invalid("missing k-path time consumption"))?;
                let element = |v: i32| -> Result<usize> {
                    elements
                        .get(usize::try_from(v).map_err(|_| invalid("negative k-path vertex"))?)
                        .copied()
                        .filter(|&e| e >= 0)
                        .map(|e| e as usize)
                        .ok_or_else(|| invalid("missing k-path arc endpoint"))
                };
                let et = element(a.tail)?;
                let eh = element(a.head)?;
                let pair = if a.tail == g.source {
                    if eh != n {
                        Some((n, eh))
                    } else if a.head == g.sink && et != n {
                        Some((et, n + 1))
                    } else {
                        None
                    }
                } else if a.head == g.sink {
                    if et != n {
                        Some((et, n + 1))
                    } else {
                        None
                    }
                } else if eh != n && et != n {
                    Some((et, eh))
                } else {
                    None
                };
                if let Some((tail, head)) = pair {
                    if distance[tail][head] > weight {
                        distance[tail][head] = weight;
                    }
                }
            }
            if two_path_resource >= 0 {
                self.pulse.set_ready_time(ready.clone());
                self.pulse.set_due_date(due.clone());
                self.pulse.set_distance_matrix(distance.clone());
            }
        }
        Ok(())
    }
}
