use super::*;

impl Separation {
    /// createAndPrepareSeparation for the complete customer graph: source 0,
    /// customers 1..n, sink n+1, one master variable per directed arc
    /// (id tail * (n+2) + head). `demands` excludes the depot.
    pub(super) fn prepare(
        demands: &[f64],
        capacity: f64,
        params: Params,
        buckets: i32,
        windows: Option<&TimeWindows>,
    ) -> Result<Self> {
        if demands.iter().any(|d| !(*d >= -1e-6)) {
            return Err(invalid("demands should be non-negative"));
        }
        if !(capacity >= 1e-6) {
            return Err(invalid("capacity should be positive"));
        }
        let n = demands.len();
        let width = n + 2;
        let sink = n + 1;
        let mut representative = vec![0; n];
        let mut num_positive_demand = 0;
        for i in 0..n {
            if demands[i] <= 0. {
                representative[i] = n;
            } else {
                num_positive_demand += 1;
            }
        }
        let mut all_demands = demands.to_vec();
        all_demands.push(0.);
        let element = |v: usize| -> usize {
            if v >= 1 && v <= n && all_demands[v - 1] >= 1e-6 {
                v - 1
            } else {
                n
            }
        };
        let mut pulse = PulseDigraph::new(buckets);
        if let Some(w) = windows {
            let mut ready = vec![1e12; n + 2];
            let mut due = vec![-1e12; n + 2];
            for v in 0..width {
                let e = element(v);
                let slot = if e == n {
                    if v == sink {
                        n + 1
                    } else if v == 0 {
                        n
                    } else {
                        continue;
                    }
                } else {
                    e
                };
                let input = if v == sink { 0 } else { v };
                if w.ready[input] < ready[slot] {
                    ready[slot] = w.ready[input];
                }
                if due[slot] < w.due[input] {
                    due[slot] = w.due[input];
                }
            }
            let mut distance = vec![vec![1e12; n + 2]; n + 2];
            for tail in 0..sink {
                for head in 1..=sink {
                    if tail == head {
                        continue;
                    }
                    let (te, he) = (element(tail), element(head));
                    let mut length =
                        w.service[tail] + w.travel[tail][if head == sink { 0 } else { head }];
                    if length < 0. {
                        length += due[n + 1];
                    }
                    if !(te == n && he == n) {
                        let from = te;
                        let to = if he == n { n + 1 } else { he };
                        if length < distance[from][to] {
                            distance[from][to] = length;
                        }
                    }
                }
            }
            pulse.set_ready_time(ready);
            pulse.set_due_date(due);
            pulse.set_distance_matrix(distance);
        }
        let mut arcs_between = vec![vec![vec![]; n + 1]; n + 1];
        for tail in 0..sink {
            for head in 1..=sink {
                if tail == head {
                    continue;
                }
                let (te, he) = (element(tail), element(head));
                let (a, b) = (te.min(he), te.max(he));
                if a == n {
                    continue;
                }
                let id = (tail * width + head) as i32;
                arcs_between[a][b].push(id);
                arcs_between[b][a].push(id);
            }
        }
        Ok(Self {
            statistics: CapacityStatistics::default(),
            params,
            n,
            num_positive_demand,
            demands: all_demands,
            capacity,
            two_path: windows.is_some(),
            arcs_between,
            graph: Graph {
                demands: vec![],
                flow: vec![vec![0.; n + 1]; n + 1],
                neighbours: vec![vec![]; n + 1],
                groups: vec![vec![]; n + 1],
                representative,
            },
            global_subsets: UnorderedSet::new(),
            two_path_subsets: UnorderedSet::new(),
            pulse,
            columns: vec![],
            num_columns: 0,
        })
    }
}
