use super::*;

impl Separation {
    pub(super) fn greedy_construction(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let mut degree = vec![0.; n];
        for i in 0..n {
            for &nb in &g.neighbours[i] {
                degree[i] = degree[i] + g.flow[i][nb];
            }
        }
        let mut visited: HashSet<Bits1024> = HashSet::new();
        let mut cut = Candidate::new(2, n);
        for seed in 0..n {
            if g.representative[seed] != seed {
                continue;
            }
            let mut roots = Bits1024::default();
            roots.set(seed);
            cut.clear();
            for &m in &g.groups[seed] {
                cut.set(m);
            }
            let mut demand = g.demands[seed];
            let mut weight = degree[seed];
            let totals = &degree;
            let mut joined = g.flow[seed].clone();
            let update = |e: &mut GreedyExtension,
                          demand: f64,
                          weight: f64,
                          joined: &[f64],
                          roots: &mut Bits1024,
                          visited: &HashSet<Bits1024>| {
                let at = e.id;
                let k = ((demand + g.demands[at]) / self.capacity).ceil() as i32;
                e.violation = (k * 2) as f64 - ((weight - (joined[at] + joined[at])) + totals[at]);
                e.delta = joined[at];
                roots.set(at);
                e.explored = visited.contains(roots);
                roots.reset(at);
            };
            let mut extensions: Vec<GreedyExtension> = vec![];
            for &nb in &g.neighbours[seed] {
                if nb == n || cut.test(nb) {
                    continue;
                }
                let mut e = GreedyExtension {
                    id: nb,
                    ..Default::default()
                };
                update(&mut e, demand, weight, &joined, &mut roots, &visited);
                extensions.push(e);
            }
            while !extensions.is_empty() {
                let by_delta = self.params.greedy_selection != 0;
                stdsort::sort(&mut extensions, |a, b| greedy_less(a, b, by_delta));
                let selected = extensions.pop().unwrap();
                if selected.explored {
                    break;
                }
                let at = selected.id;
                roots.set(at);
                visited.insert(roots);
                demand = demand + g.demands[at];
                for &m in &g.groups[at] {
                    cut.set(m);
                }
                if selected.violation > self.params.violation_tolerance {
                    cut.k = (demand / self.capacity).ceil() as i32;
                    cut.violation = selected.violation;
                    output.insert(&cut);
                }
                weight = (weight - (joined[at] + joined[at])) + totals[at];
                joined[at] = 0.;
                for &nb in &g.neighbours[at] {
                    if nb == n || cut.test(nb) {
                        continue;
                    }
                    if joined[nb] == 0.0 {
                        extensions.push(GreedyExtension {
                            id: nb,
                            ..Default::default()
                        });
                    }
                    joined[nb] = joined[nb] + g.flow[at][nb];
                }
                for e in &mut extensions {
                    update(e, demand, weight, &joined, &mut roots, &visited);
                }
            }
        }
    }

    pub(super) fn global_pool_greedy(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let score = |total: f64, crossing: f64| {
            let k = (total / self.capacity).ceil();
            (k + k) - crossing
        };
        for stored in &self.global_subsets.iter() {
            let mut subset = Bits1024::default();
            for i in 0..n {
                if stored.test(i) {
                    subset.set(g.representative[i]);
                }
            }
            let mut inside = vec![0.; n];
            let mut outside = vec![0.; n];
            let mut members: Vec<(f64, usize)> = vec![];
            let mut demand = 0.;
            let mut weight = 0.;
            for i in 0..n {
                for &nb in &g.neighbours[i] {
                    if nb != n && g.representative[nb] != nb {
                        continue;
                    }
                    let flow = g.flow[i][nb];
                    if subset.test(nb) {
                        inside[i] = inside[i] + flow;
                    } else {
                        outside[i] = outside[i] + flow;
                    }
                }
                if !subset.test(i) {
                    continue;
                }
                members.push((g.demands[i], i));
                demand = demand + g.demands[i];
                weight = weight + outside[i];
            }
            members.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut violation = score(demand, weight);
            let update_flows =
                |vertex: usize, added: bool, inside: &mut Vec<f64>, outside: &mut Vec<f64>| {
                    for &nb in &g.neighbours[vertex] {
                        if nb == n {
                            continue;
                        }
                        let flow = g.flow[vertex][nb];
                        if added {
                            inside[nb] = inside[nb] + flow;
                            outside[nb] = outside[nb] - flow;
                        } else {
                            inside[nb] = inside[nb] - flow;
                            outside[nb] = outside[nb] + flow;
                        }
                    }
                };
            for &(_, at) in &members {
                let next_demand = demand - g.demands[at];
                let next_weight = (weight - outside[at]) + inside[at];
                let next_violation = score(next_demand, next_weight);
                if !(next_violation > violation + 1e-6) {
                    continue;
                }
                subset.reset(at);
                update_flows(at, false, &mut inside, &mut outside);
                demand = next_demand;
                weight = next_weight;
                violation = next_violation;
            }
            loop {
                let mut gain = 0.;
                let (mut removed, mut added) = (-1i64, -1i64);
                for i in 0..n {
                    if g.representative[i] != i {
                        continue;
                    }
                    if subset.test(i) {
                        let next_demand = demand - g.demands[i];
                        let next_weight = inside[i] + (weight - outside[i]);
                        let mut change = score(next_demand, next_weight) - violation;
                        if gain + 1e-6 < change {
                            removed = i as i64;
                            added = -1;
                            gain = change;
                        }
                        for j in 0..n {
                            if g.representative[j] != j || subset.test(j) {
                                continue;
                            }
                            let edge = g.flow[i][j];
                            let crossing = edge + edge + ((outside[j] + next_weight) - inside[j]);
                            change = score(g.demands[j] + next_demand, crossing) - violation;
                            if gain + 1e-6 < change {
                                removed = i as i64;
                                added = j as i64;
                                gain = change;
                            }
                        }
                    } else {
                        let change =
                            score(g.demands[i] + demand, (outside[i] + weight) - inside[i])
                                - violation;
                        if gain + 1e-6 < change {
                            removed = -1;
                            added = i as i64;
                            gain = change;
                        }
                    }
                }
                if !(gain > 1e-6) {
                    break;
                }
                if added != -1 {
                    let at = added as usize;
                    subset.set(at);
                    demand = demand + g.demands[at];
                    weight = (weight + outside[at]) - inside[at];
                    update_flows(at, true, &mut inside, &mut outside);
                }
                if removed != -1 {
                    let at = removed as usize;
                    subset.reset(at);
                    weight = inside[at] + (weight - outside[at]);
                    if added != -1 {
                        let edge = g.flow[added as usize][at];
                        weight = edge + edge + weight;
                    }
                    demand = demand - g.demands[at];
                    update_flows(at, false, &mut inside, &mut outside);
                }
                violation = score(demand, weight);
            }
            if !(violation > 1e-6) {
                continue;
            }
            let mut cut = Candidate::new(3, n);
            for i in 0..n {
                if subset.test(i) {
                    for &m in &g.groups[i] {
                        cut.set(m);
                    }
                }
            }
            cut.k = (demand / self.capacity).ceil() as i32;
            cut.violation = violation;
            output.insert(&cut);
        }
    }
}
