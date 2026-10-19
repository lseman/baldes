use super::*;

impl Separation {
    pub(super) fn min_cut_heuristic(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let nodes: Vec<usize> = (0..g.neighbours.len() - 1)
            .filter(|&i| !g.neighbours[i].is_empty())
            .collect();
        let count = nodes.len();
        let (source, sink) = (count, count + 1);
        let width = count + 2;
        let mut capacity = vec![vec![0.; width]; width];
        let mut source_capacity = vec![];
        let mut sink_capacity = vec![];
        for (i, &vertex) in nodes.iter().enumerate() {
            let depot = g.flow[n][vertex];
            let fraction = g.demands[vertex] / self.capacity;
            let mut from = (depot - (fraction + fraction)).max(1e-6);
            let mut to = ((fraction + fraction) - depot).max(1e-6);
            if from <= 1e-6 {
                from = 0.;
            }
            if to <= 1e-6 {
                to = 0.;
            }
            source_capacity.push(from);
            sink_capacity.push(to);
            capacity[source][i] = from;
            capacity[i][sink] = to;
        }
        for i in 0..count {
            for j in i + 1..count {
                capacity[i][j] = g.flow[nodes[i]][nodes[j]];
                capacity[j][i] = g.flow[nodes[j]][nodes[i]];
            }
        }
        let mut graph = MinCutDigraph::new_with_terminals(capacity, source as i32, sink as i32);
        graph.find_min_cut(source as i32, sink as i32);
        let mut cut = Candidate::new(4, n);
        let mut candidate = FracCandidate {
            field_04: count as i32,
            ..Default::default()
        };
        let include = |node: i32, candidate: &mut FracCandidate, cut: &mut Candidate| {
            candidate.field_08 += 1;
            let vertex = nodes[node as usize];
            candidate.field_10 = g.demands[vertex] + candidate.field_10;
            candidate.nodes.push(node);
            candidate.subset.set(node as usize);
            for &m in &g.groups[vertex] {
                cut.set(m);
            }
        };
        let include_sink_side =
            |graph: &MinCutDigraph, candidate: &mut FracCandidate, cut: &mut Candidate| {
                for i in 0..graph.sink_side.len().saturating_sub(1) {
                    include(graph.sink_side[i], candidate, cut);
                }
            };
        include_sink_side(&graph, &mut candidate, &mut cut);
        compute_violation(g, n, self.capacity, &mut cut, 0);
        if cut.violation > self.params.violation_tolerance {
            output.insert(&cut);
        }
        cut.clear();
        let mut previous: Vec<FracCandidate> = vec![];
        let mut allowed = vec![true; count];
        let mut singleton = vec![false; count];
        let mut fixed_nodes: Vec<i32> = vec![];
        let mut fixed = Bits1024::default();
        let mut second_sink: i32 = -1;
        for _pass in 0..3 {
            for seed in 0..count {
                if !allowed[seed] {
                    continue;
                }
                graph.set_arc_value(seed as i32, sink as i32, 99.0);
                candidate.field_00 = seed as i32;
                fixed_to_source(
                    &candidate,
                    seed as i32,
                    &mut fixed_nodes,
                    &mut fixed,
                    &previous,
                );
                if singleton[seed] {
                    second_sink_node(
                        &candidate,
                        &mut second_sink,
                        seed as i32,
                        &nodes,
                        &fixed,
                        &g.flow,
                    );
                    if second_sink < 0 {
                        allowed[seed] = false;
                    } else {
                        graph.set_arc_value(second_sink, sink as i32, 99.0);
                    }
                }
                for &node in &fixed_nodes {
                    graph.set_arc_value(source as i32, node, 99.0);
                }
                graph.find_min_cut(source as i32, sink as i32);
                include_sink_side(&graph, &mut candidate, &mut cut);
                loop {
                    let node = expand_subset(
                        &mut candidate,
                        self.capacity,
                        &nodes,
                        &g.demands,
                        &fixed,
                        &g.flow,
                    );
                    if node < 0 {
                        break;
                    }
                    for &m in &g.groups[nodes[node as usize]] {
                        cut.set(m);
                    }
                }
                if candidate.field_08 == 1 {
                    singleton[seed] = true;
                }
                compute_violation(g, n, self.capacity, &mut cut, 0);
                if cut.violation > self.params.violation_tolerance {
                    output.insert(&cut);
                }
                previous.push(candidate.clone());
                candidate.field_00 = -1;
                candidate.field_08 = 0;
                candidate.field_10 = 0.;
                candidate.nodes.clear();
                candidate.subset.clear();
                cut.clear();
                graph.set_arc_value(seed as i32, sink as i32, sink_capacity[seed]);
                if second_sink >= 0 {
                    graph.set_arc_value(
                        second_sink,
                        sink as i32,
                        sink_capacity[second_sink as usize],
                    );
                    second_sink = -1;
                }
                for &node in &fixed_nodes {
                    graph.set_arc_value(source as i32, node, source_capacity[node as usize]);
                }
                fixed_nodes.clear();
                fixed.clear();
            }
        }
    }
}
