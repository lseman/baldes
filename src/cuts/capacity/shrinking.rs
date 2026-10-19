use super::*;

impl Separation {
    pub(super) fn shrinking(
        &self,
        threshold: f64,
        candidates: &mut CandidateSet,
        g: &mut Graph,
        live: bool,
    ) {
        let n = self.n;
        let mut edges: Vec<(usize, usize)> = vec![];
        let mut cut = Candidate::new(1, n);
        {
            let reps = if live {
                &g.representative
            } else {
                &self.graph.representative
            };
            for i in 0..n {
                if reps[i] != i {
                    continue;
                }
                for &j in &g.neighbours[i] {
                    if j < i && reps[j] == j && threshold - 1e-6 < g.flow[j][i] {
                        edges.push((j, i));
                    }
                }
            }
        }
        while !edges.is_empty() {
            let mut selected = 0;
            if self.params.contraction_selection == 1 {
                let mut best = 0.;
                for (idx, &(x, y)) in edges.iter().enumerate() {
                    let weight = g.flow[x][y];
                    if best < weight {
                        selected = idx;
                        best = weight;
                    }
                }
            }
            let (a, b) = edges.remove(selected);
            for &m in &g.groups[a] {
                cut.set(m);
            }
            for &m in &g.groups[b] {
                cut.set(m);
            }
            if live {
                compute_violation(g, n, self.capacity, &mut cut, 0);
            } else {
                compute_violation(&self.graph, n, self.capacity, &mut cut, 0);
            }
            if cut.violation > 1e-6 {
                candidates.insert(&cut);
            }
            cut.clear();
            let neighbours = g.neighbours[b].clone();
            for &nb in &neighbours {
                if nb != a {
                    let previous = g.flow[a][nb];
                    if previous == 0.0 {
                        g.neighbours[a].push(nb);
                        g.neighbours[nb].push(a);
                    }
                    g.flow[a][nb] = g.flow[b][nb] + previous;
                    g.flow[nb][a] = g.flow[nb][a] + g.flow[b][nb];
                    let weight = g.flow[a][nb];
                    if threshold - 1e-6 < weight && previous <= threshold - 1e-6 {
                        edges.push((a, nb));
                    }
                }
                g.flow[b][nb] = 0.;
                g.flow[nb][b] = 0.;
                if let Some(pos) = g.neighbours[nb].iter().position(|&x| x == b) {
                    g.neighbours[nb].remove(pos);
                }
            }
            g.neighbours[b].clear();
            let members = std::mem::take(&mut g.groups[b]);
            for m in members {
                g.groups[a].push(m);
                g.representative[m] = a;
            }
            g.demands[a] = g.demands[a] + g.demands[b];
            g.demands[b] = 0.;
            let max_group = self.params.max_contracted_group_size as usize;
            edges.retain(|&(x, y)| {
                let reps = if live {
                    &g.representative
                } else {
                    &self.graph.representative
                };
                !(x == n
                    || y == n
                    || reps[x] != x
                    || reps[y] != y
                    || g.groups[x].len() + g.groups[y].len() > max_group)
            });
        }
    }
}
