use super::*;

impl Separation {
    pub(super) fn connected_components(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let mut touching: Vec<Vec<i32>> = vec![vec![]; n];
        for column in &self.columns {
            for &customer in &column.customers {
                if self.demands[customer] > 1e-6 {
                    touching[customer].push(column.column);
                }
            }
        }
        let mut visited = Bits1024::default();
        let mut cut = Candidate::new(0, n);
        let mut remaining = Candidate::new(0, n);
        let mut column_seen = vec![false; self.num_columns];
        let mut columns: Vec<RouteColumn> = vec![];
        let add_columns =
            |vertex: usize, column_seen: &mut Vec<bool>, columns: &mut Vec<RouteColumn>| {
                for &id in &touching[vertex] {
                    let index = id as usize;
                    if column_seen[index] {
                        continue;
                    }
                    column_seen[index] = true;
                    let column = &self.columns[index];
                    columns.push(RouteColumn {
                        customers: column.customers.clone(),
                        key: column.value,
                    });
                }
            };
        let evaluate = |cut: &mut Candidate, output: &mut CandidateSet| {
            compute_violation(g, n, self.capacity, cut, 0);
            if cut.violation > self.params.violation_tolerance {
                output.insert(cut);
            }
        };
        let score_columns = |cut: &Candidate, columns: &mut Vec<RouteColumn>| {
            for column in columns.iter_mut() {
                for &vertex in &column.customers {
                    if self.demands[vertex] > 1e-6 && cut.test(vertex) {
                        column.key = column.key - self.demands[vertex] / self.capacity;
                    }
                }
            }
            stdsort::sort(columns, |a, b| a.key > b.key);
        };
        let shrink = |mut count: i32,
                      cut: &mut Candidate,
                      columns: &Vec<RouteColumn>,
                      output: &mut CandidateSet| {
            let limit = self.params.max_component_removals;
            let mut removed = 0;
            if limit <= 0 {
                return;
            }
            for column in columns {
                if count <= 2 {
                    break;
                }
                for &vertex in &column.customers {
                    if self.demands[vertex] > 1e-6
                        && cut.test(vertex)
                        && removed < limit
                        && count != 2
                    {
                        count -= 1;
                        removed += 1;
                        cut.reset(vertex);
                    }
                }
                evaluate(cut, output);
                if removed >= limit {
                    break;
                }
            }
        };
        let reconstruct = |component: &Bits1024,
                           cut: &mut Candidate,
                           seen: &mut Vec<bool>,
                           columns: &mut Vec<RouteColumn>| {
            cut.clear();
            let mut count = 0;
            for i in 0..n {
                if component.test(i) {
                    count += 1;
                    cut.set(i);
                    add_columns(i, seen, columns);
                }
            }
            count
        };
        let grow = |seed: usize,
                    keep_remaining: bool,
                    cut: &mut Candidate,
                    remaining: &mut Candidate,
                    visited: &mut Bits1024,
                    seen: &mut Vec<bool>,
                    columns: &mut Vec<RouteColumn>| {
            let mut component = Bits1024::default();
            let mut queue = vec![seed];
            component.set(seed);
            cut.set(seed);
            add_columns(seed, seen, columns);
            if keep_remaining {
                remaining.set(seed);
            }
            let mut index = 0;
            while index < queue.len() {
                let vertex = queue[index];
                visited.set(vertex);
                for &nb in &g.neighbours[vertex] {
                    if !keep_remaining && nb == n {
                        continue;
                    }
                    if visited.test(nb) {
                        continue;
                    }
                    cut.set(nb);
                    queue.push(nb);
                    visited.set(nb);
                    component.set(nb);
                    add_columns(nb, seen, columns);
                    if keep_remaining {
                        remaining.set(nb);
                    }
                }
                index += 1;
            }
            (component, queue.len() as i32)
        };
        let reset_columns = |seen: &mut Vec<bool>, columns: &mut Vec<RouteColumn>| {
            seen.iter_mut().for_each(|b| *b = false);
            columns.clear();
        };
        for &seed in &g.neighbours[n].clone() {
            if visited.test(seed) {
                continue;
            }
            let (mut component, count) = grow(
                seed,
                false,
                &mut cut,
                &mut remaining,
                &mut visited,
                &mut column_seen,
                &mut columns,
            );
            score_columns(&cut, &mut columns);
            evaluate(&mut cut, output);
            for i in 0..n {
                if self.demands[i] > 1e-6 {
                    component.flip(i);
                }
            }
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            let count = reconstruct(&component, &mut cut, &mut column_seen, &mut columns);
            evaluate(&mut cut, output);
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            cut.clear();
        }
        let mut components = 0;
        for seed in 0..n {
            if visited.test(seed) {
                continue;
            }
            components += 1;
            let (mut component, count) = grow(
                seed,
                true,
                &mut cut,
                &mut remaining,
                &mut visited,
                &mut column_seen,
                &mut columns,
            );
            score_columns(&cut, &mut columns);
            evaluate(&mut cut, output);
            component.flip_all();
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            let count = reconstruct(&component, &mut cut, &mut column_seen, &mut columns);
            evaluate(&mut cut, output);
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            cut.clear();
        }
        if components > 1 {
            let mut count = 0;
            for i in 0..n {
                if remaining.test(i) {
                    count += 1;
                    add_columns(i, &mut column_seen, &mut columns);
                }
            }
            compute_violation(g, n, self.capacity, &mut remaining, 0);
            // The remaining candidate is evaluated, but the cleared first
            // candidate is inserted.
            if remaining.violation > self.params.violation_tolerance {
                output.insert(&cut);
            }
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
        }
    }
}
