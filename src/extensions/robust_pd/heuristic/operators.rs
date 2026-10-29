//! Optional pair-preserving ILS operations. Final routes always pass replay.
use super::*;

impl Search<'_> {
    /// Group requests by pickup/delivery proximity and time-window proximity.
    pub(super) fn related_order(&self, requests: &mut [usize]) {
        let Some(&anchor) = requests.first() else {
            return;
        };
        let score = |r: usize| {
            let mut value = 0.0;
            for offset in [1, self.n + 1] {
                let (a, b) = (anchor + offset, r + offset);
                let (aw, bw) = (self.window(a), self.window(b));
                value += self.travel(a, b) + (aw.0 - bw.0).abs() + (aw.1 - bw.1).abs();
            }
            value
        };
        requests.sort_by(|&a, &b| score(a).total_cmp(&score(b)));
    }

    /// Rank by marginal travel saving; removal itself still verifies feasibility.
    pub(super) fn expensive_order(&self, routes: &[Route], requests: &mut [usize]) {
        let mut savings = vec![f64::NEG_INFINITY; self.n];
        for route in routes {
            for &p in route.nodes.iter().filter(|&&v| (1..=self.n).contains(&v)) {
                let nodes = self.without_pair(route, p - 1);
                let travel: f64 = nodes
                    .windows(2)
                    .map(|arc| self.travel(arc[0], arc[1]))
                    .sum();
                savings[p - 1] = route.travel - travel;
            }
        }
        requests.sort_by(|&a, &b| savings[b].total_cmp(&savings[a]));
    }

    /// Regret-2 across distinct receiving routes, including a singleton.
    pub(super) fn regret_recreate(
        &mut self,
        mut routes: Vec<Route>,
        requests: &[usize],
    ) -> Option<Vec<Route>> {
        let mut pending = requests.to_vec();
        while !pending.is_empty() {
            let mut selected = 0;
            let mut largest = f64::NEG_INFINITY;
            for (k, &r) in pending.iter().enumerate() {
                let mut alternatives = Vec::new();
                for (index, route) in routes.iter().enumerate() {
                    let mut placements = Vec::new();
                    self.placements(route, index, r, &mut placements);
                    let best = placements
                        .iter()
                        .map(|p| p.delta)
                        .fold(f64::INFINITY, f64::min);
                    if best.is_finite() {
                        alternatives.push(best);
                    }
                }
                if let Some(travel) = self.alone(r) {
                    alternatives.push(travel + self.problem.parameters.vehicle_cost);
                }
                alternatives.sort_by(f64::total_cmp);
                let regret = match alternatives.as_slice() {
                    [] => return None,
                    [_] => f64::INFINITY,
                    [first, second, ..] => second - first,
                };
                if regret > largest {
                    selected = k;
                    largest = regret;
                }
                // Finish construction safely even when the search budget expires.
                if Instant::now() >= self.deadline {
                    break;
                }
            }
            self.insert(&mut routes, pending.remove(selected), true)?;
        }
        Some(routes)
    }

    /// Put requests at a tight load cut or an attaining time segment first.
    /// The incoming shuffled order randomizes ties and remaining removals.
    pub(super) fn critical_order(&mut self, routes: &[Route], requests: &mut [usize]) {
        if routes.is_empty() {
            return;
        }
        let load_directed = self.rng.next_f64() < 0.5;
        let mut critical = vec![false; self.n];
        if load_directed {
            let mut best = f64::INFINITY;
            for route in routes {
                let mut open = vec![false; self.n];
                for (h, &v) in route.nodes.iter().enumerate() {
                    if (1..=self.n).contains(&v) {
                        open[v - 1] = true;
                    } else if (self.n + 1..=2 * self.n).contains(&v) {
                        open[v - self.n - 1] = false;
                    }
                    let slack =
                        self.problem.instance.capacity - route.load[h][0] - route.load[h][1];
                    if open.iter().any(|&x| x) && slack < best {
                        best = slack;
                        critical.clone_from(&open);
                    }
                }
            }
        } else {
            let mut best = f64::INFINITY;
            let mut selected = None;
            for (r, route) in routes.iter().enumerate() {
                for h in 1..route.nodes.len() {
                    let slack = self.window(route.nodes[h]).1 - route.forward[h][self.budget];
                    if slack < best {
                        best = slack;
                        selected = Some((r, h));
                    }
                }
            }
            if let Some((r, mut h)) = selected {
                let route = &routes[r];
                let mut g = self.budget;
                while h > 0 {
                    let (u, v) = (route.nodes[h - 1], route.nodes[h]);
                    let duration = self.problem.instance.nodes[u].service + self.travel(u, v);
                    let nominal = route.forward[h - 1][g] + duration;
                    let delayed = if g > 0 {
                        route.forward[h - 1][g - 1]
                            + duration
                            + (self.problem.parameters.time_deviation * self.travel(u, v)).ceil()
                    } else {
                        f64::NEG_INFINITY
                    };
                    if self.window(v).0 >= nominal.max(delayed) {
                        break; // Waiting, rather than an earlier arc, attains this start.
                    }
                    for node in [u, v] {
                        if (1..=2 * self.n).contains(&node) {
                            critical[(node - 1) % self.n] = true;
                        }
                    }
                    if delayed > nominal {
                        g -= 1;
                    }
                    h -= 1;
                }
            }
        }
        requests.sort_by_key(|&r| !critical[r]);
    }

    fn without_pair(&self, route: &Route, request: usize) -> Vec<usize> {
        route
            .nodes
            .iter()
            .copied()
            .filter(|&v| v != request + 1 && v != request + 1 + self.n)
            .collect()
    }

    /// Search a bounded shuffled set of cross-route exchanges. Intermediate
    /// shortened routes are not required to be feasible; both final routes are.
    pub(super) fn exchange_pairs(&mut self, routes: &mut [Route]) -> bool {
        let mut pairs = Vec::new();
        for a in 0..routes.len() {
            for b in a + 1..routes.len() {
                for &p in routes[a]
                    .nodes
                    .iter()
                    .filter(|&&v| (1..=self.n).contains(&v))
                {
                    for &q in routes[b]
                        .nodes
                        .iter()
                        .filter(|&&v| (1..=self.n).contains(&v))
                    {
                        pairs.push((a, b, p - 1, q - 1));
                    }
                }
            }
        }
        self.rng.shuffle(&mut pairs);
        for (a, b, i, j) in pairs.into_iter().take(64) {
            if Instant::now() >= self.deadline {
                break;
            }
            let left = self.best_pair_replacement(&routes[a], i, j);
            let right = self.best_pair_replacement(&routes[b], j, i);
            if let (Some(left), Some(right)) = (left, right) {
                if left.travel + right.travel < routes[a].travel + routes[b].travel - 1e-7 {
                    routes[a] = left;
                    routes[b] = right;
                    return true;
                }
            }
        }
        false
    }

    fn best_pair_replacement(&self, route: &Route, remove: usize, add: usize) -> Option<Route> {
        let nodes = self.without_pair(route, remove);
        let mut best: Option<Route> = None;
        // Full replay admits final exchanges even if the intermediate route fails.
        for i in 0..nodes.len() - 1 {
            for j in i..nodes.len() - 1 {
                if Instant::now() >= self.deadline {
                    return best;
                }
                let candidate = self.insert_nodes(&nodes, add, i, j);
                if let Some(candidate) = self.build(candidate) {
                    if best.as_ref().map_or(true, |b| candidate.travel < b.travel) {
                        best = Some(candidate);
                    }
                }
            }
        }
        best
    }

    fn closed_cuts(&self, route: &Route) -> Vec<usize> {
        let mut count = 0usize;
        let mut cuts = Vec::new();
        for h in 1..route.nodes.len() - 2 {
            if route.nodes[h] <= self.n {
                count += 1;
            } else {
                count -= 1;
            }
            if count == 0 {
                cuts.push(h);
            }
        }
        cuts
    }

    /// Exchange nonempty tails at cuts with no requests on board.
    pub(super) fn exchange_tails(&mut self, routes: &mut [Route]) -> bool {
        let mut candidates = Vec::new();
        for a in 0..routes.len() {
            for b in a + 1..routes.len() {
                for i in self.closed_cuts(&routes[a]) {
                    for j in self.closed_cuts(&routes[b]) {
                        candidates.push((a, b, i, j));
                    }
                }
            }
        }
        self.rng.shuffle(&mut candidates);
        for (a, b, i, j) in candidates.into_iter().take(128) {
            if Instant::now() >= self.deadline {
                break;
            }
            let left = routes[a].nodes[..=i]
                .iter()
                .chain(&routes[b].nodes[j + 1..])
                .copied()
                .collect();
            let right = routes[b].nodes[..=j]
                .iter()
                .chain(&routes[a].nodes[i + 1..])
                .copied()
                .collect();
            if let (Some(left), Some(right)) = (self.build(left), self.build(right)) {
                if left.travel + right.travel < routes[a].travel + routes[b].travel - 1e-7 {
                    routes[a] = left;
                    routes[b] = right;
                    return true;
                }
            }
        }
        false
    }

    /// Depth-one ejection: move i into j's route while ejecting j, then repair j
    /// in any route. Bound the proposals and restore the whole transaction on failure.
    pub(super) fn ejection_chain(&mut self, routes: &mut Vec<Route>) -> bool {
        let before = self.cost(routes);
        let mut requests: Vec<_> = (0..self.n).collect();
        self.rng.shuffle(&mut requests);
        for &i in requests.iter().take(8) {
            for &j in requests.iter().take(8) {
                if i == j {
                    continue;
                }
                if Instant::now() >= self.deadline {
                    return false;
                }
                let source = routes
                    .iter()
                    .position(|r| r.nodes.contains(&(i + 1)))
                    .unwrap();
                let target = routes
                    .iter()
                    .position(|r| r.nodes.contains(&(j + 1)))
                    .unwrap();
                if source == target {
                    continue;
                }
                let Some(replacement) = self.best_pair_replacement(&routes[target], j, i) else {
                    continue;
                };
                let mut candidate = routes.clone();
                // Remove i first so intermediate coverage remains unambiguous.
                if self.remove(&mut candidate, i).is_none() {
                    continue;
                }
                let target = candidate
                    .iter()
                    .position(|r| r.nodes.contains(&(j + 1)))
                    .unwrap();
                candidate[target] = replacement;
                if self.insert(&mut candidate, j, true).is_some()
                    && self.cost(&candidate) < before - 1e-7
                {
                    *routes = candidate;
                    return true;
                }
            }
        }
        false
    }
}
