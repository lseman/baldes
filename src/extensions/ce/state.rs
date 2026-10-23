use super::Problem;

/// Bitwise resource values avoid approximate state merging. Different prefix
/// orders may share a node only when all future feasibility information agrees.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ExactState {
    load: u64,
    time: Option<u64>,
    visited: Vec<u64>,
}

impl ExactState {
    pub fn root(problem: &Problem) -> Self {
        // When there are no deadlines, time can be omitted if all elementary
        // routes are provably finite. Use the same monotone operation order as
        // replay rather than relying on an associatively summed float bound.
        let mut upper = problem
            .windows
            .iter()
            .map(|&(ready, _)| ready)
            .fold(f64::NEG_INFINITY, f64::max);
        let service = problem.service_times.iter().copied().fold(0.0, f64::max);
        let travel = problem
            .travel_times
            .iter()
            .flatten()
            .copied()
            .fold(0.0, f64::max);
        for _ in 0..=problem.demands.len() {
            upper = upper + service + travel;
        }
        let omit_time =
            upper.is_finite() && problem.windows.iter().all(|&(_, due)| due == f64::INFINITY);
        Self {
            load: 0.0f64.to_bits(),
            time: (!omit_time).then(|| problem.windows[0].0.to_bits()),
            visited: vec![
                0;
                problem.demands.len() / 64 + usize::from(problem.demands.len() % 64 != 0)
            ],
        }
    }

    fn resources(&self, problem: &Problem, from: usize, to: usize) -> Option<(f64, f64)> {
        problem.costs[from][to]?;
        let time = self.time.map_or(0.0, |time| {
            (f64::from_bits(time) + problem.service_times[from] + problem.travel_times[from][to])
                .max(problem.windows[to].0)
        });
        let mut load = f64::from_bits(self.load);
        if to <= problem.demands.len() {
            let index = to - 1;
            if self.visited[index / 64] & (1 << (index % 64)) != 0 {
                return None;
            }
            load += problem.demands[index];
        }
        if !time.is_finite()
            || time > problem.windows[to].1
            || !load.is_finite()
            || load > problem.capacity
        {
            return None;
        }
        Some((load, time))
    }

    pub fn allows(&self, problem: &Problem, from: usize, to: usize) -> bool {
        self.resources(problem, from, to).is_some()
    }

    pub fn advance(&self, problem: &Problem, from: usize, to: usize) -> Option<Self> {
        let (load, time) = self.resources(problem, from, to)?;
        let mut next = self.clone();
        next.load = load.to_bits();
        next.time = self.time.map(|_| time.to_bits());
        if to <= problem.demands.len() {
            let index = to - 1;
            next.visited[index / 64] |= 1 << (index % 64);
        }
        Some(next)
    }
}
