//! Predecessor-queue Split for single-window routes.

use std::collections::VecDeque;

use crate::extensions::hgs::Params;

#[derive(Clone, Copy, Default)]
struct Links {
    previous: Option<usize>,
    next: Option<usize>,
}

/// Candidates stay in tour order. The two cursors partition candidates by
/// whether capacity/time-window penalties have become active.
struct Predecessors {
    links: Vec<Links>,
    head: Option<usize>,
    tail: Option<usize>,
    feasible: Option<usize>,
    unwarped: Option<usize>,
}

impl Predecessors {
    fn new(n: usize) -> Self {
        Self {
            links: vec![Links::default(); n],
            head: None,
            tail: None,
            feasible: None,
            unwarped: None,
        }
    }

    fn clear(&mut self) {
        self.links.fill(Links::default());
        self.head = None;
        self.tail = None;
        self.feasible = None;
        self.unwarped = None;
    }

    fn push(&mut self, i: usize) {
        self.links[i].previous = self.tail;
        if let Some(tail) = self.tail {
            self.links[tail].next = Some(i);
        } else {
            self.head = Some(i);
        }
        self.tail = Some(i);
        self.feasible = self.feasible.or(Some(i));
        self.unwarped = self.unwarped.or(self.feasible);
    }

    fn remove(&mut self, i: usize) {
        let Links { previous, next } = self.links[i];
        if let Some(previous) = previous {
            self.links[previous].next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.links[next].previous = previous;
        } else {
            self.tail = previous;
        }
        if self.feasible == Some(i) {
            self.feasible = next;
        }
        if self.unwarped == Some(i) {
            self.unwarped = next;
        }
        self.links[i] = Links::default();
    }

    fn second(&self) -> Option<usize> {
        self.head.and_then(|head| self.links[head].next)
    }
}

/// Reused O(n) scratch for both unlimited- and limited-fleet programs. Each
/// candidate is inserted/removed once per layer; first-warp and feasibility
/// cursors advance monotonically, so a layer takes O(n) amortized work.
pub(super) struct TimeWindowSplit {
    outgoing: Vec<f64>,
    incoming: Vec<f64>,
    earliest: Vec<f64>,
    latest: Vec<f64>,
    distance: Vec<f64>,
    duration: Vec<f64>,
    demand: Vec<i64>,
    warp: Vec<f64>,
    base: Vec<f64>,
    first_warp: Vec<f64>,
    first_warp_end: Vec<usize>,
    queue: Predecessors,
    waiting: VecDeque<usize>,
    #[cfg(test)]
    pub(super) operations: usize,
}

impl TimeWindowSplit {
    pub(super) fn new(n: usize) -> Self {
        Self {
            outgoing: vec![0.; n],
            incoming: vec![0.; n],
            earliest: vec![0.; n],
            latest: vec![0.; n],
            distance: vec![0.; n],
            duration: vec![0.; n],
            demand: vec![0; n + 1],
            warp: vec![0.; n + 1],
            base: vec![0.; n],
            first_warp: vec![0.; n],
            first_warp_end: vec![0; n],
            queue: Predecessors::new(n),
            waiting: VecDeque::with_capacity(n),
            #[cfg(test)]
            operations: 0,
        }
    }

    /// Queue dominance requires nonnegative demands/service/travel, individually
    /// feasible customers, and depot triangle inequalities along the tour.
    /// Other schedules retain the general evaluator-based Split.
    pub(super) fn prepare(&mut self, params: &Params, tour: &[usize]) -> bool {
        if tour.is_empty()
            || params.penalty_wait_time != 0.
            || !params.penalty_capacity.get().is_finite()
            || !params.penalty_time_warp.get().is_finite()
            || params.penalty_capacity.get() < 0.
            || params.penalty_time_warp.get() < 0.
        {
            return false;
        }
        self.demand[0] = 0;
        self.warp[0] = 0.;
        let mut arrival = 0.;
        for (i, &v) in tour.iter().enumerate() {
            let client = &params.cli[v];
            let outgoing = params.time_cost.get(0, v);
            let incoming = params.time_cost.get(v, 0);
            // Tightening customer latest times by the direct return bound
            // accounts for depot-return warp. Along metric depot legs, any
            // warp introduced early this way is saved from a later violation.
            let latest = client
                .latest_arrival
                .min(params.cli[0].latest_arrival - client.service_duration - incoming);
            if client.n_tw != 0
                || client.release_time != 0.
                || client.demand < 0
                || client.demand > params.vehicle_capacity
                || client.service_duration < 0.
                || !client.service_duration.is_finite()
                || !client.earliest_arrival.is_finite()
                || !latest.is_finite()
                || !outgoing.is_finite()
                || !incoming.is_finite()
                || outgoing < 0.
                || incoming < 0.
                || client.earliest_arrival.max(outgoing) > latest
            {
                return false;
            }
            self.outgoing[i] = outgoing;
            self.incoming[i] = incoming;
            self.earliest[i] = client.earliest_arrival;
            self.latest[i] = latest;
            self.demand[i + 1] = self.demand[i] + i64::from(client.demand);
            if i == 0 {
                self.distance[i] = 0.;
                self.duration[i] = 0.;
                arrival = outgoing.max(self.earliest[i]);
            } else {
                let previous = tour[i - 1];
                let travel = params.time_cost.get(previous, v);
                let service = params.cli[previous].service_duration;
                let tolerance = 32.
                    * f64::EPSILON
                    * (self.outgoing[i - 1] + outgoing + self.incoming[i - 1] + incoming + travel)
                        .max(1.);
                if !travel.is_finite()
                    || travel < 0.
                    || self.outgoing[i - 1] + travel + tolerance < outgoing
                    || travel + incoming + tolerance < self.incoming[i - 1]
                {
                    return false;
                }
                self.distance[i] = self.distance[i - 1] + travel;
                self.duration[i] = self.duration[i - 1] + service + travel;
                arrival = (arrival + service + travel).max(self.earliest[i]);
            }
            self.warp[i + 1] = self.warp[i] + (arrival - latest).max(0.);
            arrival = arrival.min(latest);
        }
        true
    }

    #[inline]
    fn capacity_penalty(&self, i: usize, end: usize, capacity: i64, alpha: f64) -> f64 {
        (self.demand[end] - self.demand[i] - capacity).max(0) as f64 * alpha
    }

    #[inline]
    fn time_penalty(&self, i: usize, end: usize, beta: f64) -> f64 {
        if self.first_warp[i] > 0. {
            beta * (self.first_warp[i] + self.warp[end] - self.warp[self.first_warp_end[i]])
        } else {
            0.
        }
    }

    #[inline]
    fn score(&self, i: usize, end: usize, capacity: i64, alpha: f64, beta: f64) -> f64 {
        self.base[i]
            + self.capacity_penalty(i, end, capacity, alpha)
            + self.time_penalty(i, end, beta)
    }

    pub(super) fn solve(
        &mut self,
        params: &Params,
        fleet: Option<usize>,
        potential: &mut [Vec<f64>],
        pred: &mut [Vec<usize>],
    ) {
        #[cfg(test)]
        {
            self.operations = 0;
        }
        if let Some(fleet) = fleet {
            potential[0].fill(f64::INFINITY);
            potential[0][0] = 0.;
            for layer in 1..=fleet {
                let (previous, current) = potential.split_at_mut(layer);
                self.layer(
                    params,
                    Some(&previous[layer - 1]),
                    &mut current[0],
                    &mut pred[layer],
                    layer,
                );
            }
        } else {
            self.layer(params, None, &mut potential[0], &mut pred[0], 1);
        }
    }

    fn layer(
        &mut self,
        params: &Params,
        previous: Option<&[f64]>,
        cost: &mut [f64],
        pred: &mut [usize],
        first: usize,
    ) {
        let n = self.outgoing.len();
        let capacity = i64::from(params.vehicle_capacity);
        let alpha = params.penalty_capacity.get();
        let beta = params.penalty_time_warp.get();
        cost.fill(f64::INFINITY);
        if previous.is_none() {
            cost[0] = 0.;
        }
        self.queue.clear();
        self.waiting.clear();
        self.first_warp.fill(0.);

        for end in first..=n {
            #[cfg(test)]
            {
                self.operations += 1;
            }
            let last = end - 1;
            let prefix = previous.map_or(cost[last], |row| row[last]);
            if prefix.is_finite() {
                self.base[last] = prefix + self.outgoing[last] - self.distance[last];
                while let Some(back) = self.queue.tail {
                    if self.base[back] <= self.base[last] {
                        break;
                    }
                    self.queue.remove(back);
                    #[cfg(test)]
                    {
                        self.operations += 1;
                    }
                }
                self.queue.push(last);
            }
            if self.queue.head.is_none() {
                continue;
            }

            while let Some(&back) = self.waiting.back() {
                if self.earliest[back] - self.duration[back]
                    > self.earliest[last] - self.duration[last]
                {
                    break;
                }
                self.waiting.pop_back();
            }
            self.waiting.push_back(last);
            while let Some(i) = self.queue.unwarped {
                #[cfg(test)]
                {
                    self.operations += 1;
                }
                while self.waiting.front().is_some_and(|&j| j < i) {
                    self.waiting.pop_front();
                }
                let waiting = *self
                    .waiting
                    .front()
                    .expect("current customer remains in waiting deque");
                let arrival = (self.outgoing[i] + self.duration[last] - self.duration[i])
                    .max(self.earliest[waiting] + self.duration[last] - self.duration[waiting]);
                let warp = (arrival - self.latest[last]).max(0.);
                self.first_warp[i] = warp;
                if warp == 0. {
                    break;
                }
                self.first_warp_end[i] = end;
                self.queue.unwarped = self.queue.links[i].next;
            }

            while let Some(i) = self.queue.feasible {
                #[cfg(test)]
                {
                    self.operations += 1;
                }
                if self.capacity_penalty(i, end, capacity, alpha) == 0.
                    && self.time_penalty(i, end, beta) == 0.
                {
                    break;
                }
                while let Some(p) = self.queue.links[i].previous {
                    if self.score(p, end, capacity, alpha, beta)
                        <= self.score(i, end, capacity, alpha, beta)
                    {
                        break;
                    }
                    self.queue.remove(p);
                    #[cfg(test)]
                    {
                        self.operations += 1;
                    }
                }
                self.queue.feasible = self.queue.links[i].next;
            }

            // The penalized/feasible border must be rechecked at EVERY end,
            // even when the feasibility cursor did not advance.
            if let Some(next) = self.queue.feasible {
                while let Some(p) = self.queue.links[next].previous {
                    if self.score(p, end, capacity, alpha, beta)
                        <= self.score(next, end, capacity, alpha, beta)
                    {
                        break;
                    }
                    self.queue.remove(p);
                    #[cfg(test)]
                    {
                        self.operations += 1;
                    }
                }
            }

            while let Some(second) = self.queue.second() {
                #[cfg(test)]
                {
                    self.operations += 1;
                }
                if self.capacity_penalty(second, end, capacity, alpha) == 0.
                    || self.time_penalty(second, end, beta) == 0.
                {
                    break;
                }
                let head = self.queue.head.unwrap();
                if self.score(head, end, capacity, alpha, beta)
                    <= self.score(second, end, capacity, alpha, beta)
                {
                    self.queue.remove(second);
                } else {
                    self.queue.remove(head);
                }
            }
            if let Some(second) = self.queue.second() {
                let head = self.queue.head.unwrap();
                if self.score(head, end, capacity, alpha, beta)
                    > self.score(second, end, capacity, alpha, beta)
                {
                    self.queue.remove(head);
                }
            }
            let best = self.queue.head.unwrap();
            cost[end] = self.score(best, end, capacity, alpha, beta)
                + self.distance[last]
                + self.incoming[last];
            pred[end] = best;
        }
    }
}
