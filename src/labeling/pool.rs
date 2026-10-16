//! Native grouped solution pool. Costs tie by monotonically assigned candidate
//! identity; rejected candidates never consume an identity.
use super::{EPSILON, NO_ROUTE_COST};
#[derive(Clone, Debug)]
pub struct Candidate<T> {
    pub id: u64,
    pub group: usize,
    pub cost: f64,
    pub secondary: f64,
    pub value: T,
}
#[derive(Clone, Debug)]
pub struct SolutionPool<T> {
    pub group_limit: i32,
    pub total: usize,
    pub inserted: u64,
    pub threshold: f64,
    next_id: u64,
    references: Vec<Option<u64>>,
    a: Vec<Candidate<T>>,
    b: Vec<Candidate<T>>,
}
fn ordered<T>(set: &mut Vec<Candidate<T>>, candidate: Candidate<T>) {
    let position = set.partition_point(|c| {
        c.cost < candidate.cost || (c.cost == candidate.cost && c.id < candidate.id)
    });
    set.insert(position, candidate);
}
impl<T> SolutionPool<T> {
    pub fn new(groups: usize, total: usize, group_limit: i32) -> Self {
        Self {
            group_limit,
            total,
            inserted: 0,
            threshold: NO_ROUTE_COST,
            next_id: 0,
            references: vec![None; groups],
            a: vec![],
            b: vec![],
        }
    }
    pub fn len(&self) -> usize {
        self.a.len() + self.b.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn reference(&self, group: usize) -> Option<&Candidate<T>> {
        self.references
            .get(group)
            .copied()
            .flatten()
            .and_then(|id| self.a.iter().find(|c| c.id == id))
    }
    pub fn cutoff(&self, group: usize) -> f64 {
        if self.group_limit == 0 {
            return self.threshold;
        }
        let aw = self.a.last().map_or(-NO_ROUTE_COST, |c| c.cost);
        let bw = self.b.last().map_or(-NO_ROUTE_COST, |c| c.cost);
        match self.reference(group) {
            None => {
                if self.a.len() == self.group_limit as usize && self.len() == self.total {
                    aw.max(bw) - EPSILON
                } else {
                    NO_ROUTE_COST
                }
            }
            Some(c) => {
                if self.total <= self.len() {
                    c.cost.max(bw) - EPSILON
                } else {
                    NO_ROUTE_COST
                }
            }
        }
    }
    pub fn insert(&mut self, group: usize, cost: f64, secondary: f64, value: T) -> bool {
        if self.total == 0 || !cost.is_finite() || group >= self.references.len() {
            return false;
        }
        if self.group_limit == 0 {
            if cost >= self.threshold {
                return false;
            }
            let c = Candidate {
                id: self.next_id,
                group,
                cost,
                secondary,
                value,
            };
            self.next_id += 1;
            self.inserted += 1;
            ordered(&mut self.b, c);
            if self.b.len() > self.total {
                self.b.pop();
                self.threshold = self.b.last().unwrap().cost - EPSILON;
            }
            return true;
        }
        let reference = self.reference(group).map(|c| (c.id, c.cost));
        let accept = match reference {
            Some((_, previous)) => cost < previous - EPSILON,
            None => {
                self.group_limit < 1
                    || self.a.len() != self.group_limit as usize
                    || cost < self.a.last().unwrap().cost - EPSILON
            }
        };
        if !accept
            && !(self.len() != self.total
                || self.b.is_empty()
                || cost < self.b.last().unwrap().cost - EPSILON)
        {
            return false;
        }
        let c = Candidate {
            id: self.next_id,
            group,
            cost,
            secondary,
            value,
        };
        self.next_id += 1;
        self.inserted += 1;
        let mut displaced = None;
        if self.group_limit < 0 {
            ordered(&mut self.b, c);
        } else if reference.is_none() {
            self.references[group] = Some(c.id);
            ordered(&mut self.a, c);
            if self.a.len() > self.group_limit as usize {
                displaced = self.a.pop();
                self.references[displaced.as_ref().unwrap().group] = None;
            }
        } else if let Some((id, previous)) = reference {
            if previous - EPSILON > cost {
                self.references[group] = Some(c.id);
                ordered(&mut self.a, c);
                let position = self.a.iter().position(|c| c.id == id).unwrap();
                displaced = Some(self.a.remove(position));
            } else {
                ordered(&mut self.b, c);
            }
        }
        if let Some(c) = displaced {
            ordered(&mut self.b, c);
        }
        if self.len() > self.total {
            self.b.pop();
        }
        true
    }
    pub fn iter(&self) -> impl Iterator<Item = &Candidate<T>> {
        let mut all: Vec<_> = self.a.iter().chain(&self.b).collect();
        all.sort_by(|x, y| x.cost.total_cmp(&y.cost).then(x.id.cmp(&y.id)));
        all.into_iter()
    }
    pub fn into_sorted(self) -> Vec<Candidate<T>> {
        let mut all = self.a;
        all.extend(self.b);
        all.sort_by(|x, y| x.cost.total_cmp(&y.cost).then(x.id.cmp(&y.id)));
        all
    }
}
