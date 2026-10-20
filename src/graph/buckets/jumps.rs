use super::*;

impl BucketGraph {
    pub fn regenerate_jumps<const D: usize>(&mut self, g: &Graph<D>, forward: bool) {
        let mut identity = self.direction(forward).arc_counter;
        let mut attempts = 0;
        for v in 0..self.order.len() {
            let count = self.direction(forward).buckets[v].len();
            let mut later: BTreeMap<i32, (usize, VecDeque<usize>)> = BTreeMap::new();
            for k in (0..count).rev() {
                if !self.direction(forward).buckets[v][k].reachable {
                    continue;
                }
                self.direction_mut(forward).buckets[v][k]
                    .arcs
                    .retain(|a| a.jump.is_none());
                let ordinary: BTreeSet<_> = self.direction(forward).buckets[v][k]
                    .arcs
                    .iter()
                    .map(|a| g.arcs[a.physical].id)
                    .collect();
                let mut add = vec![];
                for (&physical, (index, targets)) in &mut later {
                    if ordinary.contains(&physical) {
                        if self.dimensions == 1 {
                            targets.pop_front();
                        }
                        targets.push_front(k);
                    } else if self.dimensions == 1 {
                        add.push((*index, targets[0]));
                    } else {
                        let stride = self.stride[v][1];
                        let mut row = k / stride;
                        let col = k % stride;
                        let mut last_col = stride;
                        for &target in targets.iter() {
                            let next_col = target % stride;
                            let next_row = target / stride;
                            if col <= next_col && next_col < last_col && row < next_row {
                                add.push((*index, target));
                                row = next_row;
                                last_col = next_col;
                                if col == next_col {
                                    break;
                                }
                            }
                        }
                    }
                }
                for (index, target) in add {
                    let mut a = BucketArc {
                        id: identity,
                        from: BucketId {
                            vertex: v,
                            bucket: k,
                        },
                        physical: index,
                        jump: Some(target),
                        flags: 0,
                    };
                    identity += 1;
                    attempts += 1;
                    a.flags = self.flags(g, forward, &a);
                    if a.flags != 0 {
                        self.direction_mut(forward).buckets[v][k].arcs.push(a);
                    }
                }
                for a in self.direction(forward).buckets[v][k]
                    .arcs
                    .iter()
                    .filter(|a| a.jump.is_none())
                {
                    later
                        .entry(g.arcs[a.physical].id)
                        .or_insert_with(|| (a.physical, VecDeque::from([k])));
                }
            }
        }
        self.direction_mut(forward).jump_attempts = attempts;
    }
}
