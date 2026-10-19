use super::*;

impl BucketGraph {
    /// Refine cells while transferring only surviving physical arcs. Rebuilding
    /// from the physical graph here would silently undo reduced-cost fixing.
    pub fn decrease_steps(&mut self, g: &Graph) -> Result<()> {
        let old_steps = self.step;
        let old_forward = self.forward.clone();
        let old_backward = self.backward.clone();
        for r in 0..self.dimensions {
            self.step[r] = if self.dimensions == 1 {
                (old_steps[r] * 5. + 0.5).floor() / 10.
            } else {
                (old_steps[r] * 10. / 1.4 + 0.5).floor() / 10.
            };
        }
        if self.step[..self.dimensions].iter().any(|&s| s <= 0.) {
            self.step = old_steps;
            return Err(invalid("bucket step cannot be refined further"));
        }
        for forward in [true, false] {
            if !forward && !self.bidirectional {
                continue;
            }
            let old = if forward { &old_forward } else { &old_backward };
            self.regenerate(forward)?;
            let mut identity = 0;
            for v in 0..self.order.len() {
                let count = self.direction(forward).buckets[v].len();
                for k in 0..count {
                    let point = self.direction(forward).corners[v][k];
                    let mut selected = BTreeSet::new();
                    let old_row = (((self.upper[v][0] - self.lower[v][0]) / old_steps[0]).floor()
                        as usize)
                        + 1;
                    let origin = if forward {
                        point[0] - (self.lower[v][0] - 1e-6)
                    } else {
                        self.upper[v][0] + 1e-6 - point[0]
                    };
                    let mut index = (origin / old_steps[0]).floor() as usize;
                    if self.dimensions == 2 {
                        let origin = if forward {
                            point[1] - (self.lower[v][1] - 1e-6)
                        } else {
                            self.upper[v][1] + 1e-6 - point[1]
                        };
                        index += ((origin / old_steps[1]).floor() as usize) * old_row;
                    }
                    let bound0 = if forward {
                        (point[0] + self.step[0] - 1e-6).min(self.upper[v][0])
                    } else {
                        (point[0] - self.step[0] + 1e-6).max(self.lower[v][0])
                    };
                    let bound1 = if forward {
                        (point[1] + self.step[1] - 1e-6).min(self.upper[v][1])
                    } else {
                        (point[1] - self.step[1] + 1e-6).max(self.lower[v][1])
                    };
                    let within = if self.dimensions == 1 {
                        index
                    } else {
                        index % old_row
                    };
                    let origin0 = if forward {
                        self.lower[v][0] + within as f64 * old_steps[0]
                    } else {
                        self.upper[v][0] - within as f64 * old_steps[0]
                    };
                    let mut run1 = if forward {
                        self.lower[v][1] + (index / old_row) as f64 * old_steps[1]
                    } else {
                        self.upper[v][1] - (index / old_row) as f64 * old_steps[1]
                    };
                    let mut base = index;
                    loop {
                        let mut coord = origin0;
                        let mut scan = base;
                        loop {
                            if let Some(bucket) = old.buckets[v].get(scan) {
                                selected.extend(
                                    bucket
                                        .arcs
                                        .iter()
                                        .filter(|a| a.jump.is_none())
                                        .map(|a| a.physical),
                                );
                            }
                            coord = if forward {
                                coord + old_steps[0]
                            } else {
                                coord - old_steps[0]
                            };
                            if !coord.is_finite()
                                || if forward {
                                    bound0 <= coord - 1e-7
                                } else {
                                    coord + 1e-7 <= bound0
                                }
                            {
                                break;
                            }
                            scan += 1;
                        }
                        if self.dimensions != 2 {
                            break;
                        }
                        run1 = if forward {
                            run1 + old_steps[1]
                        } else {
                            run1 - old_steps[1]
                        };
                        base += old_row;
                        if base >= old.buckets[v].len()
                            || !run1.is_finite()
                            || if forward {
                                bound1 <= run1 - 1e-7
                            } else {
                                run1 + 1e-7 <= bound1
                            }
                        {
                            break;
                        }
                    }
                    let order: Vec<_> = self
                        .arc_order
                        .iter()
                        .copied()
                        .filter(|a| selected.contains(a))
                        .collect();
                    for physical in order {
                        let mut arc = BucketArc {
                            id: identity,
                            from: BucketId {
                                vertex: v,
                                bucket: k,
                            },
                            physical,
                            jump: None,
                            flags: 0,
                        };
                        identity += 1;
                        arc.flags = self.flags(g, forward, &arc);
                        if arc.flags != 0 {
                            self.direction_mut(forward).buckets[v][k].arcs.push(arc);
                        }
                    }
                }
            }
            self.direction_mut(forward).arc_counter = identity;
            // regenerateBuckets after a step change only scales the preceding
            // depths below: createBucketGraph's depth initialization (the
            // maximum with the new graph's chain depth) does not run here.
            let depth = self.direction(forward).depth.clone();
            self.order_components(g, forward)?;
            self.direction_mut(forward).depth = depth;
            self.refresh_counts(g, forward);
            self.regenerate_jumps(g, forward);
            let factor = if self.dimensions == 1 { 2. } else { 1.4 };
            for depth in &mut self.direction_mut(forward).depth {
                *depth = if *depth < 1 {
                    1
                } else {
                    (*depth as f64 * factor) as i32
                };
            }
        }
        self.version += 1;
        if self.bidirectional {
            self.backward_version += 1;
        }
        Ok(())
    }
}
