use super::*;

pub(super) type Record = (Vec<(usize, i64)>, Vec<usize>, Column);
struct Generated {
    class: ComponentClassBranching,
    variable_priority: f64,
    level: i32,
}

fn before(a: &Record, b: &Record) -> bool {
    for (&(aid, av), &(bid, bv)) in a.0.iter().zip(&b.0) {
        if aid != bid {
            return aid < bid;
        }
        if less(bv as f64, av as f64) {
            return true;
        }
        if less(av as f64, bv as f64) {
            return false;
        }
    }
    if a.0.len() != b.0.len() {
        return a.0.len() < b.0.len();
    }
    a.1 < b.1
}

pub(super) fn ordered_records(
    records: &[Record],
    mut indices: Vec<usize>,
    tree: &[&ComponentSequence],
    depth: usize,
    work: &mut ordering::Work,
) -> Result<Option<Vec<usize>>> {
    if !work.alive() {
        return Ok(None);
    }
    let Some(first) = tree.first() else {
        return Ok(
            ordering::stable(&mut indices, &|a, b| before(&records[a], &records[b]), work)
                .map(|()| indices),
        );
    };
    let mut bound = first
        .bounds
        .get(depth)
        .cloned()
        .ok_or_else(|| invalid("component ordering class prefix"))?;
    if bound.sense == ComponentSense::LessEqual {
        bound.threshold = finite(bound.threshold + 1.)?;
        bound.sense = ComponentSense::GreaterEqual;
    }
    let (mut yes, mut no) = (vec![], vec![]);
    for index in indices {
        if !work.alive() {
            return Ok(None);
        }
        if satisfies(
            &bound,
            records[index].2.values[bound.coordinate].unwrap_or(0) as f64,
        ) {
            yes.push(index);
        } else {
            no.push(index);
        }
    }
    let sibling = ComponentBound {
        coordinate: bound.coordinate,
        threshold: finite(bound.threshold - 1.)?,
        sense: ComponentSense::LessEqual,
    };
    let mut output = Vec::with_capacity(yes.len() + no.len());
    for (bound, subset) in [(bound, yes), (sibling, no)] {
        if subset.is_empty() {
            continue;
        }
        let following: Vec<_> = tree
            .iter()
            .copied()
            .filter(|class| {
                class.bounds.len() > depth + 1
                    && class.bounds[depth].coordinate == bound.coordinate
                    && class.bounds[depth].sense == bound.sense
                    && equal(class.bounds[depth].threshold, bound.threshold)
            })
            .collect();
        let Some(part) = ordered_records(records, subset, &following, depth + 1, work)? else {
            return Ok(None);
        };
        output.extend(part);
    }
    Ok(Some(output))
}

impl ComponentClassGenerator {
    fn template(&self, sequence: ComponentSequence) -> ComponentClassBranching {
        ComponentClassBranching {
            name: self.name.clone(),
            priority: self.priority,
            minimum_fractionality: self.minimum_fractionality,
            network: self.network,
            coordinates: self
                .coordinates
                .iter()
                .map(|v| v.coordinate.clone())
                .collect(),
            sequence,
        }
    }

    pub(crate) fn validate<const D: usize>(&self, graph: &Graph<D>) -> Result<()> {
        if self.name.trim().is_empty()
            || !self.priority.is_finite()
            || self.coordinates.is_empty()
            || !(0. ..=0.5).contains(&self.minimum_fractionality)
            || !(0. ..=0.5).contains(&self.integrality_tolerance)
            || self.coordinates.iter().any(|v| {
                !v.priority.is_finite()
                    || v.upper.is_nan()
                    || v.current_upper.is_nan()
                    || v.current_upper > v.upper
            })
        {
            return Err(invalid("component generator parameters"));
        }
        self.template(ComponentSequence::default())
            .for_graph(graph)?;
        if !self.ordering_coordinates.is_empty() {
            let mut ordering = self.template(ComponentSequence::default());
            ordering.coordinates = self.ordering_coordinates.clone();
            ordering.for_graph(graph)?;
            for coordinate in &self.coordinates {
                let matching = self
                    .ordering_coordinates
                    .iter()
                    .find(|c| c.id == coordinate.coordinate.id);
                if !matching.is_some_and(|c| c.arc_values == coordinate.coordinate.arc_values) {
                    return Err(invalid(
                        "component ordering registry omits branching coordinate",
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "multi-time-windows")]
    pub(crate) fn remap_arcs(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        for coordinate in self
            .coordinates
            .iter_mut()
            .map(|v| &mut v.coordinate)
            .chain(&mut self.ordering_coordinates)
        {
            let original = coordinate.arc_values.clone();
            for &(new, old) in copies {
                if new != coordinate.arc_values.len() {
                    return Err(invalid("component split arc mapping"));
                }
                coordinate.arc_values.push(
                    *original
                        .get(old)
                        .ok_or_else(|| invalid("component original arc mapping"))?,
                );
            }
        }
        Ok(())
    }

    fn values(
        coordinates: &[ComponentCoordinate],
        arcs: &[usize],
        work: &mut ordering::Work,
    ) -> Result<Option<Vec<Option<i64>>>> {
        let mut result = Vec::with_capacity(coordinates.len());
        for coordinate in coordinates {
            let mut value = None;
            for &arc in arcs {
                if work.step().is_none() {
                    return Ok(None);
                }
                let Some(next) = coordinate.arc_values[arc] else {
                    continue;
                };
                value = Some(
                    value
                        .unwrap_or(0_i64)
                        .checked_add(next)
                        .filter(|v| v.unsigned_abs() <= (1_u64 << 53))
                        .ok_or_else(|| invalid("component generator coordinate overflow"))?,
                );
            }
            result.push(value);
        }
        Ok(Some(result))
    }

    fn priority_of(&self, candidate: &Candidate) -> f64 {
        candidate
            .sequence
            .bounds
            .last()
            .map_or(100., |b| self.coordinates[b.coordinate].priority)
    }

    #[cfg(test)]
    pub(in crate::master::model) fn proposals<const D: usize>(
        &self,
        output: &Output<D>,
        quota: usize,
        tolerance: f64,
        deadline: Option<Instant>,
    ) -> Result<Vec<model_strong::Proposal>> {
        collect(
            output,
            std::slice::from_ref(self),
            quota,
            tolerance,
            deadline,
        )
    }

    fn raw_candidates<const D: usize>(
        &self,
        output: &Output<D>,
        deadline: Option<Instant>,
    ) -> Result<Vec<Generated>> {
        let mut work = ordering::Work::new(deadline);
        if !work.alive() {
            return Ok(vec![]);
        }
        let oracle = output
            .snapshot
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
            .ok_or_else(|| invalid("component generator network"))?;
        // Tree startup validates this registry before starting the solve budget.
        let coordinates: Vec<_> = self
            .coordinates
            .iter()
            .map(|v| v.coordinate.clone())
            .collect();
        let ordering = if self.ordering_coordinates.is_empty() {
            &coordinates
        } else {
            &self.ordering_coordinates
        };
        let mut ordering_positions = BTreeMap::new();
        for (position, coordinate) in ordering.iter().enumerate() {
            if work.step().is_none() {
                return Ok(vec![]);
            }
            ordering_positions.insert(coordinate.id, position);
        }
        let mut records = vec![];
        for (route, &quantity) in output.routes.iter().zip(&output.values) {
            if !work.alive() {
                return Ok(vec![]);
            }
            if route.network != self.network {
                continue;
            }
            let Some((remainder, fractional)) = self.record(quantity)? else {
                continue;
            };
            let mut arcs = Vec::with_capacity(route.route.arcs.len());
            for id in &route.route.arcs {
                if work.step().is_none() {
                    return Ok(vec![]);
                }
                arcs.push(
                    oracle
                        .index
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("component route arc"))?,
                );
            }
            let Some(values) = Self::values(&coordinates, &arcs, &mut work)? else {
                return Ok(vec![]);
            };
            let column = Column {
                quantity,
                remainder,
                fractional,
                values,
            };
            let ordering_values = if self.ordering_coordinates.is_empty() {
                column.values.clone()
            } else {
                let Some(values) = Self::values(ordering, &arcs, &mut work)? else {
                    return Ok(vec![]);
                };
                values
            };
            let mut key = vec![];
            for (&id, &position) in &ordering_positions {
                if work.step().is_none() {
                    return Ok(vec![]);
                }
                if let Some(value) = ordering_values[position] {
                    key.push((id, value));
                }
            }
            records.push((key, arcs, column));
        }
        let mut tree = vec![];
        for class in &output.snapshot.state.component_branches {
            if class.name != self.name || class.network != self.network {
                continue;
            }
            if class.coordinates.len() != coordinates.len()
                || class.coordinates.iter().any(|old| {
                    !coordinates
                        .iter()
                        .any(|new| old.id == new.id && old.arc_values == new.arc_values)
                })
            {
                return Err(invalid(
                    "component generator registry changed on inherited branch",
                ));
            }
            if class.sequence.bounds.is_empty() {
                continue;
            }
            let mut sequence = class.sequence.clone();
            for bound in &mut sequence.bounds {
                bound.coordinate = coordinates
                    .iter()
                    .position(|c| c.id == bound.coordinate)
                    .ok_or_else(|| invalid("component inherited coordinate"))?;
            }
            tree.push(sequence);
        }
        let Some(order) = ordered_records(
            &records,
            (0..records.len()).collect(),
            &tree.iter().collect::<Vec<_>>(),
            0,
            &mut work,
        )?
        else {
            return Ok(vec![]);
        };
        let columns: Vec<_> = order.iter().map(|&i| &records[i].2).collect();
        let mut candidates = vec![];
        self.explore(
            &columns,
            &mut vec![true; coordinates.len()],
            &mut ComponentSequence::default(),
            &tree.iter().collect::<Vec<_>>(),
            &mut candidates,
            deadline,
        )?;
        if !work.alive() {
            return Ok(vec![]);
        }
        let mut generated = vec![];
        for mut candidate in candidates {
            if !work.alive() {
                return Ok(vec![]);
            }
            if !candidate.class_weight.is_finite() {
                return Err(invalid("component class weight"));
            }
            let variable_priority = self.priority_of(&candidate);
            for bound in &mut candidate.sequence.bounds {
                bound.coordinate = coordinates[bound.coordinate].id;
            }
            generated.push(Generated {
                class: self.template(candidate.sequence),
                variable_priority,
                level: candidate.level,
            });
        }
        if !work.alive() {
            return Ok(vec![]);
        }
        Ok(generated)
    }
}

/// Raw publisher keys survive until the coordinator knows the current adaptive
/// quota. Publication runs at that quota rather than assuming a prefix law.
pub(in crate::master::model) struct ComponentPublication {
    keys: Vec<ordering::PriorityKey>,
}
impl ComponentPublication {
    pub(in crate::master::model) fn publish(
        &self,
        quota: usize,
        deadline: Option<Instant>,
    ) -> Option<Vec<usize>> {
        let mut work = ordering::Work::new(deadline);
        ordering::publish(&self.keys, quota, &mut work)
    }
}

#[cfg(test)]
pub(in crate::master::model::component) fn collect<const D: usize>(
    output: &Output<D>,
    generators: &[ComponentClassGenerator],
    quota: usize,
    tolerance: f64,
    deadline: Option<Instant>,
) -> Result<Vec<model_strong::Proposal>> {
    let mut proposals = collect_admission(output, generators, quota, tolerance, deadline)?;
    proposals.retain(|proposal| {
        proposal
            .component_admission
            .as_ref()
            .is_none_or(|view| view.legacy_order.is_some())
    });
    Ok(proposals)
}

pub(in crate::master::model::component) fn collect_admission<const D: usize>(
    output: &Output<D>,
    generators: &[ComponentClassGenerator],
    quota: usize,
    tolerance: f64,
    deadline: Option<Instant>,
) -> Result<Vec<model_strong::Proposal>> {
    let mut work = ordering::Work::new(deadline);
    let mut generated = vec![];
    let mut keys = vec![];
    for (family, generator) in generators.iter().enumerate() {
        if !work.alive() {
            return Ok(vec![]);
        }
        for candidate in generator.raw_candidates(output, deadline)? {
            keys.push(ordering::PriorityKey {
                family,
                family_priority: generator.priority,
                rule: if generator.priority_rule == ComponentPriorityRule::CoordinatePriority {
                    2
                } else {
                    0
                },
                variable_priority: candidate.variable_priority,
                added_level: candidate.level,
            });
            generated.push(Some(candidate));
        }
    }
    // Keep the original shared publisher as the compatibility admission view.
    // Native separation receives the whole quota independently per registration.
    let Some(legacy) = ordering::publish(&keys, quota, &mut work) else {
        return Ok(vec![]);
    };
    let mut publications = Vec::new();
    let mut local_indices = Vec::new();
    let mut start = 0;
    while start < keys.len() {
        let family = keys[start].family;
        let end = keys[start..]
            .iter()
            .position(|key| key.family != family)
            .map_or(keys.len(), |length| start + length);
        let publication = std::sync::Arc::new(ComponentPublication {
            keys: keys[start..end].to_vec(),
        });
        for index in start..end {
            publications.push(publication.clone());
            local_indices.push(index - start);
        }
        start = end;
    }
    let legacy_order: HashMap<_, _> = legacy
        .iter()
        .enumerate()
        .map(|(ordinal, &index)| (index, ordinal))
        .collect();
    // Legacy members must enter the tolerant candidate set in precisely their
    // prior order. Native-only members are removed before legacy sorting.
    let mut retained = legacy;
    for index in 0..keys.len() {
        if !legacy_order.contains_key(&index) {
            retained.push(index);
        }
    }
    let mut proposals = vec![];
    for index in retained {
        let candidate = generated[index].take().unwrap();
        if !work.alive() {
            return Ok(vec![]);
        }
        if let Some(mut proposal) = candidate.class.proposal(output, tolerance)? {
            proposal.restriction = model_strong::Restriction::GeneratedComponent {
                class: candidate.class,
                ordinal: legacy_order.get(&index).copied().unwrap_or(usize::MAX),
            };
            proposal.origin = Some(model_strong::ProducerFamily::ComponentGenerator(
                keys[index].family,
            ));
            proposal.component_admission = Some(Box::new(model_strong::ComponentAdmission {
                publication: publications[index].clone(),
                raw_index: local_indices[index],
                native_order: None,
                legacy_order: legacy_order.get(&index).copied(),
            }));
            proposals.push(proposal);
        }
    }
    if !work.alive() {
        return Ok(vec![]);
    }
    Ok(proposals)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/model/component_family_publication_tests.rs"]
mod family_publication_tests;
