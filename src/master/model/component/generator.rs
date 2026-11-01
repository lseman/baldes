//! Component-class proposals from fractional column disaggregation.
//!
//! Publication and phase-one selection are handled by the tree dispatcher.
use super::*;
mod integration;
mod ordering;
pub(super) use integration::collect;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComponentVariableKind {
    Binary,
    Integer,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComponentGeneratorCoordinate {
    pub coordinate: ComponentCoordinate,
    pub kind: ComponentVariableKind,
    pub priority: f64,
    pub current_upper: f64,
    pub upper: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComponentPriorityRule {
    #[default]
    AddedDepth,
    CoordinatePriority,
}

/// Registered family of integer additive column coordinates.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComponentClassGenerator {
    pub name: String,
    pub network: i32,
    /// Outer family priority level. Class-coordinate priority and added depth
    /// are retained separately when selecting equal-level component families.
    pub priority: f64,
    pub coordinates: Vec<ComponentGeneratorCoordinate>,
    /// Complete sparse original-coordinate order when the branching family is
    /// only a subset of the integer coordinates stored in column solutions.
    #[serde(default)]
    pub ordering_coordinates: Vec<ComponentCoordinate>,
    #[serde(default)]
    pub priority_rule: ComponentPriorityRule,
    #[serde(default = "default_minimum_fractionality")]
    pub minimum_fractionality: f64,
    #[serde(default = "default_integrality_tolerance")]
    pub integrality_tolerance: f64,
}

fn default_integrality_tolerance() -> f64 {
    1e-8
}

/// The record carried by the fractional column list. The full quantity is
/// retained separately from its fractional remainder when forming class weights.
pub(super) struct Column {
    pub quantity: f64,
    pub remainder: f64,
    pub fractional: bool,
    pub values: Vec<Option<i64>>,
}

pub(super) struct Candidate {
    pub sequence: ComponentSequence,
    pub level: i32,
    pub class_weight: f64,
}

#[derive(Clone)]
struct Split {
    bound: ComponentBound,
    greater: usize,
    lesser: usize,
    position: f64,
}

fn less(a: f64, b: f64) -> bool {
    a < b - (a.abs().max(b.abs()) * 1e-10 + 1e-6)
}

fn equal(a: f64, b: f64) -> bool {
    !less(a, b) && !less(b, a)
}

fn unit_bound(value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    let slack = value.abs().max(1.) * 1e-6 + 1e-6;
    !(value < 1. - slack || 1. < value - slack)
}

fn finite(value: f64) -> Result<f64> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| invalid("component generator arithmetic"))
}

fn floor(value: f64) -> f64 {
    let mut rounded = (value + value * 1e-10 + 1e-6).floor();
    if value - 1. + value * 1e-10 + 1e-6 > rounded {
        rounded += 1.;
    }
    if rounded.abs() <= 1e-6 {
        rounded = 0.;
    }
    rounded
}

fn ceil(value: f64) -> f64 {
    let mut rounded = (value - value * 1e-10 - 1e-6).ceil();
    if value + 1. <= rounded {
        rounded -= 1.;
    }
    if rounded.abs() <= 1e-6 {
        rounded = 0.;
    }
    rounded
}

fn fraction(value: f64) -> f64 {
    let distance = (value - floor(value)).min(ceil(value) - value);
    if distance.abs() <= 1e-6 {
        0.
    } else {
        distance
    }
}

fn satisfies(bound: &ComponentBound, value: f64) -> bool {
    match bound.sense {
        ComponentSense::GreaterEqual => !less(value, bound.threshold),
        ComponentSense::LessEqual => !less(bound.threshold, value),
    }
}

/// Pack decreasing coordinate values across consecutive unit quantities.
/// Equal values retain column-list order, including explicitly stored zeros.
fn disaggregate(
    columns: &[&Column],
    coordinate: usize,
    deadline: Option<Instant>,
) -> Result<Option<Vec<f64>>> {
    let mut work = ordering::Work::new(deadline);
    let Some(order) = column_order(columns, coordinate, &mut work) else {
        return Ok(None);
    };
    let mut bins = vec![0.];
    let mut mass: f64 = 0.;
    let mut index = 0;
    for column_index in order {
        if !work.alive() {
            return Ok(None);
        }
        let column = columns[column_index];
        let value = column.values[coordinate].unwrap() as f64;
        let mut remaining = column.remainder;
        while less(0., remaining) {
            let boundary = (index + 1) as f64;
            let slack = mass.abs().max(boundary.abs()) * 1e-6 + 1e-6;
            if !(mass < boundary - slack) && mass - slack <= boundary {
                bins.push(0.);
                index += 1;
            }
            let boundary = (index + 1) as f64;
            let total = finite(mass + remaining)?;
            let consumed = if less(boundary, total) {
                boundary - mass
            } else {
                remaining
            };
            mass = finite(mass + consumed)?;
            remaining = finite(remaining - consumed)?;
            bins[index] = finite(value * consumed + bins[index])?;
        }
    }
    Ok(Some(bins))
}

fn column_order(
    columns: &[&Column],
    coordinate: usize,
    work: &mut ordering::Work,
) -> Option<Vec<usize>> {
    let mut ordered = ordering::Tree::new();
    for (index, column) in columns.iter().enumerate() {
        if column.values[coordinate].is_some()
            && ordered
                .insert_equal(
                    index,
                    &|a, b| {
                        less(
                            columns[b].values[coordinate].unwrap() as f64,
                            columns[a].values[coordinate].unwrap() as f64,
                        )
                    },
                    work,
                )
                .is_none()
        {
            return None;
        }
    }
    work.alive().then(|| ordered.ordered())
}

impl ComponentClassGenerator {
    fn groups<'a>(
        &self,
        columns: &[&'a Column],
        eligible: &[bool],
        deadline: Option<Instant>,
    ) -> Option<Vec<(usize, Vec<&'a Column>)>> {
        let mut work = ordering::Work::new(deadline);
        let mut tree = ordering::Tree::new();
        let mut groups: Vec<(usize, Vec<&Column>)> = vec![];
        let mut coordinate_order = BTreeMap::new();
        for (i, coordinate) in self.coordinates.iter().enumerate() {
            work.step()?;
            coordinate_order.insert(coordinate.coordinate.id, i);
        }
        for &column in columns {
            if !work.alive() {
                return None;
            }
            for &v in coordinate_order.values() {
                if !eligible[v] || column.values[v].is_none() {
                    continue;
                }
                let key = groups.len();
                let (group, inserted) = tree.insert_unique(
                    key,
                    &|a, b| {
                        let a = if a == key { v } else { groups[a].0 };
                        let b = if b == key { v } else { groups[b].0 };
                        let a = &self.coordinates[a];
                        let b = &self.coordinates[b];
                        if less(b.priority, a.priority) {
                            true
                        } else if less(a.priority, b.priority) {
                            false
                        } else {
                            a.coordinate.id < b.coordinate.id
                        }
                    },
                    &mut work,
                )?;
                if inserted {
                    groups.push((v, vec![]));
                }
                groups[group].1.push(column);
            }
        }
        let mut groups: Vec<_> = groups.into_iter().map(Some).collect();
        Some(
            tree.ordered()
                .into_iter()
                .map(|i| groups[i].take().unwrap())
                .collect(),
        )
    }
    fn fractional(&self, value: f64) -> bool {
        less(self.minimum_fractionality, fraction(value))
    }

    fn record(&self, quantity: f64) -> Result<Option<(f64, bool)>> {
        if !quantity.is_finite() || quantity.abs() >= (1_u64 << 53) as f64 {
            return Err(invalid("component generator LP quantity"));
        }
        let distance = fraction(quantity);
        if !less(1e-10, distance) {
            return Ok(None);
        }
        Ok(Some((
            finite(quantity - floor(quantity))?,
            less(self.integrality_tolerance, distance),
        )))
    }

    fn explore(
        &self,
        columns: &[&Column],
        eligible: &mut [bool],
        sequence: &mut ComponentSequence,
        tree: &[&ComponentSequence],
        candidates: &mut Vec<Candidate>,
        deadline: Option<std::time::Instant>,
    ) -> Result<()> {
        if columns.is_empty() || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        let Some(first) = tree.first() else {
            return self.separate(columns, eligible, sequence, 0, candidates, deadline);
        };
        let depth = sequence.bounds.len();
        let mut bound = first
            .bounds
            .get(depth)
            .cloned()
            .ok_or_else(|| invalid("component class tree prefix"))?;
        if bound.sense == ComponentSense::LessEqual {
            bound.threshold = finite(bound.threshold + 1.)?;
            bound.sense = ComponentSense::GreaterEqual;
        }
        let v = bound.coordinate;
        let (satisfying, other): (Vec<_>, Vec<_>) = columns
            .iter()
            .copied()
            .partition(|c| satisfies(&bound, c.values[v].unwrap_or(0) as f64));
        let weight = satisfying
            .iter()
            .try_fold(0., |sum, c| finite(sum + c.remainder))?;
        if self.fractional(weight) {
            let Some(bins) = disaggregate(&satisfying, v, deadline)? else {
                return Ok(());
            };
            let last = *bins.last().unwrap();
            let threshold = ceil(last);
            let slack = threshold.abs().max(bound.threshold.abs()) * 1e-6 + 1e-6;
            if !(threshold < bound.threshold - slack || bound.threshold < threshold - slack)
                && self.fractional(last)
            {
                let mut sequence = sequence.clone();
                sequence.bounds.push(bound.clone());
                candidates.push(Candidate {
                    sequence,
                    level: 0,
                    class_weight: weight,
                });
            }
        }
        let minimum = if self.coordinates[v].kind == ComponentVariableKind::Integer {
            1
        } else {
            2
        };
        let sibling = ComponentBound {
            coordinate: v,
            threshold: finite(bound.threshold - 1.)?,
            sense: ComponentSense::LessEqual,
        };
        for (bound, columns) in [(bound, satisfying), (sibling, other)] {
            if columns.iter().filter(|c| c.fractional).count() < minimum {
                continue;
            }
            let following: Vec<_> = tree
                .iter()
                .copied()
                .filter(|class| {
                    class.bounds.len() > depth + 1
                        && class.bounds[depth].coordinate == v
                        && class.bounds[depth].sense == bound.sense
                        && equal(class.bounds[depth].threshold, bound.threshold)
                })
                .collect();
            let saved = eligible[v];
            if unit_bound(self.coordinates[v].upper) {
                eligible[v] = false;
            }
            sequence.bounds.push(bound);
            self.explore(
                &columns, eligible, sequence, &following, candidates, deadline,
            )?;
            sequence.bounds.pop();
            eligible[v] = saved;
        }
        Ok(())
    }

    fn separate(
        &self,
        columns: &[&Column],
        eligible: &mut [bool],
        sequence: &mut ComponentSequence,
        level: i32,
        candidates: &mut Vec<Candidate>,
        deadline: Option<std::time::Instant>,
    ) -> Result<()> {
        if columns.is_empty() || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        let weight = columns
            .iter()
            .try_fold(0., |sum, column| finite(sum + column.remainder))?;
        if self.fractional(weight) {
            candidates.push(Candidate {
                sequence: sequence.clone(),
                level,
                class_weight: weight,
            });
            return Ok(());
        }
        let Some(groups) = self.groups(columns, eligible, deadline) else {
            return Ok(());
        };
        let mut best: Option<Split> = None;
        let mut splits = vec![];
        let mut found = false;
        for (v, group) in groups {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(());
            }
            let coordinate = &self.coordinates[v];
            let Some(bins) = disaggregate(&group, v, deadline)? else {
                return Ok(());
            };
            let start = if coordinate.kind == ComponentVariableKind::Binary {
                bins.len() - 1
            } else {
                0
            };
            for (j, &bin) in bins.iter().enumerate().skip(start) {
                let bound = ComponentBound {
                    coordinate: v,
                    threshold: ceil(bin),
                    sense: ComponentSense::GreaterEqual,
                };
                let mut greater_weight = 0.;
                let mut greater = 0;
                let mut lesser = 0;
                for column in &group {
                    let Some(value) = column.values[v] else {
                        continue;
                    };
                    if satisfies(&bound, value as f64) {
                        greater_weight = finite(greater_weight + column.quantity)?;
                        greater += usize::from(column.fractional);
                    } else {
                        lesser += usize::from(column.fractional);
                    }
                }
                let integer = coordinate.kind == ComponentVariableKind::Integer;
                let valid = !integer
                    || sequence
                        .bounds
                        .iter()
                        .all(|old| old.coordinate != v || satisfies(old, bound.threshold));
                let same = integer
                    && sequence.bounds.iter().any(|old| {
                        old.coordinate == v
                            && old.sense == bound.sense
                            && equal(old.threshold, bound.threshold)
                    });
                if self.fractional(bin) && self.fractional(greater_weight) {
                    if valid {
                        let mut sequence = sequence.clone();
                        sequence.bounds.push(bound);
                        candidates.push(Candidate {
                            sequence,
                            level: level + 1,
                            class_weight: greater_weight,
                        });
                        found = true;
                    }
                } else if valid && !same && greater >= if integer { 1 } else { 2 } {
                    let split = Split {
                        bound,
                        greater,
                        lesser,
                        position: fraction((j + 1) as f64 / bins.len() as f64),
                    };
                    if level < 0 {
                        splits.push(split);
                    } else if best.as_ref().map_or(true, |previous| {
                        let previous_priority =
                            self.coordinates[previous.bound.coordinate].priority;
                        !less(coordinate.priority, previous_priority)
                            && (less(previous_priority, coordinate.priority)
                                || !less(split.position, previous.position))
                    }) {
                        best = Some(split);
                    }
                }
            }
        }
        if level >= 2 && self.priority_rule != ComponentPriorityRule::CoordinatePriority && found {
            return Ok(());
        }
        if let Some(best) = best {
            splits.push(best);
        }
        for split in splits {
            let v = split.bound.coordinate;
            let coordinate = &self.coordinates[v];
            let minimum = if coordinate.kind == ComponentVariableKind::Integer {
                1
            } else {
                2
            };
            let sibling = ComponentBound {
                coordinate: v,
                threshold: finite(split.bound.threshold - 1.)?,
                sense: ComponentSense::LessEqual,
            };
            for (bound, count, remove) in [
                (
                    split.bound,
                    split.greater,
                    unit_bound(coordinate.current_upper),
                ),
                (sibling, split.lesser, unit_bound(coordinate.upper)),
            ] {
                if count < minimum {
                    continue;
                }
                let filtered: Vec<_> = columns
                    .iter()
                    .filter(|column| satisfies(&bound, column.values[v].unwrap_or(0) as f64))
                    .copied()
                    .collect();
                let saved = eligible[v];
                if remove {
                    eligible[v] = false;
                }
                sequence.bounds.push(bound);
                self.separate(
                    &filtered,
                    eligible,
                    sequence,
                    level + 1,
                    candidates,
                    deadline,
                )?;
                sequence.bounds.pop();
                eligible[v] = saved;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/component_generator_tests.rs"]
mod tests;
