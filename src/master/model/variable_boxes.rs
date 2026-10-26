//! Native-style variable boxes, verified as envelopes of the pricing domain.
use super::*;
use crate::master::preprocessing::Bounds;
use std::collections::VecDeque;

/// One physical arc variable envelope; omitted upper bounds stay unbounded.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcVariableBound {
    pub arc: i32,
    #[serde(default)]
    pub lower: f64,
    /// None denotes an unbounded variable.
    #[serde(default)]
    pub upper: Option<f64>,
}

/// Register the variable bounds used by native box admission. These are
/// envelopes, not new route restrictions: finite caps must already follow
/// from graph walks or an explicitly forbidden arc. A positive lower bound
/// must be implied by every complete walk in that domain. Unknown
/// bounds cannot silently restrict the model or prove pricing completion.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableBox {
    pub network: i32,
    pub variables: Vec<ArcVariableBound>,
}

pub(super) fn prepare<'a, const D: usize>(
    definitions: &[VariableBox],
    graphs: impl Iterator<Item = (i32, &'a Graph<D>, &'a [bool])>,
) -> Result<HashMap<i32, Vec<Bounds>>> {
    let graphs: HashMap<_, _> = graphs
        .map(|(id, graph, forbidden)| (id, (graph, forbidden)))
        .collect();
    let mut boxes = HashMap::new();
    for definition in definitions {
        if boxes.contains_key(&definition.network) {
            return Err(invalid("duplicate native variable-box network"));
        }
        let (graph, forbidden) = graphs
            .get(&definition.network)
            .ok_or_else(|| invalid("unknown native variable-box network"))?;
        // Tree registrations are validated before a zero-budget return and
        // before full problem preparation; graph walks must not index bad input.
        if graph.source >= graph.vertices.len()
            || graph.sink >= graph.vertices.len()
            || graph
                .arcs
                .iter()
                .any(|a| a.tail >= graph.vertices.len() || a.head >= graph.vertices.len())
        {
            return Err(invalid("invalid native variable-box graph endpoints"));
        }
        let positions: HashMap<_, _> = graph
            .arcs
            .iter()
            .enumerate()
            .map(|(p, a)| (a.id, p))
            .collect();
        if positions.len() != graph.arcs.len() {
            return Err(invalid("duplicate native variable-box graph arc IDs"));
        }
        let mut seen = HashSet::new();
        let mut bounds = vec![
            Bounds {
                lower: 0.,
                upper: f64::INFINITY
            };
            graph.arcs.len()
        ];
        let mut proof = None;
        for variable in &definition.variables {
            let &position = positions
                .get(&variable.arc)
                .ok_or_else(|| invalid("unknown native variable-box arc"))?;
            if !seen.insert(variable.arc) || !variable.lower.is_finite() {
                return Err(invalid(
                    "native variable boxes require unique arcs and a finite lower envelope",
                ));
            }
            let upper = variable.upper.unwrap_or(f64::INFINITY);
            if upper.is_nan() || upper < 0. || upper < variable.lower {
                return Err(invalid("invalid native variable-box upper envelope"));
            }
            let forbidden_arc = forbidden.get(position) == Some(&true);
            if variable.lower > 0. {
                if let Some(minimum) = proof
                    .get_or_insert_with(|| WalkProof::new(graph, forbidden))
                    .minimum(position)
                {
                    if variable.lower > minimum as f64 {
                        return Err(invalid(
                            "variable-box lower envelope is unproved for the graph-walk relaxation",
                        ));
                    }
                }
            }
            if upper.is_finite() {
                let cap = if forbidden_arc {
                    0
                } else {
                    let arc = &graph.arcs[position];
                    proof
                        .get_or_insert_with(|| WalkProof::new(graph, forbidden))
                        .upper_bound(arc.tail, arc.head, upper)
                        .ok_or_else(|| {
                            invalid("variable-box cap is unproved for cyclic pricing walks")
                        })?
                };
                // Comparing floor(upper) as an integer avoids rounding a large
                // walk count down to its neighbouring binary64 integer.
                if (upper as u128) < cap as u128 {
                    return Err(invalid(
                        "native variable-box cap must already contain the pricing walk domain",
                    ));
                }
            }
            bounds[position] = Bounds {
                lower: variable.lower,
                upper,
            };
        }
        boxes.insert(definition.network, bounds);
    }
    Ok(boxes)
}

/// A graph-walk relaxation of every feasible pricing path. Resources can
/// remove walks, so its minima/maxima are conservative envelope proofs.
struct WalkProof {
    source: usize,
    sink: usize,
    outgoing: Vec<Vec<(usize, usize)>>,
    from_source: Vec<Option<usize>>,
    to_sink: Vec<Option<usize>>,
    cap: Option<usize>,
    acyclic: bool,
}

fn distances(
    outgoing: &[Vec<(usize, usize)>],
    source: usize,
    excluded: Option<usize>,
) -> Vec<Option<usize>> {
    let mut distance = vec![None::<usize>; outgoing.len()];
    let mut queue = VecDeque::new();
    distance[source] = Some(0);
    queue.push_back(source);
    while let Some(tail) = queue.pop_front() {
        for &(head, edge) in &outgoing[tail] {
            if Some(edge) != excluded && distance[head].is_none() {
                distance[head] = distance[tail].and_then(|d| d.checked_add(1));
                if distance[head].is_some() {
                    queue.push_back(head);
                }
            }
        }
    }
    distance
}

impl WalkProof {
    fn new<const D: usize>(graph: &Graph<D>, forbidden: &[bool]) -> Self {
        let mut outgoing = vec![vec![]; graph.vertices.len()];
        let mut incoming = vec![vec![]; graph.vertices.len()];
        for (position, arc) in graph.arcs.iter().enumerate() {
            if forbidden.get(position) == Some(&true) {
                continue;
            }
            outgoing[arc.tail].push((arc.head, position));
            incoming[arc.head].push((arc.tail, position));
        }
        Self {
            source: graph.source,
            sink: graph.sink,
            from_source: distances(&outgoing, graph.source, None),
            to_sink: distances(&incoming, graph.sink, None),
            outgoing,
            cap: graph.max_route_arcs(),
            acyclic: graph.is_acyclic(),
        }
    }

    fn within_cap(&self, length: Option<usize>) -> bool {
        length.is_some_and(|length| self.cap.map_or(true, |cap| length <= cap))
    }

    fn minimum(&self, position: usize) -> Option<usize> {
        // Empty pricing domains satisfy every ordered envelope. In particular,
        // a child that forbids a formerly mandatory arc must keep its parent's
        // valid envelope rather than fail registration before infeasibility work.
        if !self.within_cap(self.from_source[self.sink]) {
            return None;
        }
        // A shortest walk avoids repeated vertices. If removing this arc
        // destroys every in-budget completion, every completion uses it once
        // or more. This proves a lower envelope without imposing a new row.
        Some(usize::from(!self.within_cap(
            distances(&self.outgoing, self.source, Some(position))[self.sink],
        )))
    }

    fn upper_bound(&self, tail: usize, head: usize, requested: f64) -> Option<usize> {
        if !self.within_cap(self.from_source[self.sink]) {
            return Some(0);
        }
        let (Some(prefix), Some(suffix)) = (self.from_source[tail], self.to_sink[head]) else {
            return Some(0);
        };
        let first = prefix as u128 + 1 + suffix as u128;
        if self.cap.is_some_and(|cap| first > cap as u128) {
            return Some(0);
        }
        if self.acyclic {
            return Some(1);
        }
        // A loose declared cap needs no stronger per-arc analysis. Preserve
        // that value in the registered box, as native preparation does.
        if let Some(cap) = self.cap {
            if requested as u128 >= cap as u128 {
                return Some(cap);
            }
        }
        let Some(back) = distances(&self.outgoing, head, None)[tail] else {
            return Some(1);
        };
        let cap = self.cap?;
        // k occurrences need a shortest prefix, one traversal, k-1 returns
        // from head to tail, and a shortest suffix. Joining those pieces
        // attains this maximum in the graph-walk relaxation, without O(cap)
        // unrolling or floating-point conversion of large visit counts.
        let count = 1 + (cap as u128 - first) / (back as u128 + 1);
        usize::try_from(count).ok()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/variable_boxes_tests.rs"]
mod tests;
