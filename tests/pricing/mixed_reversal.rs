use super::*;
use baldes::graph::coupled::CoupledResource;

fn event(g: &Graph<2>, arc: usize, second: bool) -> (f64, f64) {
    let arc = &g.arcs[arc];
    let a = if arc.tail == g.sink { 0 } else { arc.tail };
    let b = if arc.head == g.sink { 0 } else { arc.head };
    let pair = (a.min(b), a.max(b));
    if pair == (1, 2) {
        (4., 0.)
    } else if pair == if second { (0, 2) } else { (0, 1) } {
        (0., 4.)
    } else if pair == (2, 3) && !second {
        (0., 1.)
    } else {
        (0., 0.)
    }
}

fn mixed_solver(elementary: bool, symmetric: bool, cut: bool, multiple: bool) -> Solver<2> {
    let mut s = solver(symmetric, cut);
    s.graph.elementary = elementary;
    for v in 0..s.graph.vertices.len() {
        s.graph.set_ng(v, &[]).unwrap();
    }
    for arc in &mut s.graph.arcs {
        arc.reduced_cost = -2.;
    }
    let mut definitions = vec![CoupledResource::Selection {
        initial: 3,
        allowed: vec![3; s.graph.arcs.len()],
    }];
    for second in 0..=usize::from(multiple) {
        let events: Vec<_> = (0..s.graph.arcs.len())
            .map(|arc| event(&s.graph, arc, second != 0))
            .collect();
        definitions.push(CoupledResource::SimultaneousPickupDelivery {
            capacity: 5.,
            deliveries: events.iter().map(|x| x.0).collect(),
            pickups: events.iter().map(|x| x.1).collect(),
        });
    }
    s.set_coupled_resources(&definitions).unwrap();
    s
}

// Enumerate literal physical walks without using a coupled state, completion
// hull, reversal transform or production dominance. With empty NG neighborhoods
// non-elementary walks may revisit a customer; the scalar unit consumption
// bounds every walk to four arcs.
fn walks(s: &Solver<2>, cut: bool, multiple: bool, cutoff: f64) -> BTreeMap<Vec<i32>, f64> {
    fn visit(
        s: &Solver<2>,
        vertex: usize,
        seen: u64,
        q: f64,
        cost: f64,
        path: &mut Vec<usize>,
        out: &mut BTreeMap<Vec<i32>, f64>,
        cut: bool,
        multiple: bool,
        cutoff: f64,
    ) {
        if vertex == s.graph.sink {
            for second in 0..=usize::from(multiple) {
                let total_delivery: f64 = path
                    .iter()
                    .map(|&i| event(&s.graph, i, second != 0).0)
                    .sum();
                let mut load = total_delivery;
                if load > 5. {
                    return;
                }
                for &arc in path.iter() {
                    let (delivery, pickup) = event(&s.graph, arc, second != 0);
                    load -= delivery;
                    assert!(load >= 0.);
                    load += pickup;
                    if load > 5. {
                        return;
                    }
                }
            }
            let cost = cost + if cut && seen & 6 != 0 { 1.25 } else { 0. };
            if cost < cutoff {
                out.insert(path.iter().map(|&i| s.graph.arcs[i].id).collect(), cost);
            }
            return;
        }
        for (i, arc) in s.graph.arcs.iter().enumerate() {
            if arc.tail != vertex || (s.graph.elementary && seen & (1 << arc.head) != 0) {
                continue;
            }
            let bounds = arc.bounds[0].unwrap_or(s.graph.vertices[arc.head].bounds[0]);
            let next = (q + arc.consumption[0]).max(bounds.lower);
            if next > bounds.upper {
                continue;
            }
            path.push(i);
            visit(
                s,
                arc.head,
                seen | (1 << arc.head),
                next,
                cost + arc.reduced_cost,
                path,
                out,
                cut,
                multiple,
                cutoff,
            );
            path.pop();
        }
    }
    let mut out = BTreeMap::new();
    visit(
        s,
        s.graph.source,
        1,
        0.,
        0.,
        &mut Vec::new(),
        &mut out,
        cut,
        multiple,
        cutoff,
    );
    out
}

#[test]
fn mixed_load_reversal_prices_fixes_and_enumerates_literal_walks() {
    for elementary in [false, true] {
        for symmetric in [false, true] {
            for cut in [false, true] {
                for multiple in [false, true] {
                    let mut s = mixed_solver(elementary, symmetric, cut, multiple);
                    let expected = walks(&s, cut, multiple, 100.);
                    assert!(!expected.is_empty());
                    let minimum = expected.values().copied().fold(f64::INFINITY, f64::min);
                    let priced = s.price_phase(0, 1000, false).unwrap();
                    assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
                    assert_eq!(priced.minimum_reduced_cost, minimum);
                    for column in priced.columns {
                        assert!(expected.contains_key(&column.arcs), "{:?}", column.arcs);
                        let original: f64 = column
                            .arcs
                            .iter()
                            .map(|id| s.graph.arcs.iter().find(|a| a.id == *id).unwrap().cost)
                            .sum();
                        assert_eq!(column.original_cost, original);
                    }
                    // Pricing may relax service memory through NG, whereas
                    // enumeration intentionally returns elementary paths.
                    let enumerable: BTreeMap<_, _> = expected
                        .into_iter()
                        .filter(|(path, _)| {
                            let mut seen = 1_u64;
                            path.iter().all(|id| {
                                let head = s.graph.arcs.iter().find(|a| a.id == *id).unwrap().head;
                                let new = seen & (1 << head) == 0;
                                seen |= 1 << head;
                                new
                            })
                        })
                        .collect();
                    assert_costs(&costs(s.enumerate(100.).unwrap()), &enumerable);
                    s.fix(-5.).unwrap();
                    let filtered = enumerable.into_iter().filter(|(_, c)| *c < -5.).collect();
                    assert_costs(&costs(s.enumerate(-5.).unwrap()), &filtered);
                }
            }
        }
    }
}

#[test]
fn mixed_load_reversal_keeps_reverse_only_suffixes_and_isolates_children() {
    let mut parent = mixed_solver(true, true, false, false);
    let expected = walks(&parent, false, false, 100.);
    let wanted: Vec<_> = [(0, 2), (2, 1), (1, 4)]
        .iter()
        .map(|&(u, v)| {
            parent
                .graph
                .arcs
                .iter()
                .find(|a| a.tail == u && a.head == v)
                .unwrap()
                .id
        })
        .collect();
    assert!(expected.contains_key(&wanted));
    parent.price_phase(0, 1000, false).unwrap();
    assert!(parent.engine.as_ref().unwrap().consumption_joins);
    let mut child = parent.fork_for_node();
    let forbidden = child
        .graph
        .arcs
        .iter()
        .position(|a| a.id == wanted[1])
        .unwrap();
    child.set_forbidden_arcs(vec![forbidden]).unwrap();
    child.price_phase(0, 1000, false).unwrap();
    assert!(!child.engine.as_ref().unwrap().consumption_joins);
    let allowed = expected
        .iter()
        .filter(|(path, _)| !path.contains(&wanted[1]))
        .map(|(p, c)| (p.clone(), *c))
        .collect();
    assert_costs(&costs(child.enumerate(100.).unwrap()), &allowed);
    assert_costs(&costs(parent.enumerate(100.).unwrap()), &expected);
}

#[test]
fn mixed_union_enumeration_does_not_apply_a_single_orientation_completion_callback() {
    use baldes::enumeration::{Controls, Enumeration};
    use std::cell::Cell;
    let mut s = mixed_solver(true, true, false, false);
    let expected = walks(&s, false, false, 100.);
    s.price_phase(0, 1000, false).unwrap();
    let engine = s.engine.as_ref().unwrap();
    assert!(engine.consumption_joins);
    let calls = Cell::new(0);
    let enumeration = Enumeration::run_with_symmetry(
        &s.graph,
        &engine.grid,
        &Controls {
            cutoff: 100.,
            preserve_arc_paths: true,
            ..Default::default()
        },
        &engine.reverse,
        None,
        |_, label| {
            calls.set(calls.get() + 1);
            label.coupled_state.as_ref().map_or(true, |state| {
                state
                    .load_summaries()
                    .iter()
                    .all(|load| load[0] + load[2] <= 5.)
            })
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 0, "ordinary-role callback has no union proof");
    assert_eq!(enumeration.paths.len, expected.len());
}
