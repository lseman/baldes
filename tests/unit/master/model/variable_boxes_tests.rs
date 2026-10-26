use super::*;

fn chain() -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
}

fn register(
    graph: &Graph<2>,
    forbidden: &[bool],
    arc: i32,
    lower: f64,
    upper: Option<f64>,
) -> Result<Vec<Bounds>> {
    let definition = VariableBox {
        network: 7,
        variables: vec![ArcVariableBound { arc, lower, upper }],
    };
    Ok(
        prepare(&[definition], std::iter::once((7, graph, forbidden)))?
            .remove(&7)
            .unwrap(),
    )
}

#[test]
fn mandatory_positive_lower_envelope_keeps_the_declared_value() {
    let graph = chain();
    let bounds = register(&graph, &[], 0, 0.25, Some(9.)).unwrap();
    assert_eq!(bounds[0].lower, 0.25);
    assert_eq!(bounds[0].upper, 9.);
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_ok());
    assert!(register(&graph, &[], 0, 2., Some(2.)).is_err());
    let mut bypass = graph;
    bypass.add_arc(0, 2, 0.).unwrap();
    assert!(register(&bypass, &[], 0, 1., Some(1.)).is_err());
}

#[test]
fn parallel_arcs_are_individual_variables_not_mandatory_vertex_visits() {
    let mut graph = chain();
    graph.add_arc(0, 1, 0.).unwrap();
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_err());
    assert!(register(&graph, &[], 2, 1., Some(1.)).is_err());
    assert!(register(&graph, &[], 1, 1., Some(1.)).is_ok());
}

#[test]
fn walk_horizon_can_make_an_arc_mandatory_without_global_domination() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(0, 3, 0.).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_err());
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 2 })
        .unwrap();
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_ok());
}

#[test]
fn cyclic_bridge_unreachable_arc_and_repeat_limits_have_distinct_caps() {
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (1, 4), (3, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    assert!(register(&graph, &[], 0, 0., Some(1.)).is_ok());
    assert!(register(&graph, &[], 3, 0., Some(1.)).is_ok());
    assert!(register(&graph, &[], 4, 0., Some(0.)).is_ok());
    assert!(register(&graph, &[], 1, 0., Some(2.)).is_err());
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 6 })
        .unwrap();
    assert!(register(&graph, &[], 1, 0., Some(2.)).is_ok());
    assert!(register(&graph, &[], 2, 0., Some(2.)).is_ok());
    assert!(register(&graph, &[], 1, 0., Some(1.)).is_err());
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_ok());
}

#[test]
fn positive_parent_envelope_survives_an_empty_forbidden_child() {
    let graph = chain();
    assert!(register(&graph, &[], 0, 1., Some(1.)).is_ok());
    assert!(register(&graph, &[true, false], 0, 1., Some(1.)).is_ok());
    let mut bypass = graph;
    bypass.add_arc(0, 2, 0.).unwrap();
    assert!(register(&bypass, &[true, false, false], 0, 1., Some(1.)).is_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn large_visit_counts_are_compared_to_float_envelopes_without_rounding_down() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks {
            max_arcs: 9_007_199_254_740_995,
        })
        .unwrap();
    // At most H-2 self-loop visits. H-2 rounds down if cast to f64.
    assert!(register(&graph, &[], 1, 0., Some(9_007_199_254_740_992.)).is_err());
    assert!(register(&graph, &[], 1, 0., Some(9_007_199_254_740_994.)).is_ok());
}

fn enumerate(graph: &Graph<2>, forbidden: &[bool], cap: usize) -> Option<(Vec<usize>, Vec<usize>)> {
    fn visit(
        graph: &Graph<2>,
        forbidden: &[bool],
        vertex: usize,
        left: usize,
        counts: &mut [usize],
        result: &mut Option<(Vec<usize>, Vec<usize>)>,
    ) {
        if vertex == graph.sink {
            if let Some((minimum, maximum)) = result {
                for (i, &count) in counts.iter().enumerate() {
                    minimum[i] = minimum[i].min(count);
                    maximum[i] = maximum[i].max(count);
                }
            } else {
                *result = Some((counts.to_vec(), counts.to_vec()));
            }
        }
        // Continue through terminals too: the graph-walk relaxation contains
        // these walks even if a particular pricing engine stops at first sink.
        if left == 0 {
            return;
        }
        for (i, arc) in graph.arcs.iter().enumerate() {
            if arc.tail != vertex || forbidden.get(i) == Some(&true) {
                continue;
            }
            counts[i] += 1;
            visit(graph, forbidden, arc.head, left - 1, counts, result);
            counts[i] -= 1;
        }
    }
    let mut result = None;
    visit(
        graph,
        forbidden,
        graph.source,
        cap,
        &mut vec![0; graph.arcs.len()],
        &mut result,
    );
    result
}

#[test]
fn walk_envelopes_match_independent_exhaustive_small_graph_walks() {
    for mask in 0u16..512 {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for tail in 0..3 {
            for head in 0..3 {
                if mask & (1 << (tail * 3 + head)) != 0 {
                    graph.add_arc(tail, head, 0.).unwrap();
                }
            }
        }
        for filtered in [false, true] {
            let forbidden: Vec<_> = (0..graph.arcs.len())
                .map(|i| filtered && i % 3 == 0)
                .collect();
            for cap in 0..=6 {
                let mut proof = WalkProof::new(&graph, &forbidden);
                proof.cap = Some(cap);
                let expected = enumerate(&graph, &forbidden, cap);
                for (i, arc) in graph.arcs.iter().enumerate() {
                    assert_eq!(
                        proof.minimum(i),
                        expected.as_ref().map(|(lo, _)| lo[i]),
                        "mask={mask} cap={cap} arc={i}"
                    );
                    let maximum = if forbidden[i] {
                        Some(0)
                    } else {
                        proof.upper_bound(arc.tail, arc.head, 0.)
                    };
                    assert_eq!(
                        maximum,
                        Some(expected.as_ref().map_or(0, |(_, hi)| hi[i])),
                        "mask={mask} cap={cap} arc={i}"
                    );
                }
            }
        }
    }
}
