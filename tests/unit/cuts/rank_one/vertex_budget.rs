use super::*;

fn fixture(limit: usize) -> Separation {
    let mut separator = Separation::new(Parameters {
        max_per_vertex: limit,
        ..Default::default()
    });
    separator.graph_info = vec![Some(GraphInfo {
        packing_set_vertices: vec![vec![1], vec![2], vec![3]],
        covering_set_vertices: vec![vec![1], vec![2], vec![3]],
        ..Default::default()
    })];
    separator.preprocessed = vec![Some(Incidence {
        arc_tail: vec![1, 2],
        arc_head: vec![2, 3],
        ..Default::default()
    })];
    separator
}
fn cut(row: i32, memory_type: i32, memory: &[i32]) -> SeparatedCut {
    SeparatedCut {
        num_rows: 1,
        set_ids: vec![row],
        numerators: vec![1],
        denominator: 2,
        memory_type,
        memory: BTreeMap::from([(0, memory.iter().copied().collect())]),
        ..Default::default()
    }
}
fn select(separator: &mut Separation, candidates: Vec<SeparatedCut>, out: &mut Vec<SeparatedCut>) {
    let mut ranked = CandidateSet::new();
    for (i, mut cut) in candidates.into_iter().enumerate() {
        cut.id = i as i32;
        let handle = separator.push_candidate(cut);
        separator.insert_candidate(&mut ranked, i as f64, handle);
    }
    separator.insert_best(&[], &mut ranked, 2, false, true, out);
}

#[test]
fn vertex_budget_skips_overlap_and_refills_from_lower_ranked_candidates() {
    let mut separator = fixture(1);
    let mut out = vec![];
    select(
        &mut separator,
        vec![cut(0, 0, &[1, 2]), cut(1, 0, &[]), cut(2, 0, &[])],
        &mut out,
    );
    assert_eq!(
        out.iter().map(|c| c.set_ids[0]).collect::<Vec<_>>(),
        vec![0, 2]
    );
    // Previously selected cuts are charged across insert_best calls.
    select(&mut separator, vec![cut(1, 0, &[])], &mut out);
    assert_eq!(out.len(), 2);
}

#[test]
fn arc_endpoints_and_existing_load_are_charged_once_per_graph() {
    let mut separator = fixture(2);
    let candidate = cut(0, 1, &[0, 1]);
    assert_eq!(
        separator.cut_vertices(&candidate),
        BTreeSet::from([(0, 1), (0, 2), (0, 3)])
    );
    separator.vertex_cut_load.insert((0, 2), 2);
    // Another graph's saturated vertex does not consume this graph's budget.
    separator.vertex_cut_load.insert((1, 3), 2);
    let mut out = vec![];
    select(&mut separator, vec![candidate, cut(2, 0, &[])], &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].set_ids, vec![2]);
}

#[test]
fn disabled_vertex_budget_preserves_ranked_selection() {
    let mut separator = fixture(0);
    separator.vertex_cut_load.insert((0, 1), 100);
    let mut out = vec![];
    select(
        &mut separator,
        vec![cut(0, 0, &[1, 2]), cut(1, 0, &[1])],
        &mut out,
    );
    assert_eq!(out.len(), 2);
}
