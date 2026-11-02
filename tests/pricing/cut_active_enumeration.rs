use baldes::{
    cuts::{CutClass, MemoryKind, RankOneCut},
    pricing::Solver,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    str::FromStr,
};
fn take<T: FromStr>(t: &mut std::str::SplitWhitespace<'_>) -> T {
    t.next().unwrap().parse().ok().unwrap()
}
fn cuts(
    t: &mut std::str::SplitWhitespace<'_>,
    n: usize,
    count: usize,
    covering: bool,
) -> Vec<RankOneCut> {
    (0..count)
        .map(|_| {
            let id = take(t);
            let denominator = take(t);
            let price: f64 = take(t);
            let rows: usize = take(t);
            let ids: Vec<usize> = (0..rows).map(|_| take(t)).collect();
            let nums: Vec<i32> = (0..rows).map(|_| take(t)).collect();
            let mut weights = vec![0; n];
            let mut memory = vec![false; n];
            for (&id, &num) in ids.iter().zip(&nums) {
                weights[id + 1] = num;
                memory[id + 1] = true;
            }
            RankOneCut {
                weights,
                graph_support: Default::default(),
                graph_pricing_support: Default::default(),
                general_kpath_support: None,
                memory,
                graph_dependent_memory: false,
                graph_memories: Default::default(),
                memory_kind: MemoryKind::Vertex,
                arc_memory: Default::default(),
                graph_arc_memories: Default::default(),
                arc_weights: Default::default(),
                graph_arc_weights: Default::default(),
                visit_once: false,
                type2: false,
                denominator,
                dual: -price,
                class: if covering {
                    CutClass::Covering
                } else {
                    CutClass::Packing
                },
                rhs: if covering { 2 } else { 1 },
                kind: if covering { 1 } else { 0 },
                id,
                ordered_support: Some(ids.into_iter().zip(nums).map(|(i, v)| (i + 1, v)).collect()),
            }
        })
        .collect()
}
fn check_cut_active_enumeration_and_inspection(data: &str, covering: bool) {
    let mut t = data.split_whitespace();
    let n: usize = take(&mut t);
    let m: usize = take(&mut t);
    let cutoff: f64 = take(&mut t);
    let np: usize = take(&mut t);
    let ncuts: usize = take(&mut t);
    let rounds: usize = take(&mut t);
    let mut s = Solver::<2>::new(n, 1).unwrap();
    s.graph.vertices[0].elements.clear();
    s.graph.vertices[n - 1].elements.clear();
    for v in 0..n {
        s.graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for _ in 0..m {
        let tail = take(&mut t);
        let head = take(&mut t);
        let cost = take(&mut t);
        let reduced = take(&mut t);
        let consumption = take(&mut t);
        let id = s.graph.add_arc(tail, head, cost).unwrap();
        let a = s.graph.arc_mut(id).unwrap();
        a.reduced_cost = reduced;
        a.consumption[0] = consumption;
    }
    let native: BTreeSet<Vec<usize>> = (0..np)
        .map(|_| {
            let len: usize = take(&mut t);
            (0..len).map(|_| take(&mut t)).collect()
        })
        .collect();
    s.cuts.cuts = cuts(&mut t, n, ncuts, covering);
    s.bucket_steps[0] = 1.;
    let actual: BTreeSet<Vec<usize>> = s
        .enumerate(cutoff)
        .unwrap()
        .iter()
        .map(|r| r.arcs.iter().map(|&i| i as usize).collect())
        .collect();
    assert_eq!(actual, native, "enumeration");
    for round in 0..rounds {
        let ncuts: usize = take(&mut t);
        let npaths: usize = take(&mut t);
        s.cuts.cuts = cuts(&mut t, n, ncuts, covering);
        for a in &mut s.graph.arcs {
            a.reduced_cost = take(&mut t)
        }
        let expected: BTreeMap<Vec<i32>, u64> = (0..npaths)
            .map(|_| {
                let len: usize = take(&mut t);
                let path = (0..len).map(|_| take(&mut t)).collect();
                let bits = take(&mut t);
                (path, bits)
            })
            .collect();
        let actual: BTreeMap<_, _> = s
            .inspect_enumeration(f64::INFINITY)
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|r| (r.arcs, r.cost.to_bits()))
            .collect();
        assert_eq!(actual, expected, "round {round}");
        if round == 1 {
            let kept = s.inspect_enumeration(0.49999999640299997).unwrap().unwrap();
            assert_eq!(kept.len(), 15);
        }
    }
    println!("matched {np} cut-active paths and {rounds} repricing snapshots including native inspection cutoff");
}

#[test]
fn cut_active_enumeration_and_inspection_match_reference() {
    check_cut_active_enumeration_and_inspection(
        include_str!("../fixtures/enumeration/cut-active-reference-capture.txt"),
        false,
    );
}

#[test]
fn covering_cut_enumeration_and_inspection_match_controlled_reference() {
    check_cut_active_enumeration_and_inspection(
        include_str!("../fixtures/enumeration/covering-wide-pricing-capture.txt"),
        true,
    );
}

fn check_full_memory_join(capture: &str, mask: u32, denominator: i32, covering: bool) {
    use baldes::cuts::shared::state::{
        full_memory_concatenation_cost, CutResourceInfo, FullField, FullPreparation,
    };

    // The packing descriptor and potential come from the initialized native model;
    // the covering descriptor was substituted during the native probe.
    let mut info = CutResourceInfo::default();
    info.vertex_cut_potential.resize(11, 0.);
    let prep = FullPreparation {
        fields: vec![FullField {
            mask,
            denominator,
            word: 0,
            shift: 0,
            covering,
            cost: 1.,
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut data = capture.split_whitespace();
    for case in 0..32 {
        let left = [take::<u32>(&mut data)];
        let right = [take::<u32>(&mut data)];
        let extra: f64 = take(&mut data);
        let expected: u64 = take(&mut data);
        let actual =
            full_memory_concatenation_cost(&info, &prep, 10, (&left, extra), (&right, -0.125), 0.5);
        assert_eq!(actual.to_bits(), expected, "case {case}");
    }
    assert!(data.next().is_none());
}

#[test]
fn full_memory_packing_join_matches_reference_remainders() {
    check_full_memory_join(
        include_str!("../fixtures/enumeration/cut-join-reference-capture.txt"),
        1,
        2,
        false,
    );
}

#[test]
fn full_memory_covering_join_matches_reference_remainders() {
    check_full_memory_join(
        include_str!("../fixtures/enumeration/cut-join-covering-reference-capture.txt"),
        3,
        -2,
        true,
    );
}

#[test]
fn full_memory_multiple_fields_match_reference_accumulation() {
    use baldes::cuts::shared::state::{
        full_memory_concatenation_cost, CutResourceInfo, FullField, FullPreparation,
    };
    let mut info = CutResourceInfo::default();
    info.vertex_cut_potential.resize(11, 0.);
    info.vertex_cut_potential[10] = 0.375;
    let fields = [
        (1, 2, 0, 0, false, 1e16),
        (3, -2, 0, 4, true, 0.5),
        (1, 2, 1, 31, false, -1e16),
        (3, 3, 2, 1, false, 1.25),
    ];
    let prep = FullPreparation {
        fields: fields
            .into_iter()
            .map(
                |(mask, denominator, word, shift, covering, cost)| FullField {
                    mask,
                    denominator,
                    word,
                    shift,
                    covering,
                    cost,
                    ..Default::default()
                },
            )
            .collect(),
        ..Default::default()
    };
    let mut data =
        include_str!("../fixtures/enumeration/cut-join-multifield-reference-capture.txt")
            .split_whitespace();
    for case in 0..256 {
        let left = [take(&mut data), take(&mut data), take(&mut data)];
        let right = [take(&mut data), take(&mut data), take(&mut data)];
        let forward = f64::from_bits(take(&mut data));
        let backward = f64::from_bits(take(&mut data));
        let initial = f64::from_bits(take(&mut data));
        let expected: u64 = take(&mut data);
        let actual = full_memory_concatenation_cost(
            &info,
            &prep,
            10,
            (&left, forward),
            (&right, backward),
            initial,
        );
        assert_eq!(actual.to_bits(), expected, "case {case}");
    }
    assert!(data.next().is_none());
}
