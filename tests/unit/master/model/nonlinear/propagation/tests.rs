use super::*;
use crate::cuts::CutRegistry;
use std::time::Duration;

fn graph(n: usize, arcs: &[(usize, usize)]) -> Graph<7> {
    let mut g = Graph::<7>::new(n, 1).unwrap();
    for &(a, b) in arcs {
        g.add_arc(a, b, 0.).unwrap();
    }
    g
}
fn cut(n: usize, support: &[usize]) -> RankOneCut {
    let mut registry = CutRegistry::default();
    registry.add_strong_k_path_cut(n, &[1], 1, 0.).unwrap();
    let mut cut = registry.cuts.remove(0);
    cut.weights.fill(0);
    for &v in support {
        cut.weights[v] = 1;
    }
    cut.visit_once = false;
    cut.class = CutClass::Packing;
    cut
}
fn run(g: &Graph<7>, cuts: &[RankOneCut]) -> Vec<usize> {
    zero_arcs(
        g,
        &cuts.iter().collect::<Vec<_>>(),
        |_| false,
        100_000,
        Instant::now() + Duration::from_secs(10),
    )
    .0
    .unwrap()
}

#[test]
fn joint_zero_kernels_remove_a_prefix_that_each_kernel_alone_supports() {
    let g = graph(6, &[(0, 1), (1, 2), (2, 5), (1, 3), (3, 5), (0, 4), (4, 5)]);
    let a = cut(6, &[1, 2]);
    let b = cut(6, &[1, 3]);
    assert!(!run(&g, &[a.clone()]).contains(&0));
    assert!(!run(&g, &[b.clone()]).contains(&0));
    assert_eq!(run(&g, &[a.clone(), b]), vec![0, 1, 2, 3, 4]);
    assert_eq!(run(&g, &[a.clone(), a]), vec![1, 2]);
}

#[test]
fn packing_resets_follow_vertex_and_physical_arc_memory() {
    let g = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
    let mut c = cut(5, &[1, 3]);
    assert_eq!(run(&g, &[c.clone()]), vec![0, 1, 2, 3]);
    c.memory[2] = false;
    assert!(run(&g, &[c.clone()]).is_empty());
    c.memory_kind = MemoryKind::Arc;
    c.arc_memory = [0, 1, 3].into_iter().collect();
    assert!(run(&g, &[c.clone()]).is_empty());
    c.arc_memory.insert(2);
    assert_eq!(run(&g, &[c]), vec![0, 1, 2, 3]);
}

#[test]
fn covering_entry_type_two_and_arc_weight_zero_contracts_remain_distinct() {
    let g = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
    let mut c = cut(5, &[1, 3]);
    c.memory[2] = false;
    c.class = CutClass::Covering;
    assert_eq!(run(&g, &[c.clone()]), vec![0, 1, 2, 3]);
    c.type2 = true;
    assert!(run(&g, &[c.clone()]).is_empty());
    c.type2 = false;
    c.visit_once = true;
    assert_eq!(run(&g, &[c.clone()]), vec![0, 1, 2, 3]);
    c.visit_once = false;
    c.class = CutClass::Packing;
    c.weights.fill(0);
    c.memory_kind = MemoryKind::Arc;
    c.arc_memory = (0..4).collect();
    c.arc_weights = [(0, 1), (2, 1)].into_iter().collect();
    assert_eq!(run(&g, &[c.clone()]), vec![0, 1, 2, 3]);
    c.arc_memory.remove(&2);
    assert!(run(&g, &[c]).is_empty());
}

#[test]
fn cycles_repeated_visits_and_parallel_arcs_keep_walk_support() {
    let g = graph(4, &[(0, 1), (1, 2), (2, 1), (1, 3), (0, 1), (2, 3)]);
    let mut c = cut(4, &[1]);
    assert_eq!(run(&g, &[c.clone()]), vec![2]);
    c.memory[2] = false;
    // Resetting the remainder permits arbitrarily many zero-cost cycles.
    assert!(run(&g, &[c.clone()]).is_empty());
    c.memory[2] = true;
    let (blocked, _) = zero_arcs(
        &g,
        &[&c],
        |a| a == 0 || a == 4,
        100_000,
        Instant::now() + Duration::from_secs(10),
    );
    assert_eq!(blocked.unwrap(), (0..6).collect::<Vec<_>>());
}

fn exhaustive(g: &Graph<7>, cuts: &[RankOneCut]) -> Vec<usize> {
    let mut registry = CutRegistry::default();
    registry.cuts = cuts.to_vec();
    let mut used = vec![false; g.arcs.len()];
    fn visit(
        g: &Graph<7>,
        registry: &CutRegistry,
        v: usize,
        path: &mut Vec<usize>,
        used: &mut [bool],
    ) {
        if v == g.sink {
            let physical: Vec<_> = path
                .iter()
                .map(|&a| {
                    let arc = &g.arcs[a];
                    (arc.tail, arc.head, arc.id)
                })
                .collect();
            if (0..registry.cuts.len())
                .all(|i| registry.route_coefficient(g, i, &physical, false).unwrap() == 0)
            {
                for &a in path.iter() {
                    used[a] = true;
                }
            }
            return;
        }
        for (a, arc) in g.arcs.iter().enumerate().filter(|(_, a)| a.tail == v) {
            path.push(a);
            visit(g, registry, arc.head, path, used);
            path.pop();
        }
    }
    visit(g, &registry, g.source, &mut vec![], &mut used);
    used.into_iter()
        .enumerate()
        .filter_map(|(a, yes)| (!yes).then_some(a))
        .collect()
}

#[test]
fn registered_zero_support_matches_every_path_in_8192_dag_domains() {
    let edges: Vec<_> = (0..5)
        .flat_map(|a| (a + 1..5).map(move |b| (a, b)))
        .collect();
    for mask in 0..1usize << edges.len() {
        let selected: Vec<_> = edges
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| (mask & (1 << i) != 0).then_some(e))
            .collect();
        let mut g = graph(5, &selected);
        for (i, arc) in g.arcs.iter_mut().enumerate() {
            arc.id = 7 + 13 * i as i32;
        }
        let a = cut(5, &[1, 2, 3]);
        let mut third = a.clone();
        third.denominator = 3;
        let mut reset = cut(5, &[1, 3]);
        reset.memory[2] = false;
        let mut arc = reset.clone();
        arc.memory_kind = MemoryKind::Arc;
        arc.arc_memory = g
            .arcs
            .iter()
            .filter(|a| a.tail != 2 && a.head != 2)
            .map(|a| a.id)
            .collect();
        if let Some(first) = g.arcs.first() {
            arc.arc_weights.insert(first.id, 1);
        }
        let mut covering = a.clone();
        covering.class = CutClass::Covering;
        let mut entry = reset.clone();
        entry.visit_once = true;
        entry.class = CutClass::Covering;
        let mut type_two = reset.clone();
        type_two.class = CutClass::Covering;
        type_two.type2 = true;
        let left = cut(5, &[1, 2]);
        let right = cut(5, &[2, 3]);
        for (kind, cuts) in [
            vec![a],
            vec![third],
            vec![reset],
            vec![arc],
            vec![covering],
            vec![entry],
            vec![type_two],
            vec![left.clone(), right, left],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                run(&g, &cuts),
                exhaustive(&g, &cuts),
                "mask={mask} kind={kind}"
            );
        }
    }
}

#[test]
fn interrupted_or_unsupported_walk_search_publishes_no_partial_exclusions() {
    let g = graph(6, &[(0, 1), (1, 2), (2, 5), (1, 3), (3, 5), (0, 4), (4, 5)]);
    let a = cut(6, &[1, 2]);
    let b = cut(6, &[1, 3]);
    let deadline = Instant::now() + Duration::from_secs(10);
    let (answer, spent) = zero_arcs(&g, &[&a, &b], |_| false, 100_000, deadline);
    let expected = answer.unwrap();
    for budget in [0, 1, 8, 32, spent / 2, spent - 1, spent] {
        let (result, used) = zero_arcs(&g, &[&a, &b], |_| false, budget, deadline);
        assert!(used <= budget);
        if budget < spent {
            assert!(result.is_none());
        } else {
            assert_eq!(result.unwrap(), expected);
        }
    }
    assert_eq!(
        zero_arcs(
            &g,
            &[&a],
            |_| false,
            100_000,
            Instant::now() - Duration::from_secs(1)
        ),
        (None, 0)
    );
    let mut unsupported = a;
    unsupported.graph_dependent_memory = true;
    assert!(zero_arcs(&g, &[&unsupported], |_| false, 100_000, deadline)
        .0
        .is_none());
}
