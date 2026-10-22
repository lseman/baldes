use super::*;

#[test]
fn independent_primal_order_preserves_fill_and_kkt_solution() {
    let (m, n) = (9, 80);
    let mut entries = Vec::new();
    for j in 0..n {
        for i in 0..m {
            if j == 0 || i == j % m || i == (3 * j + 1) % m {
                entries.push((i, j, 0.25 + ((i + j) % 5) as f64 * 0.2));
            }
        }
    }
    let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
    let mut rows = vec![vec![]; m];
    for &(i, j, _) in &entries {
        rows[i].push(j);
    }
    let order = block_order(&a);
    assert_eq!(order[..n], (0..n).collect::<Vec<_>>());
    let mut previous = column_order(&rows, n);
    assert_ne!(previous, order[..n]);
    previous.extend_from_slice(&order[n..]);
    let mut kkt: Vec<_> = (0..n).map(|j| (j, j, -1. - j as f64 / n as f64)).collect();
    kkt.extend((0..m).map(|i| (n + i, n + i, 1.)));
    for &(i, j, value) in &entries {
        kkt.push((n + i, j, value));
        kkt.push((j, n + i, value));
    }
    let kkt = SparseMatrix::from_triplets(n + m, n + m, &kkt).unwrap();
    let current = Symbolic::analyze(&kkt, order, n);
    let previous = Symbolic::analyze(&kkt, previous, n);
    let fill = |symbolic: &Symbolic| {
        symbolic
            .fronts
            .iter()
            .map(|f| {
                let np = f.end - f.lo;
                np * (np - 1) / 2 + np * f.updates.len()
            })
            .sum::<usize>()
    };
    assert_eq!(fill(&current), fill(&previous));
    let expected: Vec<_> = (0..n + m).map(|i| (i % 11) as f64 / 11. - 0.25).collect();
    let rhs = kkt.mul(&expected);
    for symbolic in [current, previous] {
        let sparse = Uplooking::new(&symbolic, n).unwrap();
        for factor in [
            symbolic.factorize(&kkt, true).unwrap(),
            sparse.factorize(&symbolic, &kkt).unwrap(),
        ] {
            let solution = factor.solve(&rhs).unwrap();
            for (x, wanted) in solution.iter().zip(&expected) {
                assert!((x - wanted).abs() < 1e-10);
            }
            for (value, wanted) in kkt.mul(&solution).iter().zip(&rhs) {
                assert!((value - wanted).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn minimum_degree_matches_explicit_elimination() {
    // Exhaust all undirected five-vertex graphs, including graphs that
    // become complete only after fill, degree ties and isolated vertices.
    fn reference(mut graph: Vec<BTreeSet<usize>>) -> Vec<usize> {
        let n = graph.len();
        let mut active = vec![true; n];
        let mut order = Vec::new();
        for _ in 0..n {
            let k = (0..n)
                .filter(|&i| active[i])
                .min_by_key(|&i| (graph[i].len(), i))
                .unwrap();
            let neighbors: Vec<_> = graph[k].iter().copied().collect();
            for &i in &neighbors {
                graph[i].remove(&k);
            }
            for (p, &i) in neighbors.iter().enumerate() {
                for &j in &neighbors[p + 1..] {
                    graph[i].insert(j);
                    graph[j].insert(i);
                }
            }
            graph[k].clear();
            active[k] = false;
            order.push(k);
        }
        order
    }
    for n in 0..=5 {
        let pairs: Vec<_> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        for mask in 0..1usize << pairs.len() {
            let mut graph = vec![BTreeSet::new(); n];
            for (bit, &(i, j)) in pairs.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    graph[i].insert(j);
                    graph[j].insert(i);
                }
            }
            let mut packed = vec![vec![0u64; (n + 63) / 64]; n];
            for (i, row) in graph.iter().enumerate() {
                for &j in row {
                    packed[i][j / 64] |= 1 << (j % 64);
                }
            }
            assert_eq!(
                minimum_degree_bitset(packed),
                reference(graph.clone()),
                "packed n={n} mask={mask}"
            );
            assert_eq!(
                minimum_degree(graph.clone()),
                reference(graph),
                "n={n} mask={mask}"
            );
        }
    }
}

#[test]
fn column_order_matches_explicit_graph_across_word_boundaries() {
    for n in [63, 64, 65, 127, 129, 257] {
        // Dense overlapping rows and disconnected columns exercise
        // fill, complete tails, and masks spanning multiple words.
        for stride in [2, 3, 7] {
            let rows: Vec<Vec<usize>> = (0..9)
                .map(|r| {
                    (0..n)
                        .filter(|&j| (j + r * 13) % (stride + r % 3) == 0 && j % 17 != 1)
                        .collect()
                })
                .collect();
            let mut graph = vec![BTreeSet::new(); n];
            for row in &rows {
                for &i in row {
                    for &j in row {
                        if i != j {
                            graph[i].insert(j);
                        }
                    }
                }
            }
            assert_eq!(
                column_order(&rows, n),
                minimum_degree(graph),
                "n={n} stride={stride}"
            );
        }
    }
}

#[test]
fn pivot_permutations_relabel_rows_from_earlier_fronts() {
    // Column 0 is a separate front with an update into the pivoted front.
    // A disconnected component also exercises reuse of the local row map.
    for (diagonal, coupling) in [(0.21, 1.), (0.2, 2.)] {
        let a = SparseMatrix::from_triplets(
            6,
            6,
            &[
                (0, 0, 5.),
                (1, 1, diagonal),
                (2, 2, 10.),
                (3, 3, 3.),
                (0, 1, 1.),
                (1, 0, 1.),
                (1, 2, coupling),
                (2, 1, coupling),
                (1, 3, 0.2),
                (3, 1, 0.2),
                (2, 3, 2.),
                (3, 2, 2.),
                (4, 4, 2.),
                (5, 5, -3.),
                (4, 5, 0.5),
                (5, 4, 0.5),
            ],
        )
        .unwrap();
        let symbolic = Symbolic::analyze(&a, (0..6).collect(), 3);
        assert_eq!(symbolic.fronts[0].end, 1);
        let factor = symbolic.factorize(&a, false).unwrap();
        assert_ne!(
            factor.order, symbolic.order,
            "fixture must interchange pivots"
        );
        let expected = [1., -2., 3., -4., 5., -6.];
        let rhs = a.mul(&expected);
        let result = factor.solve(&rhs).unwrap();
        for (x, e) in result.iter().zip(expected) {
            assert!((x - e).abs() < 1e-10, "{result:?}");
        }
    }
}

#[test]
fn two_by_two_and_pivot_interchange() {
    for values in [
        vec![0., 2., 1., 2., 0., 3., 1., 3., 4.],
        vec![0.01, 1., 0., 1., 10., 2., 0., 2., 3.],
    ] {
        let entries: Vec<_> = (0..3)
            .flat_map(|i| {
                let v = &values;
                (0..3).map(move |j| (i, j, v[3 * i + j]))
            })
            .collect();
        let a = SparseMatrix::from_triplets(3, 3, &entries).unwrap();
        let factor = Symbolic::analyze(&a, vec![0, 1, 2], 3)
            .factorize(&a, false)
            .unwrap();
        let rhs = a.mul(&[1., -2., 3.]);
        let x = factor.solve(&rhs).unwrap();
        for (v, w) in x.iter().zip([1., -2., 3.]) {
            assert!((v - w).abs() < 1e-10, "{x:?}");
        }
    }
}
