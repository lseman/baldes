use super::*;

#[test]
fn envelope_preserves_crossings_gaps_and_isolated_feasible_points() {
    let mut ws = Workspace {
        arena: Vec::new(),
        work: 0,
        limit: 10_000,
    };
    let piece = |lo, hi, a, b| Piece {
        lo,
        hi,
        a,
        b,
        p: 1.0,
        q: 0.0,
        incoming: 0.0,
        parent: None,
    };
    let frontier = ws
        .envelope(vec![
            piece(0.0, 4.0, 1.0, 0.0),
            piece(0.0, 4.0, -1.0, 4.0),
            piece(6.0, 6.0, 0.0, 1.0),
        ])
        .unwrap();
    for (energy, expected) in [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 1.0),
        (4.0, 0.0),
        (5.0, f64::INFINITY),
        (6.0, 1.0),
    ] {
        let actual = frontier
            .iter()
            .map(|&id| ws.arena[id])
            .filter(|p| p.lo <= energy && energy <= p.hi)
            .map(|p| p.value(energy))
            .fold(f64::INFINITY, f64::min);
        assert_eq!(actual, expected);
    }
}

#[test]
fn dominance_checks_interior_crossings_and_full_domain_coverage() {
    let piece = |lo, hi, a, b| Piece {
        lo,
        hi,
        a,
        b,
        p: 1.0,
        q: 0.0,
        incoming: 0.0,
        parent: None,
    };
    let mut ws = Workspace {
        arena: Vec::new(),
        work: 0,
        limit: 10000,
    };
    let triangle = ws
        .envelope(vec![piece(0.0, 4.0, 1.0, 0.0), piece(0.0, 4.0, -1.0, 4.0)])
        .unwrap();
    let constant = ws.envelope(vec![piece(0.0, 4.0, 0.0, 1.0)]).unwrap();
    assert!(!ws.dominates(&triangle, &constant).unwrap());
    assert!(!ws.dominates(&constant, &triangle).unwrap());
    let earlier = ws.envelope(vec![piece(0.0, 4.0, 0.0, 0.0)]).unwrap();
    assert!(ws.dominates(&earlier, &triangle).unwrap());
    let gap = ws
        .envelope(vec![piece(0.0, 1.0, 0.0, -1.0), piece(3.0, 4.0, 0.0, -1.0)])
        .unwrap();
    assert!(!ws.dominates(&gap, &earlier).unwrap());
    let isolated = ws.envelope(vec![piece(2.0, 2.0, 0.0, 0.0)]).unwrap();
    assert!(!ws.dominates(&gap, &isolated).unwrap());
    assert!(ws.dominates(&earlier, &isolated).unwrap());
}
