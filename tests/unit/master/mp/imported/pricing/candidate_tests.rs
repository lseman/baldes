use super::*;

fn bounds(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}

fn column(id: i32, cost: f64, entries: Vec<(i32, f64)>) -> Column {
    Column {
        id,
        cost,
        entries,
        bounds: bounds(0., f64::INFINITY),
        integer: false,
    }
}

fn session() -> MultiImport<2> {
    let definition = Definition {
        problem: Problem {
            rows: vec![],
            networks: vec![],
        },
        variables: [0, 17, 5]
            .into_iter()
            .map(|id| MasterVariable {
                id,
                cost: 0.,
                bounds: bounds(-100., 100.),
                integer: false,
                entries: vec![],
            })
            .collect(),
        subproblems: vec![
            Subproblem {
                id: 7,
                variables: vec![17, 0],
                multiplicity: bounds(0., 8.),
                columns: vec![column(2, 1., vec![(17, 0.5), (0, -0.25)])],
            },
            Subproblem {
                id: 9,
                variables: vec![0],
                multiplicity: bounds(0., f64::INFINITY),
                columns: vec![],
            },
            Subproblem {
                id: 42,
                variables: vec![5, 17],
                multiplicity: bounds(0., 9.),
                columns: vec![column(2, 2., vec![(5, 1.)])],
            },
        ],
        envelopes: BTreeMap::new(),
    };
    let root = lifecycle::rebuild(Rc::new(definition), None, Options::default()).unwrap();
    root.snapshot()
        .unwrap()
        .begin_imports(&[7, 9, 42])
        .unwrap()
        .with_variable_bounds(0, bounds(-2., 2.))
        .unwrap()
        .with_multiplicity(7, bounds(0., 5.))
        .unwrap()
        .with_column_bounds(42, 2, bounds(0., 3.))
        .unwrap()
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 7.,
            },
            &[],
            &[(0, 1.), (17, 2.)],
        )
        .unwrap()
        .with_multiplicity(7, bounds(1., 4.))
        .unwrap()
        .with_multiplicity(9, bounds(0., 6.))
        .unwrap()
}

// The former production path remains an independent compiler-based oracle.
// Inserting in an earlier group shifts later internal IDs; restrictions must
// continue to identify their original group/column rather than those IDs.
fn compiled_candidate(
    point: &ImportPricing<2>,
    position: usize,
    column: Column,
) -> Result<MasterVariable> {
    if column.bounds.lower != 0. {
        return Err(invalid(
            "pricing candidate requires zero quantity lower bound",
        ));
    }
    let old = &point.definition;
    let mut definition = Definition {
        problem: old.problem.clone(),
        variables: old.variables.clone(),
        subproblems: old.subproblems.clone(),
        envelopes: old.envelopes.clone(),
    };
    let group = definition.subproblems[position].id;
    let id = column.id;
    definition.envelopes.remove(&group);
    definition.subproblems[position].columns.push(column);
    let compiled = lifecycle::compile_definition(&definition, Some(&point.context))?;
    let id = compiled.registry.columns[&(group, id)];
    Ok(compiled.variables.into_iter().find(|v| v.id == id).unwrap())
}

#[test]
fn sparse_candidates_match_compilation_with_interleaved_restrictions_and_group_views() {
    let session = session();
    let mut rng = crate::rng::Xoroshiro128Plus::from_seed(0x5130_cafe);
    for mut point in [
        session.pricing(7, Options::default()).unwrap(),
        session.feasibility_pricing(7, Options::default()).unwrap(),
    ] {
        // Nonzero, mixed-sign stored duals exercise every row and preserve
        // arithmetic ordering, independent of which LP basis was selected.
        point.duals = (0..point.duals.len())
            .map(|i| (i as f64 - 3.5) / 8.)
            .collect();
        for group in [7, 9, 42] {
            let view = point.for_subproblem(group).unwrap();
            assert!(Rc::ptr_eq(&point.registry, &view.registry));
            for i in 0..128 {
                let entries = view
                    .subproblem()
                    .variables
                    .iter()
                    .rev()
                    .map(|&id| {
                        let value = (rng.uniform_below(65) as f64 - 32.) / 16.;
                        (id, if i % 5 == 0 { -0. } else { value })
                    })
                    .collect();
                let column = column(100 + i, i as f64 / 16. - 4., entries);
                let expected = compiled_candidate(&view, view.position, column.clone()).unwrap();
                let actual = view.candidate(column.clone()).unwrap();
                assert_eq!(actual.cost.to_bits(), expected.cost.to_bits());
                assert_eq!(actual.entries, expected.entries);
                let cost = if view.kind == ImportPricingKind::Objective {
                    expected.cost
                } else {
                    0.
                };
                let reduced = expected.entries.iter().fold(cost, |sum, &(row, a)| {
                    (-a).mul_add(view.duals[view.shared_offset + row], sum)
                });
                assert_eq!(
                    view.reduced_cost(column.clone()).unwrap().to_bits(),
                    reduced.to_bits()
                );
                let interval = view.reduced_cost_bounds(column).unwrap().unwrap();
                assert!(interval.lower <= reduced && interval.upper >= reduced);
            }
        }
    }
}

#[test]
fn sparse_candidates_preserve_validation_and_symbolic_envelope_queries() {
    let session = session()
        .with_envelope(
            7,
            ColumnEnvelope {
                cost_lower: 1.,
                coordinates: vec![(17, bounds(0., 1.)), (0, bounds(-1., 0.))],
            },
        )
        .unwrap();
    let point = session.pricing(7, Options::default()).unwrap();
    let mut invalid = vec![
        column(-1, 0., vec![]),
        column(2, 0., vec![]),
        column(100, f64::NAN, vec![]),
        column(100, f64::INFINITY, vec![]),
        column(100, 0., vec![(5, 1.)]), // Valid coordinate, wrong group.
        column(100, 0., vec![(99, 0.)]),
        column(100, 0., vec![(17, 0.), (17, -0.)]),
        column(100, 0., vec![(17, f64::NAN)]),
        column(100, 0., vec![(17, f64::INFINITY)]),
    ];
    for b in [bounds(1., 2.), bounds(0., -1.), bounds(0., f64::NAN)] {
        let mut c = column(100, 0., vec![]);
        c.bounds = b;
        invalid.push(c);
    }
    for c in invalid {
        assert!(compiled_candidate(&point, point.position, c.clone()).is_err());
        assert!(point.reduced_cost(c.clone()).is_err());
        assert!(point.reduced_cost_bounds(c).is_err());
    }
    // Pricing algebra deliberately permits vectors outside the envelope.
    for c in [column(100, 0., vec![]), column(100, -2., vec![(17, -3.)])] {
        let expected = compiled_candidate(&point, point.position, c.clone()).unwrap();
        assert_eq!(point.candidate(c).unwrap().entries, expected.entries);
    }
    assert!(point.costs().is_ok());
    assert!(point.envelope_reduced_cost_bound().unwrap().is_some());
}
