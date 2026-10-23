//! Nonlinear routing restrictions must outlive changes to imported columns.
use super::*;
use baldes::cuts::{
    knapsack::{KnapsackKind, RouteLoadKnapsackCut},
    CutRegistry,
};
use baldes::master::preprocessing::Bounds;
use mp::{
    imported::{
        self, Column, ColumnBatch, ImportOptions, ImportPricingKind, ImportTreeStatus,
        ImportedSnapshot, MultiImport, Subproblem,
    },
    MasterVariable, RegisteredCutCoefficients, ResourceKnapsackCoefficients,
};

fn bounds(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}
fn column(id: i32, cost: f64, value: f64) -> Column {
    Column {
        id,
        cost,
        bounds: bounds(0., f64::INFINITY),
        integer: true,
        entries: vec![(0, value)],
    }
}
fn fixture(mode: BoundMode) -> imported::ImportedOutput<2> {
    let mut networks = vec![network(7, &[(5., &[0])], 2), network(99, &[(7., &[1])], 2)];
    for n in &mut networks {
        n.graph.arcs[0].consumption[0] = 0.5;
    }
    let mut opts = options();
    opts.cg.bound_mode = mode;
    imported::solve(
        Problem {
            rows: vec![
                Row {
                    sense: RowSense::Equal,
                    rhs: 1.
                };
                2
            ],
            networks,
        },
        vec![MasterVariable {
            id: 0,
            cost: 0.,
            bounds: bounds(0., 2.),
            integer: false,
            entries: vec![],
        }],
        vec![Subproblem {
            id: 42,
            variables: vec![0],
            multiplicity: bounds(0., 2.),
            columns: vec![column(0, 100., 1.)],
        }],
        opts,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Registered,
    Composite,
    Resource,
}
impl Kind {
    fn row(self) -> Row {
        Row {
            sense: RowSense::LessEqual,
            rhs: 0.,
        }
    }
    fn original(self) -> [(i32, f64); 1] {
        [(
            0,
            if matches!(self, Self::Composite) {
                -2.
            } else {
                -1.
            },
        )]
    }
    fn registered(self) -> Vec<RegisteredCutCoefficients> {
        let mut registry = CutRegistry::default();
        registry.add_strong_k_path_cut(3, &[1], 1, 0.).unwrap();
        let cut = registry.cuts.remove(0);
        let mut result = vec![];
        for id in [7, 99] {
            for _ in 0..if matches!(self, Self::Composite) {
                2
            } else {
                1
            } {
                result.push(RegisteredCutCoefficients {
                    network: id,
                    cut: cut.clone(),
                });
            }
        }
        result
    }
    fn resources(self) -> Vec<ResourceKnapsackCoefficients> {
        [(7, 8., 3), (99, 4., 2)]
            .into_iter()
            .map(|(network, scale, threshold)| ResourceKnapsackCoefficients {
                network,
                resource: 0,
                scale,
                cut: RouteLoadKnapsackCut {
                    kind: KnapsackKind::Steps,
                    numerator: 1,
                    denominator: 1,
                    rhs: 1,
                    lhs: 0.,
                    violation: 0.,
                    thresholds: vec![threshold],
                    coefficients: std::collections::BTreeMap::from([(threshold, 1)]),
                },
            })
            .collect()
    }
    fn snapshot(self, node: &ImportedSnapshot<2>) -> ImportedSnapshot<2> {
        match self {
            Self::Registered => {
                node.with_registered_cut_row(self.row(), &self.registered(), &self.original())
            }
            Self::Composite => {
                node.with_composite_cut_row(self.row(), &self.registered(), &self.original())
            }
            Self::Resource => {
                node.with_resource_knapsack_row(self.row(), &self.resources(), &self.original())
            }
        }
        .unwrap()
    }
    fn session(self, node: &MultiImport<2>) -> MultiImport<2> {
        match self {
            Self::Registered => {
                node.with_registered_cut_row(self.row(), &self.registered(), &self.original())
            }
            Self::Composite => {
                node.with_composite_cut_row(self.row(), &self.registered(), &self.original())
            }
            Self::Resource => {
                node.with_resource_knapsack_row(self.row(), &self.resources(), &self.original())
            }
        }
        .unwrap()
    }
}

#[test]
fn imported_registered_composite_and_resource_rows_replay_fixed_routes_and_new_vectors() {
    // Each original routing domain has one mandatory route. Its coefficient is
    // one (two for the repeated composite), so the row requires x=2. The source
    // supplies x=2 at cost 3, independently proving objective 5+7+3=15.
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for kind in [Kind::Registered, Kind::Composite, Kind::Resource] {
            let root = fixture(mode);
            optimal(&root.output, 12.);
            let parent = root.snapshot().unwrap();
            let route = root
                .output
                .routes
                .iter()
                .position(|r| r.network == 7)
                .unwrap();
            let mut opts = options();
            opts.cg.bound_mode = mode;
            if mode == BoundMode::Certified {
                // Registered/resource walk relaxations can drop a positive
                // nonlinear penalty. A correct RMP alone is not a tight proof.
                let weak = kind.snapshot(&parent).solve(opts.clone()).unwrap();
                assert!((weak.output.master.lp - 212.).abs() < 1e-7);
                assert!(weak.output.master.certified_bound < 212.);
                assert_ne!(weak.output.master.status, Status::Optimal);
                opts.enumeration = Some(Default::default());
            }
            optimal(
                &kind.snapshot(&parent).solve(opts.clone()).unwrap().output,
                212.,
            );
            for fix_first in [false, true] {
                let child = if fix_first {
                    kind.snapshot(&parent.fix_route(route, 1.).unwrap())
                } else {
                    kind.snapshot(&parent).fix_route(route, 1.).unwrap()
                };
                let expanded = child
                    .solve_with_columns(42, vec![column(1, 3., 2.)], opts.clone())
                    .unwrap();
                optimal(&expanded.output, 15.);
                assert_eq!(expanded.values.original, vec![(0, 2.)]);
                assert_eq!(expanded.output.fixed_routes.len(), 1);
                let streamed = expanded
                    .snapshot()
                    .unwrap()
                    .begin_import(42)
                    .unwrap()
                    .collect(&ImportOptions::default(), |_, _| {
                        Ok(ColumnBatch {
                            columns: vec![column(2, 2., 2.)],
                            complete: true,
                        })
                    })
                    .unwrap()
                    .session
                    .solve(opts.clone())
                    .unwrap();
                optimal(&streamed.output, 14.);
                assert_eq!(streamed.values.original, vec![(0, 2.)]);
                #[cfg(feature = "highs")]
                {
                    let snapshot = streamed.snapshot().unwrap();
                    let complete = snapshot.node().enumerate(&Default::default()).unwrap();
                    let result = complete
                        .solve_enumerated_mip(&Default::default())
                        .unwrap()
                        .unwrap();
                    assert_eq!(result.mip.solution.unwrap().objective, 14.);
                }
            }
            optimal(&parent.solve(opts).unwrap().output, 12.);
        }
    }
}

#[test]
fn open_import_trees_keep_registered_composite_and_resource_rows_during_phase_one() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for kind in [Kind::Registered, Kind::Composite, Kind::Resource] {
            let root = fixture(mode);
            let route = root
                .output
                .routes
                .iter()
                .position(|r| r.network == 7)
                .unwrap();
            let parent = root.snapshot().unwrap().fix_route(route, 1.).unwrap();
            let base = parent
                .begin_imports(&[42])
                .unwrap()
                .with_column_bounds(42, 0, bounds(0., 0.))
                .unwrap()
                .with_multiplicity(42, bounds(0., 2.))
                .unwrap();
            let session = kind.session(&base);
            let mut tree = mp::AssignmentTreeOptions {
                node: options(),
                ..Default::default()
            };
            tree.node.cg.bound_mode = mode;
            if mode == BoundMode::Certified {
                // Phase-I requests ignore enumeration while sources are open.
                // Once completion is declared, routing enumeration supplies the
                // complete-domain certificate needed for a certified optimum.
                tree.node.enumeration = Some(Default::default());
            }
            let mut calls = 0;
            let done = session
                .begin_tree()
                .run(&tree, &ImportOptions::default(), |point, _| {
                    calls += 1;
                    assert_eq!(point.kind(), ImportPricingKind::Feasibility);
                    assert!(point.reduced_cost(column(1, 3., 2.)).unwrap() < 0.);
                    Ok(ColumnBatch {
                        columns: vec![column(1, 3., 2.)],
                        complete: true,
                    })
                })
                .unwrap();
            assert_eq!(calls, 1, "{kind:?} {mode:?}");
            assert_eq!(done.status, ImportTreeStatus::Optimal, "{kind:?} {mode:?}");
            assert_eq!(done.bound, 15.);
            assert_eq!(done.incumbent, 15.);
            assert_eq!(done.tree.solution().unwrap().values.original, vec![(0, 2.)]);
            assert_eq!(done.tree.solution().unwrap().routes.len(), 2);
            assert!(!session.is_complete());
            optimal(&parent.solve(options()).unwrap().output, 12.);
        }
    }
}

#[test]
fn invalid_imported_kernel_descriptors_leave_parents_usable() {
    let root = fixture(BoundMode::Numerical);
    let parent = root.snapshot().unwrap();
    let session = parent.begin_imports(&[42]).unwrap();
    let duplicate = Kind::Composite.registered();
    assert!(session
        .with_registered_cut_row(Kind::Registered.row(), &duplicate, &[(0, -1.)])
        .is_err());
    assert!(parent
        .with_registered_cut_row(Kind::Registered.row(), &duplicate, &[(0, -1.)])
        .is_err());
    let mut unknown = Kind::Registered.registered();
    unknown[0].network = 123;
    assert!(session
        .with_composite_cut_row(Kind::Composite.row(), &unknown, &[(0, -2.)])
        .is_err());
    assert!(session
        .with_registered_cut_row(
            Kind::Registered.row(),
            &Kind::Registered.registered(),
            &[(123, 1.)]
        )
        .is_err());
    let mut resource = Kind::Resource.resources();
    resource[0].scale = f64::NAN;
    assert!(session
        .with_resource_knapsack_row(Kind::Resource.row(), &resource, &[(0, -1.)])
        .is_err());
    assert!(parent
        .with_resource_knapsack_row(Kind::Resource.row(), &resource, &[(0, -1.)])
        .is_err());
    assert!(session
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.
            },
            &Kind::Resource.resources(),
            &[(0, -1.)]
        )
        .is_err());
    optimal(&parent.solve(options()).unwrap().output, 12.);
}
