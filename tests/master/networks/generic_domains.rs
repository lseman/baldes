//! Shared external features, original fixing and counts versus exact fixtures.
use super::*;
use baldes::master::preprocessing::Bounds;
use multi_pricing::{
    imported::{self, Column, ColumnBatch, ImportOptions, ImportOracleReply, ImportTreeStatus},
    AssignmentTreeOptions, MasterVariable,
};
use serde_json::Value;
fn b(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/generic_oracle_domains.json")).unwrap()
}
fn row(v: &Value) -> (Row, Vec<(i32, f64)>) {
    let sense = match v["sense"].as_str().unwrap() {
        "=" => RowSense::Equal,
        "<=" => RowSense::LessEqual,
        ">=" => RowSense::GreaterEqual,
        _ => unreachable!(),
    };
    (
        Row {
            sense,
            rhs: number(&v["rhs"]),
        },
        v["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| (e[0].as_i64().unwrap() as i32, number(&e[1])))
            .collect(),
    )
}
fn columns(group: &Value, offset: i32) -> Vec<Column> {
    group["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, vector)| Column {
            id: id as i32,
            cost: 0.,
            bounds: b(0., f64::INFINITY),
            integer: true,
            entries: vector
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .filter_map(|(j, v)| {
                    let a = number(v);
                    (a != 0.).then_some((offset + j as i32, a))
                })
                .collect(),
        })
        .collect()
}
fn base(spec: &Value, case: &Value) -> imported::ImportedOutput<7> {
    let variables = (0..12)
        .map(|id| {
            let feature = id % 6;
            let group = id / 6;
            MasterVariable {
                id,
                cost: match feature {
                    3 => number(&case["counter_cost"][group as usize]),
                    4 => number(&case["feature_cost"]),
                    _ => 0.,
                },
                integer: feature != 5,
                bounds: if feature == 5 {
                    b(f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    b(0., if feature == 3 { 3. } else { 1. })
                },
                entries: if feature < 3 {
                    vec![(feature as usize, 1.)]
                } else {
                    vec![]
                },
            }
        })
        .collect();
    let groups = spec["groups"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(g, group)| imported::Subproblem {
            id: group["id"].as_i64().unwrap() as i32,
            variables: (0..6).map(|i| g as i32 * 6 + i).collect(),
            multiplicity: b(0., 3.),
            columns: columns(group, g as i32 * 6),
        })
        .collect();
    imported::solve(
        Problem::<7> {
            rows: (0..3)
                .map(|_| Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                })
                .collect(),
            networks: vec![],
        },
        variables,
        groups,
        options(),
    )
    .unwrap()
}
#[test]
fn generic_fixing_links_and_nonlinear_features_match_exact_domains_and_numerical_oracles() {
    let spec = fixture();
    for case in spec["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let parent = base(&spec, case);
        let mut node = parent.snapshot().unwrap();
        for g in 0..2 {
            node = node
                .with_multiplicity(
                    [7, 42][g],
                    b(
                        number(&case["multiplicity"][g][0]),
                        number(&case["multiplicity"][g][1]),
                    ),
                )
                .unwrap();
        }
        for restriction in case["rows"].as_array().unwrap() {
            let (r, entries) = row(restriction);
            node = if entries.len() == 1 && r.sense == RowSense::Equal {
                let value = r.rhs / entries[0].1;
                node.with_variable_bounds(entries[0].0, b(value, value))
                    .unwrap()
            } else {
                node.with_row(r, &[], &entries).unwrap()
            };
        }
        let lp = number(&case["expected"]["lp_value"]);
        let ip = number(&case["expected"]["integer_value"]);
        let solved = node.solve(options()).unwrap();
        optimal(&solved.output, lp);
        for integer_policy in [false, true] {
            let mut configured = options();
            configured.preprocessing.integer_domains = integer_policy;
            let checked = node.solve(configured).unwrap();
            assert_eq!(
                checked.output.master.status,
                Status::Optimal,
                "{name}/{integer_policy}"
            );
            assert!(
                checked.output.master.lp >= lp - 1e-7 && checked.output.master.lp <= ip + 1e-7,
                "{name}: {} outside [{lp},{ip}]",
                checked.output.master.lp
            );
        }
        let session = node.begin_imports(&[7, 42]).unwrap();
        let oracle = session
            .begin_tree()
            .run_with_oracles(
                &AssignmentTreeOptions {
                    node: options(),
                    ..Default::default()
                },
                &ImportOptions::default(),
                |point, _| {
                    let g = if point.subproblem().id == 7 { 0 } else { 1 };
                    let minimum = columns(&spec["groups"][g], g as i32 * 6)
                        .into_iter()
                        .map(|mut c| {
                            c.id += 1000;
                            point.reduced_cost(c).unwrap()
                        })
                        .fold(f64::INFINITY, f64::min);
                    Ok(ImportOracleReply {
                        batch: ColumnBatch {
                            columns: vec![],
                            complete: false,
                        },
                        minimum: Some(point.numerical_minimum(minimum).unwrap()),
                    })
                },
            )
            .unwrap();
        assert_eq!(
            oracle.status,
            ImportTreeStatus::Optimal,
            "{name}: {oracle:?}"
        );
        assert!(
            (oracle.incumbent - ip).abs() < 1e-7,
            "{name}: {} != {ip}",
            oracle.incumbent
        );
        assert!(!session.is_complete());
        #[cfg(feature = "highs")]
        {
            let mip = solved
                .output
                .snapshot
                .solve_enumerated_mip(&baldes::master::solvers::mip::MipOptions::default())
                .unwrap()
                .unwrap();
            assert!(
                (mip.mip.solution.unwrap().objective - ip).abs() < 1e-7,
                "{name}"
            );
        }
        // Original source definitions and parents remain reusable after both
        // journalled fixing and tree/MIP operations on their descendants.
        let parent_again = parent.snapshot().unwrap().solve(options()).unwrap();
        assert!(
            (parent_again.output.master.lp - parent.output.master.lp).abs() < 1e-7,
            "{name}"
        );
    }
}
