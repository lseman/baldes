//! Exhaustive original-unit checks, independent of LP and propagation code.
use super::*;
use baldes::master::{preprocessing::Bounds, solvers::mip::MipStatus};
use mp::{
    imported::{self, Column, Subproblem},
    MasterVariable,
};

fn bounds(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}
fn next(rng: &mut baldes::rng::Xoroshiro128Plus, modulus: i32) -> i32 {
    rng.uniform_below(modulus as u64) as i32
}
fn permits(sense: RowSense, lhs: i32, rhs: i32) -> bool {
    match sense {
        RowSense::Equal => lhs == rhs,
        RowSense::LessEqual => lhs <= rhs,
        RowSense::GreaterEqual => lhs >= rhs,
    }
}

#[test]
fn signed_import_fixing_and_propagation_preserve_every_exhaustive_integer_witness() {
    let mut seed = baldes::rng::Xoroshiro128Plus::from_seed(0x5035_2026_u64);
    let mut feasible_cases = 0;
    let mut infeasible_cases = 0;
    let mut checked_bounds = 0;
    for case in 0..96 {
        // y entries and row RHSs use integer half-units; objectives use quarters.
        let vectors: Vec<_> = (0..4)
            .map(|_| (next(&mut seed, 5) - 2, next(&mut seed, 7) - 3))
            .collect();
        let costs: Vec<_> = (0..4).map(|_| next(&mut seed, 9) - 4).collect();
        let x_lower = case % 3 - 2;
        let x_upper = 2 + case % 2;
        let y_lower = -3 + case % 3;
        let y_upper = 2 + case % 4;
        let y_integer = case % 2 == 0;
        let sense =
            [RowSense::Equal, RowSense::LessEqual, RowSense::GreaterEqual][case as usize % 3];
        let rhs = case % 7 - 3;
        let branch_sense = if case % 2 == 0 {
            RowSense::GreaterEqual
        } else {
            RowSense::LessEqual
        };
        let branch_rhs = if case % 2 == 0 { -1 } else { 1 };
        let group_lower = [case % 2, (case / 2) % 2];
        let mut witnesses = vec![];
        let mut best = None;
        // Four integer quantities in [0,2], filtered by original constraints.
        for encoded in 0..81 {
            let mut remaining = encoded;
            let mut q = [0; 4];
            for amount in &mut q {
                *amount = remaining % 3;
                remaining /= 3;
            }
            let multiplicity = [q[0] + q[1], q[2] + q[3]];
            if (0..2).any(|g| multiplicity[g] < group_lower[g] || multiplicity[g] > 2) {
                continue;
            }
            if case % 4 == 0 && q[0] != 1 || case % 4 == 1 && q[0] != 0 {
                continue;
            }
            if case % 3 == 0 && multiplicity[1] > 1 || case % 3 == 1 && multiplicity[1] != 1 {
                continue;
            }
            let x: i32 = (0..4).map(|j| vectors[j].0 * q[j]).sum();
            let y: i32 = (0..4).map(|j| vectors[j].1 * q[j]).sum();
            if x < x_lower || x > x_upper || y < y_lower || y > y_upper || y_integer && y % 2 != 0 {
                continue;
            }
            if case % 3 == 0 && y < 0 || case % 3 != 0 && y > 1 {
                continue;
            }
            if !permits(sense, 2 * x + y, rhs) || !permits(branch_sense, 2 * x - y, branch_rhs) {
                continue;
            }
            let objective: i32 = 4 * x - y + (0..4).map(|j| 4 * costs[j] * q[j]).sum::<i32>();
            best = Some(best.map_or(objective, |old: i32| old.min(objective)));
            witnesses.push((x as f64, y as f64 / 2.));
        }
        if best.is_some() {
            feasible_cases += 1;
        } else {
            infeasible_cases += 1;
        }
        for (enabled, integer_domains, work_limit) in [
            (false, false, 0),
            (true, false, 10_000),
            (true, true, 10_000),
            (true, true, 0),
        ] {
            let mut opts = options();
            opts.preprocessing.enabled = enabled;
            opts.preprocessing.integer_domains = integer_domains;
            opts.preprocessing.work_limit = work_limit;
            let problem = Problem::<2> {
                rows: vec![Row {
                    sense,
                    rhs: rhs as f64 / 2.,
                }],
                networks: vec![],
            };
            let variables = vec![
                MasterVariable {
                    id: 0,
                    cost: 1.,
                    bounds: bounds(x_lower as f64, x_upper as f64),
                    integer: true,
                    entries: vec![(0, 1.)],
                },
                MasterVariable {
                    id: 1,
                    cost: -0.5,
                    bounds: bounds(y_lower as f64 / 2., y_upper as f64 / 2.),
                    integer: y_integer,
                    entries: vec![(0, 1.)],
                },
            ];
            let groups = (0..2)
                .map(|g| Subproblem {
                    id: [7, 42][g],
                    variables: vec![0, 1],
                    multiplicity: bounds(group_lower[g] as f64, 2.),
                    columns: (0..2)
                        .map(|j| {
                            let k = 2 * g + j;
                            Column {
                                id: j as i32,
                                cost: costs[k] as f64,
                                bounds: bounds(0., 2.),
                                integer: true,
                                entries: vec![
                                    (0, vectors[k].0 as f64),
                                    (1, vectors[k].1 as f64 / 2.),
                                ],
                            }
                        })
                        .collect(),
                })
                .collect();
            let root = imported::solve(problem, variables, groups, opts.clone()).unwrap();
            let mut child = root.snapshot().unwrap();
            if case % 4 == 0 || case % 4 == 1 {
                let fixed = if case % 4 == 0 { 1. } else { 0. };
                child = child
                    .with_column_bounds(7, 0, bounds(fixed, fixed))
                    .unwrap();
            }
            if case % 3 != 2 {
                child = child
                    .with_multiplicity(42, bounds(if case % 3 == 1 { 1. } else { 0. }, 1.))
                    .unwrap();
            }
            child = child
                .with_variable_bounds(
                    1,
                    if case % 3 == 0 {
                        bounds(0., f64::INFINITY)
                    } else {
                        bounds(f64::NEG_INFINITY, 0.5)
                    },
                )
                .unwrap()
                .with_row(
                    Row {
                        sense: branch_sense,
                        rhs: branch_rhs as f64 / 2.,
                    },
                    &[],
                    &[(0, 1.), (1, -1.)],
                )
                .unwrap();
            let solved = child.solve(opts).unwrap();
            let report = solved.output.snapshot.preprocessing_report();
            assert!(
                report.propagation.work_used <= work_limit,
                "preprocessing exceeded budget in case {case}: {report:?}"
            );
            assert!(
                !report.propagation.infeasible || witnesses.is_empty(),
                "case {case}, policy {enabled}/{integer_domains}/{work_limit}: {report:?}"
            );
            for &(id, range) in &report.master_bounds {
                if id != 0 && id != 1 {
                    continue;
                }
                checked_bounds += 1;
                for &(x, y) in &witnesses {
                    let value = if id == 0 { x } else { y };
                    assert!(
                        range.lower <= value && value <= range.upper,
                        "case {case}, original {id}={value}, inferred {range:?}"
                    );
                }
            }
            let snapshot = solved.snapshot().unwrap();
            let mip = snapshot
                .node()
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap();
            match best {
                Some(value) => {
                    assert_eq!(
                        mip.mip.status,
                        MipStatus::Optimal,
                        "case {case}, policy {enabled}/{integer_domains}/{work_limit}: {mip:?}"
                    );
                    assert_eq!(
                        mip.mip.solution.as_ref().unwrap().objective,
                        value as f64 / 4.,
                        "case {case}"
                    );
                }
                None => assert_eq!(
                    mip.mip.status,
                    MipStatus::Infeasible,
                    "case {case}, policy {enabled}/{integer_domains}/{work_limit}: {mip:?}"
                ),
            }
        }
    }
    assert!(feasible_cases > 0 && infeasible_cases > 0 && checked_bounds > 0);
    eprintln!("original-domain oracle: {feasible_cases} feasible, {infeasible_cases} infeasible cases; {checked_bounds} reported original bounds checked");
}

#[test]
fn shared_presence_fixing_and_import_extension_match_exhaustive_original_route_counts() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
    use mp::PresenceCoefficients;
    let mut feasible = 0;
    let mut infeasible = 0;
    let mut checked = 0;
    for case in 0..24 {
        for mode in [Mode::PairCount, Mode::AnyPair] {
            let duplicates = 1 + (case / 2) % 2;
            let coefficient = if mode == Mode::PairCount {
                duplicates
            } else {
                1
            };
            let sense =
                [RowSense::LessEqual, RowSense::GreaterEqual, RowSense::Equal][case as usize % 3];
            let rhs = case % 4;
            let fixed = (case / 4) % 2;
            let force_y_one = case % 4 == 3 || case % 6 == 0;
            let required_import = (case / 3) % 2;
            let second_cap = 1 + case % 2;
            let new_cost = case % 4 - 2;
            let mut witnesses = vec![];
            let mut best = None;
            for encoded in 0..81 {
                let mut remainder = encoded;
                let mut routes = [0; 4];
                for q in &mut routes {
                    *q = remainder % 3;
                    remainder /= 3;
                }
                if routes[0] < fixed
                    || routes[0] + routes[1] > 2
                    || routes[2] + routes[3] > second_cap
                {
                    continue;
                }
                for (old, new) in [(0, 0), (1, 0), (0, 1)] {
                    let x = old + new;
                    if x < required_import || case % 4 == 0 && old != 0 || force_y_one && old != 1 {
                        continue;
                    }
                    if routes[0] + routes[1] + x < 1 || routes[2] + routes[3] + x < 1 {
                        continue;
                    }
                    if !permits(sense, coefficient * (routes[0] + routes[2]) + old, rhs) {
                        continue;
                    }
                    let objective = 4 * routes[0]
                        + 10 * routes[1]
                        + 2 * routes[2]
                        + 8 * routes[3]
                        + 201 * old
                        + 2 * new_cost * new;
                    best = Some(best.map_or(objective, |prior: i32| prior.min(objective)));
                    witnesses.push((routes, x, old));
                }
            }
            if best.is_some() {
                feasible += 1;
            } else {
                infeasible += 1;
            }
            for bound_mode in [BoundMode::Numerical, BoundMode::Certified] {
                for work_limit in [0, 10_000] {
                    let mut opts = options();
                    opts.cg.bound_mode = bound_mode;
                    opts.preprocessing.work_limit = work_limit;
                    opts.preprocessing.integer_domains = true;
                    let mut networks = vec![
                        network(99, &[(2., &[0]), (5., &[0])], 2),
                        network(77, &[(1., &[1]), (4., &[1])], 2),
                    ];
                    for network in &mut networks {
                        network.multiplicity.upper = 2.;
                    }
                    let root = imported::solve(
                        Problem::<2> {
                            rows: vec![
                                Row {
                                    sense: RowSense::GreaterEqual,
                                    rhs: 1.
                                };
                                2
                            ],
                            networks,
                        },
                        vec![
                            MasterVariable {
                                id: 0,
                                cost: 0.,
                                bounds: bounds(0., 1.),
                                integer: true,
                                entries: vec![(0, 1.), (1, 1.)],
                            },
                            MasterVariable {
                                id: 1,
                                cost: 0.5,
                                bounds: bounds(0., 1.),
                                integer: true,
                                entries: vec![],
                            },
                        ],
                        vec![Subproblem {
                            id: 7,
                            variables: vec![0, 1],
                            multiplicity: bounds(0., 1.),
                            columns: vec![Column {
                                id: 0,
                                cost: 100.,
                                bounds: bounds(0., 1.),
                                integer: true,
                                entries: vec![(0, 1.), (1, 1.)],
                            }],
                        }],
                        opts.clone(),
                    )
                    .unwrap();
                    let mut child = root.snapshot().unwrap();
                    if fixed == 1 {
                        let route = root
                            .output
                            .routes
                            .iter()
                            .position(|r| r.network == 99 && r.route.arcs == vec![0, 1])
                            .unwrap();
                        child = child.fix_route(route, 1.).unwrap();
                    }
                    let descriptors: Vec<_> = [99, 77]
                        .into_iter()
                        .map(|network| PresenceCoefficients {
                            network,
                            row: PresenceRow::new(2, vec![(0, 1); duplicates as usize], mode)
                                .unwrap(),
                            arc_events: vec![vec![0], vec![1], vec![0], vec![]],
                        })
                        .collect();
                    child = child
                        .with_presence_row(
                            Row {
                                sense,
                                rhs: rhs as f64,
                            },
                            &descriptors,
                            &[(1, 1.)],
                        )
                        .unwrap()
                        .with_multiplicity(7, bounds(required_import as f64, 1.))
                        .unwrap()
                        .with_routing_multiplicity(77, bounds(0., second_cap as f64))
                        .unwrap();
                    if case % 4 == 0 {
                        child = child.with_column_bounds(7, 0, bounds(0., 0.)).unwrap();
                    }
                    if force_y_one {
                        child = child.with_variable_bounds(1, bounds(1., 1.)).unwrap();
                    }
                    let solved = child
                        .solve_with_columns(
                            7,
                            vec![Column {
                                id: 1,
                                cost: new_cost as f64,
                                bounds: bounds(0., 1.),
                                integer: true,
                                entries: vec![(0, 1.)],
                            }],
                            opts.clone(),
                        )
                        .unwrap();
                    let report = solved.output.snapshot.preprocessing_report();
                    assert!(
                        report.propagation.work_used <= work_limit,
                        "preprocessing exceeded budget in case {case}: {report:?}"
                    );
                    assert!(
                        !report.propagation.infeasible || witnesses.is_empty(),
                        "case {case}, {mode:?}, {bound_mode:?}: {report:?}"
                    );
                    for activity in &report.networks {
                        let offset = if activity.network == 99 { 0 } else { 2 };
                        let fixed_here = if activity.network == 99 { fixed } else { 0 };
                        for &(routes, _, _) in &witnesses {
                            let a = routes[offset] - fixed_here;
                            let b = routes[offset + 1];
                            assert!(
                                activity.multiplicity.lower <= (a + b) as f64
                                    && (a + b) as f64 <= activity.multiplicity.upper
                            );
                            for (arc, value) in [a, a, b, b].into_iter().enumerate() {
                                assert!(activity.arcs[arc].lower <= value as f64 && value as f64 <= activity.arcs[arc].upper,
                                    "case {case}, {mode:?}, activity {activity:?}, witness {routes:?}");
                                checked += 1;
                            }
                        }
                    }
                    for &(id, range) in &report.master_bounds {
                        if id != 0 && id != 1 {
                            continue;
                        }
                        for &(_, x, y) in &witnesses {
                            let value = if id == 0 { x } else { y } as f64;
                            assert!(range.lower <= value && value <= range.upper);
                        }
                    }
                    let snapshot = solved.snapshot().unwrap();
                    let complete = snapshot.node().enumerate(&Default::default()).unwrap();
                    let mip = complete
                        .solve_enumerated_mip(&Default::default())
                        .unwrap()
                        .unwrap();
                    match best {
                        Some(value) => {
                            assert_eq!(mip.mip.status, MipStatus::Optimal, "case {case}, {mode:?}, {bound_mode:?}, budget {work_limit}: {mip:?}");
                            assert_eq!(
                                mip.mip.solution.as_ref().unwrap().objective,
                                value as f64 / 2.,
                                "case {case}"
                            );
                        }
                        None => assert_eq!(
                            mip.mip.status,
                            MipStatus::Infeasible,
                            "case {case}, {mode:?}, {bound_mode:?}: {mip:?}"
                        ),
                    }
                }
            }
        }
    }
    eprintln!(
        "coverage counters: feasible={feasible}, infeasible={infeasible}, arc_checks={checked}"
    );
    assert!(feasible > 0 && infeasible > 0 && checked > 0);
    eprintln!("shared-presence oracle: {feasible} feasible, {infeasible} infeasible domains; {checked} physical-arc bounds checked");
}
