use baldes::enumeration::{Controls, Enumeration};
use baldes::pricing::Solver;
use baldes::vrptw::{Instance, Rounding};
fn r201_solver() -> Solver {
    let mut inst =
        Instance::parse(include_str!("../instances/R201.txt")).unwrap();
    inst.truncate(10).unwrap();
    let mut graph = inst.graph(8, Rounding::None).unwrap();
    for v in 0..12 {
        let ids = if v == 0 || v == 11 {
            vec![v]
        } else {
            let mut ids: Vec<_> = (1..=10).collect();
            ids.sort_by(|&a, &b| {
                inst.distance(v, a, Rounding::None)
                    .total_cmp(&inst.distance(v, b, Rounding::None))
                    .then(a.cmp(&b))
            });
            ids.truncate(8);
            ids
        };
        graph.set_ng(v, &ids).unwrap();
    }
    let duals = [
        0.,
        5.2927978298679506,
        21.252810251675285,
        12.21470139966371,
        41.970938164481396,
        20.003235324569687,
        18.197048992639537,
        21.1696994951823,
        24.31322948881612,
        42.82998162301914,
        39.680737132109016,
    ];
    for a in &mut graph.arcs {
        let pi = if a.head == 11 { 0. } else { duals[a.head] };
        a.reduced_cost = ((a.cost - pi) * 1e8 + 0.5).floor() / 1e8;
    }
    let mut solver = Solver::new(12, 2).unwrap();
    solver.graph = graph;
    solver
}
#[test]
fn native_r201_10_fixing_preserves_complete_elementary_universe() {
    let mut solver = r201_solver();
    solver.price_phase(0, 150, false).unwrap();
    solver.complete_labels(true).unwrap();
    let controls = Controls {
        cutoff: 2.2760040285411094,
        ..Default::default()
    };
    let engine = solver.engine.as_ref().unwrap();
    let plain = Enumeration::run(&solver.graph, &engine.grid, &controls, None).unwrap();
    let directional =
        Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |_, _| true).unwrap();
    let completed = Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |f, l| {
        engine.enumeration_completion(&solver.graph, f, l, controls.cutoff, true)
    })
    .unwrap();
    let fixed = solver.fix(controls.cutoff).unwrap();
    let engine = solver.engine.as_ref().unwrap();
    let fixed_plain = Enumeration::run(&solver.graph, &engine.grid, &controls, None).unwrap();
    let fixed_directional =
        Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |_, _| true).unwrap();
    solver.complete_labels(true).unwrap();
    let engine = solver.engine.as_ref().unwrap();
    let after_plain = Enumeration::run(&solver.graph, &engine.grid, &controls, None).unwrap();
    let after_directional =
        Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |_, _| true).unwrap();
    let after_completed =
        Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |f, l| {
            engine.enumeration_completion(&solver.graph, f, l, controls.cutoff, true)
        })
        .unwrap();
    let routes = solver.enumerate(controls.cutoff).unwrap();
    assert_eq!(fixed.eliminated_forward, 100);
    assert_eq!(fixed.eliminated_backward, 104);
    assert_eq!(fixed_plain.paths.len, 47);
    assert_eq!(fixed_directional.paths.len, 47);
    assert_eq!(after_plain.paths.len, 47);
    assert_eq!(after_directional.paths.len, 47);
    assert_eq!(after_completed.paths.len, 47);
    assert_eq!(plain.paths.len, 47);
    assert_eq!(directional.paths.len, 47);
    assert_eq!(completed.paths.len, 47);
    assert_eq!(routes.len(), 47);
}

#[cfg(feature = "clp")]
#[test]
fn r201_10_fixing_root_matches_native_objective_and_iterations() {
    use baldes::master::{
        column_generation::Status,
        root::{self, RootOptions, RootProblem},
    };
    let mut graph = r201_solver().graph;
    for arc in &mut graph.arcs {
        arc.reduced_cost = arc.cost;
    }
    let mut options = RootOptions::default();
    options.cg.multiplicity = 10.;
    let result = root::solve(RootProblem::set_partitioning(graph, 10.).unwrap(), options).unwrap();
    assert_eq!(result.master.status, Status::Optimal);
    assert!((result.master.lp - 246.92517970202414).abs() < 1e-9);
    assert_eq!(result.master.iterations, 24);
    assert_eq!(result.master.pricing_calls, 34);
    assert_eq!(result.master.columns, 58);
    let mut coverage = [0.; 10];
    for (&tag, &value) in result.master.column_tags.iter().zip(&result.values) {
        if value > 1e-6 {
            assert!(tag >= 0, "artificial column remains positive");
            for &v in
                &result.routes[tag as usize].nodes[1..result.routes[tag as usize].nodes.len() - 1]
            {
                coverage[v as usize - 1] += value;
            }
        }
    }
    assert!(coverage.iter().all(|&value| (value - 1f64).abs() < 1e-9));
}
