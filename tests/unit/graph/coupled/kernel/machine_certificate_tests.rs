use super::*;
use crate::graph::coupled::descriptor::{
    EvaluationPolicy, FrontierBudget, FrontierOutcome, SlopeFront, SourceReplayWitness,
};

#[test]
fn software_ieee_operations_match_independent_exact_rational_rounding_cells() {
    // Produced independently with Python Fraction and verified nearest/even
    // rounding cells, including signed zeros, subnormals, ties and overflow.
    let cases: Vec<(String, u64, u64, Option<u64>)> =
        serde_json::from_str(include_str!("fixtures/machine_ieee.json")).unwrap();
    assert_eq!(cases.len(), 2720);
    for (operation, a, b, expected) in cases {
        let (a, b) = (f64::from_bits(a), f64::from_bits(b));
        let actual = match operation.as_str() {
            "add" => ieee_add(a, b),
            "sub" => ieee_add(a, f64::from_bits(b.to_bits() ^ SIGN)),
            "mul" => ieee_multiply(a, b),
            "div" => ieee_divide(a, b),
            _ => unreachable!(),
        };
        assert_eq!(
            actual.map(f64::to_bits),
            expected,
            "{operation} {a:?} {b:?}"
        );
    }
}

#[test]
fn software_ieee_rounding_preserves_halfway_even_subnormal_and_overflow_boundaries() {
    let half_ulp = 2f64.powi(-53);
    assert_eq!(ieee_add(1., half_ulp).unwrap().to_bits(), 1f64.to_bits());
    assert_eq!(
        ieee_add(f64::from_bits(1f64.to_bits() + 1), half_ulp)
            .unwrap()
            .to_bits(),
        1f64.to_bits() + 2
    );
    assert_eq!(
        ieee_add(1., -2f64.powi(-54)).unwrap().to_bits(),
        1f64.to_bits()
    );
    let tiny = f64::from_bits(1);
    assert_eq!(ieee_multiply(tiny, 0.5).unwrap().to_bits(), 0);
    assert_eq!(ieee_multiply(-tiny, 0.5).unwrap().to_bits(), SIGN);
    assert_eq!(ieee_multiply(f64::from_bits(3), 0.5).unwrap().to_bits(), 2);
    assert_eq!(ieee_divide(tiny, 2.).unwrap().to_bits(), 0);
    assert_eq!(ieee_divide(f64::from_bits(3), 2.).unwrap().to_bits(), 2);
    assert_eq!(ieee_divide(1., 3.).unwrap().to_bits(), 0x3fd5555555555555);
    assert_eq!(ieee_add(f64::MAX, 2f64.powi(969)).unwrap(), f64::MAX);
    assert!(ieee_add(f64::MAX, 2f64.powi(970)).is_none());
    assert_eq!(
        ieee_multiply(f64::MIN_POSITIVE, 0.5).unwrap().to_bits(),
        1u64 << 51
    );
}

#[derive(serde::Deserialize)]
struct NativeCase {
    left: Vec<(f64, f64)>,
    right: Vec<(f64, f64)>,
    pairs: Vec<(u64, u64)>,
    minimum: u64,
}
#[derive(serde::Deserialize)]
struct NativeCases {
    binary_sha256: String,
    elf_address: String,
    cases: Vec<NativeCase>,
}

#[test]
fn independent_and_production_native_sum_match_all_retained_binary_fixtures() {
    let cases: NativeCases =
        serde_json::from_str(include_str!("fixtures/native_sum_67.json")).unwrap();
    assert_eq!(
        cases.binary_sha256,
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(cases.elf_address, "0x71e840");
    assert_eq!(cases.cases.len(), 67);
    for case in cases.cases {
        let mut work = 1_000_000;
        let mut m = Machine::new(&mut work, None);
        let left = vec![prepare(&case.left, &mut m).unwrap()];
        let right = vec![prepare(&case.right, &mut m).unwrap()];
        let checked = add(
            &left,
            &right,
            FrontOperationPolicy::NativeFrontAlgebra,
            &mut m,
        )
        .unwrap();
        assert_eq!(checked.len(), 1);
        assert_eq!(
            checked[0]
                .iter()
                .map(|&(v, x)| (v.to_bits(), x.to_bits()))
                .collect::<Vec<_>>(),
            case.pairs
        );
        assert_eq!(
            minimum(&checked, &mut m).unwrap().unwrap().to_bits(),
            case.minimum
        );

        let left = PreparedSlopeFunction::from_prepared_pairs(case.left).unwrap();
        let right = PreparedSlopeFunction::from_prepared_pairs(case.right).unwrap();
        let FrontierOutcome::Complete(actual) = SlopeFront::from_prepared(left).add(
            &SlopeFront::from_prepared(right),
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ) else {
            panic!("retained native addition must complete")
        };
        assert_eq!(
            actual.components()[0]
                .pairs()
                .iter()
                .map(|&(v, x)| (v.to_bits(), x.to_bits()))
                .collect::<Vec<_>>(),
            case.pairs
        );
        let FrontierOutcome::Complete(actual) = actual.minimum(FrontierBudget::default()) else {
            panic!("minimum")
        };
        assert_eq!(actual.value.to_bits(), case.minimum);
    }
}

#[derive(serde::Deserialize)]
struct RawSpec {
    initial: [f64; 2],
    bounds: [f64; 2],
    increments: Vec<f64>,
    functions: Vec<Vec<(f64, f64)>>,
    #[serde(default = "one")]
    scale: f64,
}
fn one() -> f64 {
    1.
}
#[derive(serde::Deserialize)]
struct RawCase {
    name: String,
    spec: RawSpec,
    status: String,
    value: Option<u64>,
}

#[test]
fn general_native_raw_certificates_match_independent_nonlattice_machine_programs() {
    let cases: Vec<RawCase> =
        serde_json::from_str(include_str!("fixtures/machine_raw_102.json")).unwrap();
    assert_eq!(cases.len(), 102);
    for case in cases {
        let spec = case.spec;
        let mut graph = Graph::<2>::new(spec.functions.len() + 1, 0).unwrap();
        for index in 0..spec.functions.len() {
            graph.add_arc(index, index + 1, 0.).unwrap();
        }
        let functions = spec
            .functions
            .into_iter()
            .map(|p| PreparedSlopeFunction::from_prepared_pairs(p).unwrap())
            .collect();
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(spec.initial[0], spec.initial[1]).unwrap(),
                bounds: Window::new(spec.bounds[0], spec.bounds[1]).unwrap(),
                increments: spec.increments,
                functions,
                policy: FrontOperationPolicy::NativeFrontAlgebra,
            }])
            .unwrap();
        if spec.scale == 0. {
            graph.suppress_coupled_costs();
        }
        // Fixtures describe the explicitly selected MACHINE feasibility
        // model. Its scope cannot silently replace a default source witness.
        graph.set_native_front_program_domain(true).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.has_machine_front_contracts());
        let path: Vec<_> = (0..graph.arcs.len()).collect();
        let actual = kernel.certify(&path);
        match case.status.as_str() {
            "complete" => {
                let (true, bounds) = actual.unwrap() else {
                    panic!("{} unexpectedly empty", case.name)
                };
                let expected = f64::from_bits(case.value.unwrap());
                assert_eq!(bounds, [expected, expected], "{}", case.name);
                assert_eq!(
                    kernel.replay(&path).unwrap(),
                    Some(expected),
                    "{}",
                    case.name
                );
            }
            "empty" => assert_eq!(actual, Some((false, [0.; 2])), "{}", case.name),
            "unknown" => assert!(actual.is_none(), "{}", case.name),
            other => panic!("unknown fixture status {other}"),
        }
    }
}

#[test]
fn machine_feasibility_requires_explicit_model_choice_and_survives_cloning() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let definition = CoupledResource::SlopeIntervalCost {
        initial: Window::new(0.1, 0.1).unwrap(),
        bounds: Window::new(0., 1.).unwrap(),
        increments: vec![0.2],
        functions: vec![PreparedSlopeFunction::constant_point(3., 0.30000000000000004).unwrap()],
        policy: FrontOperationPolicy::NativeFrontAlgebra,
    };
    graph.set_coupled_resources(&[definition.clone()]).unwrap();
    assert!(!graph.native_front_program_domain());
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .certify_front_program(&[usize::MAX], &mut 10000, None)
        .is_none());
    // The machine translation rounds0.1+0.2 to the source point domain.
    // Outward original-domain replay straddles that singleton, so borrowing
    // machine feasibility would weaken the default proof contract.
    assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), None);
    let alternate = graph
        .coupled_kernel()
        .unwrap()
        .certify_front_program(&[0], &mut 10000, None)
        .unwrap();
    assert!(alternate.feasible);
    assert_eq!(alternate.objective, [3., 3.]);
    assert!(!graph.native_front_program_domain());

    graph.set_native_front_program_domain(true).unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0]),
        Some((true, [3., 3.]))
    );
    let mut child = graph.clone();
    child.suppress_coupled_costs();
    assert!(child.native_front_program_domain());
    assert_eq!(
        child.coupled_kernel().unwrap().certify(&[0]),
        Some((true, [0., 0.]))
    );
    child.set_native_front_program_domain(false).unwrap();
    assert!(graph.native_front_program_domain());
    assert!(!child.native_front_program_domain());
    graph.set_coupled_resources(&[definition]).unwrap();
    assert!(!graph.native_front_program_domain());
}

#[test]
fn computed_front_emptiness_cannot_prove_default_source_infeasibility() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 0.5).unwrap(),
            bounds: Window::new(0., 2.).unwrap(),
            increments: vec![0.],
            functions: vec![
                PreparedSlopeFunction::from_prepared_pairs(vec![(1., 1.), (0.3, 2.)]).unwrap(),
            ],
            policy: FrontOperationPolicy::NativeFrontAlgebra,
        }])
        .unwrap();
    assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), None);
    graph.set_native_front_program_domain(true).unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0]),
        Some((false, [0.; 2]))
    );
}

#[test]
fn machine_component_points_do_not_turn_rounded_aggregate_costs_into_exact_points() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let definitions: Vec<_> = [0.1, 0.2]
        .into_iter()
        .map(|value| CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.],
            functions: vec![PreparedSlopeFunction::from_prepared_pairs(vec![
                (value, 0.),
                (0., 1.),
            ])
            .unwrap()],
            policy: FrontOperationPolicy::NativeFrontAlgebra,
        })
        .collect();
    graph.set_coupled_resources(&definitions).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let (true, cost) = kernel.certify(&[0]).unwrap() else {
        panic!("source witness exists")
    };
    assert!(cost[0] < cost[1]);
    let rounded = kernel.replay(&[0]).unwrap().unwrap();
    assert!(cost[0] <= rounded && rounded <= cost[1]);
}

#[test]
fn source_objective_walk_relaxation_never_overbounds_native_machine_costs() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra,
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(0., 2.).unwrap(),
                bounds: Window::new(0., 2.).unwrap(),
                increments: vec![0.; 2],
                functions: vec![
                    PreparedSlopeFunction::from_prepared_pairs(vec![
                        (2.1, 0.),
                        (-0.2, 0.7),
                        (-0.5, 2.),
                    ])
                    .unwrap(),
                    PreparedSlopeFunction::from_prepared_pairs(vec![
                        (0.2, 0.),
                        (-0.4, 0.7),
                        (-0.1, 2.),
                    ])
                    .unwrap(),
                ],
                policy,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.has_machine_cost_contracts());
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        assert_eq!(
            kernel.objective_walk_lower_bound(&graph, 2, &mut 10000, deadline),
            None
        );
        if policy == FrontOperationPolicy::NativeFrontAlgebra {
            assert!(kernel.certify(&[0, 1]).unwrap().0);
        }
    }
}

#[test]
fn machine_scope_does_not_claim_a_source_control_minimum() {
    let functions: Vec<_> = [
        vec![(1073741824., 0.), (1., 1.)],
        vec![(-1073741824., 0.), (0., 1.)],
        vec![(0., 0.), (-1., 1.)],
    ]
    .into_iter()
    .map(|p| PreparedSlopeFunction::from_prepared_pairs(p).unwrap())
    .collect();
    let mut work = 10000;
    assert_eq!(
        raw(
            Window::new(0., 1.).unwrap(),
            Window::new(0., 1.).unwrap(),
            &[0.; 3],
            &functions,
            &[0, 1, 2],
            1.,
            &mut work,
            None
        ),
        Some((true, [0.; 2]))
    );
    let x = 2f64.powi(-12) + 2f64.powi(-24);
    let source = SourceReplayWitness::evaluate_shared_argument(
        &functions,
        x,
        EvaluationPolicy::NativeTolerance,
    )
    .unwrap()
    .unwrap();
    assert_eq!(source.value(), -2f64.powi(-24));
}

#[test]
fn machine_checker_rejects_invalid_topology_overflow_and_exhausted_proofs() {
    let mut work = 100;
    let mut m = Machine::new(&mut work, None);
    assert!(prepare(&[(0., 0.), (1., 0.)], &mut m).is_none());
    assert!(prepare(&[(f64::MAX, 0.), (f64::MAX, 2.)], &mut m).is_none());
    assert!(m.mul(f64::MAX, 2.).is_none());
    let mut none = 0;
    assert!(Machine::new(&mut none, None).add(1., 2.).is_none());
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap();
    let mut remaining = 100;
    assert!(Machine::new(&mut remaining, Some(expired))
        .mul(1., 2.)
        .is_none());
    assert_eq!(remaining, 100);
}
