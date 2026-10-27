use super::*;
use crate::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, PreparedSlopeFunction, PreparedSlopeRelationArc, SlopeFront,
};
use crate::graph::Window;

fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
}
fn affine(y: f64, lo: f64, s: f64, hi: f64) -> PreparedSlopeFunction {
    function(&[(y, lo), (s, hi)])
}
fn descriptor(
    initial: PreparedSlopeFunction,
    arcs: Vec<SlopeRelationArc>,
    normalization: PairedCostNormalization,
) -> CoupledResource {
    let (lo, hi) = initial.domain();
    CoupledResource::SlopePaired {
        initial,
        initial_base: Window::new(lo, hi).unwrap(),
        arcs,
        mode: PairedResourceMode::CostUnbounded,
        normalization,
        policy: NATIVE,
    }
}
fn machine_value(
    definition: &CoupledResource,
    path: &[usize],
    scale: f64,
) -> Option<(bool, [f64; 2])> {
    certify(definition, path, scale, &mut 1_000_000, None)
}
fn production_value(
    definition: &CoupledResource,
    path: &[usize],
    scale: f64,
) -> Option<(bool, [f64; 2])> {
    let CoupledResource::SlopePaired {
        initial,
        initial_base,
        arcs,
        mode,
        normalization,
        policy,
    } = definition
    else {
        panic!()
    };
    let front = SlopeFront::from_prepared(initial.clone());
    let mut front = match front.restrict(
        initial_base.lower,
        initial_base.upper,
        *policy,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(f) => f,
        FrontierOutcome::Empty => return Some((false, [0.; 2])),
        _ => return None,
    };
    let baseline = match front.minimum(FrontierBudget::default()) {
        FrontierOutcome::Complete(x) => x.value,
        _ => return None,
    };
    for &i in path {
        front = match PreparedSlopeRelationArc::new(&arcs[i], *policy)
            .unwrap()
            .extend(&front, FrontierBudget::default())
        {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Some((false, [0.; 2])),
            _ => return None,
        };
    }
    if path.is_empty() || *mode == PairedResourceMode::Feasibility || scale == 0. {
        return Some((true, [0.; 2]));
    }
    match front.minimum(FrontierBudget::default()) {
        FrontierOutcome::Complete(x) => {
            let c = normalization.charge(baseline, x.value, scale).unwrap();
            Some((true, [c, c]))
        }
        _ => None,
    }
}

#[test]
fn general_machine_paired_programs_cover_nonconstant_nonunit_and_signed_waiting() {
    let mut complete = 0;
    for normalization in [
        PairedCostNormalization::AbsoluteFinalValue,
        PairedCostNormalization::FinalMinusInitialMinimum,
    ] {
        for waiting in [-2, -1, 0, 1, 2] {
            for scale in [0., 0.25, 1., 2.] {
                for arrival_slope in [1., 1.5, 2.] {
                    let initial = function(&[(1., 0.), (-0.5, 2.), (0.75, 4.)]);
                    let arc = SlopeRelationArc {
                        arrival: affine(1., 0., arrival_slope, 4.),
                        increment: function(&[(0.25, 0.), (0.125, 2.), (-0.25, 4.)]),
                        lower: affine(-20., 0., 0., 12.),
                        upper: affine(100., 0., 0., 12.),
                        waiting,
                    };
                    let def = descriptor(initial, vec![arc], normalization);
                    let expected = production_value(&def, &[0], scale);
                    let proof = machine_value(&def, &[0], scale);
                    assert_eq!(
                        proof, expected,
                        "waiting={waiting} slope={arrival_slope} scale={scale}"
                    );
                    complete += usize::from(proof.is_some());
                }
            }
        }
    }
    assert!(complete >= 100, "complete={complete}");
}

#[test]
fn machine_paired_disconnected_output_is_verified_without_source_lattice_assumptions() {
    let def = descriptor(
        function(&[(0., 0.), (1., 2.), (-1., 4.)]),
        vec![SlopeRelationArc {
            arrival: affine(1., 0., 1., 5.),
            increment: affine(0., 0., 0., 5.),
            lower: affine(-10., 1., 0., 5.),
            upper: affine(0., 1., 0., 5.),
            waiting: 1,
        }],
        PairedCostNormalization::AbsoluteFinalValue,
    );
    assert_eq!(machine_value(&def, &[0], 1.), Some((true, [0., 0.])));
    assert_eq!(
        machine_value(&def, &[0], 1.),
        production_value(&def, &[0], 1.)
    );
}

#[test]
fn machine_paired_bounds_and_multiple_arcs_keep_literal_program_order() {
    for waiting in [-1, 0, 1] {
        let a = SlopeRelationArc {
            arrival: affine(1., 0., 1.5, 4.),
            increment: affine(1., 0., 0.5, 4.),
            lower: affine(-2., 0., 0.25, 12.),
            upper: affine(8., 0., -0.25, 12.),
            waiting,
        };
        let b = SlopeRelationArc {
            arrival: affine(1., 0., 1.25, 12.),
            increment: affine(-0.25, 0., 0.125, 12.),
            lower: affine(-2., 0., 0., 18.),
            upper: affine(20., 0., 0., 18.),
            waiting: -waiting,
        };
        let def = descriptor(
            affine(0.5, 0., 0.25, 4.),
            vec![a, b],
            PairedCostNormalization::FinalMinusInitialMinimum,
        );
        assert_eq!(
            machine_value(&def, &[0, 1], 0.75),
            production_value(&def, &[0, 1], 0.75)
        );
        assert_eq!(certify(&def, &[0], 1., &mut 0, None), None);
        assert_eq!(
            certify(&def, &[0], 1., &mut 100000, Some(Instant::now())),
            None
        );
    }
}

#[test]
fn machine_paired_scope_never_certifies_the_strict_source_contract() {
    let mut def = descriptor(
        affine(0., 0., 0.5, 2.),
        vec![],
        PairedCostNormalization::AbsoluteFinalValue,
    );
    assert_eq!(machine_value(&def, &[], 1.), Some((true, [0., 0.])));
    let CoupledResource::SlopePaired { policy, .. } = &mut def else {
        unreachable!()
    };
    *policy = STRICT;
    assert_eq!(machine_value(&def, &[], 1.), None);
}

#[test]
fn paired_machine_secants_match_retained_native_composition_bits() {
    // Actual original helper output, selected from the pinned binary corpus.
    // Only previously Complete inputs are retained; no whole-solver claim.
    for (index, line) in
        include_str!("../../../../fixtures/coupled/paired-machine-native-composition.txt")
            .lines()
            .enumerate()
    {
        let (args, expected) = line.split_once('|').unwrap();
        let x: Vec<f64> = args
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let outer = vec![(x[0], 0.), (x[1], 0.7), (x[2], 2.)];
        let inner = if x[8] == 0. {
            vec![(x[3], 0.), (x[4], 2.)]
        } else {
            vec![(x[3], 0.), (x[4], 0.9), (x[5], 2.)]
        };
        let expected: Vec<_> = expected
            .split_whitespace()
            .skip(2)
            .map(|pair| {
                let (a, b) = pair.split_once(':').unwrap();
                (
                    u64::from_str_radix(a, 16).unwrap(),
                    u64::from_str_radix(b, 16).unwrap(),
                )
            })
            .collect();
        let mut work = 100000;
        let mut machine = Machine::new(&mut work, None);
        let actual = compose(&outer, &inner, x[6], x[7], &mut machine)
            .unwrap()
            .unwrap();
        let actual: Vec<_> = actual
            .iter()
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
            .collect();
        assert_eq!(actual, expected, "native corpus case {index}");
    }
}

#[test]
fn original_domain_machine_pairs_require_an_explicit_model_opt_in() {
    use crate::graph::Graph;
    let def = descriptor(
        affine(1., 0., 0.25, 4.),
        vec![SlopeRelationArc {
            arrival: affine(1., 0., 1.5, 4.),
            increment: affine(0.25, 0., 0.125, 4.),
            lower: affine(-20., 0., 0., 12.),
            upper: affine(100., 0., 0., 12.),
            waiting: 0,
        }],
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let expected = machine_value(&def, &[0], 1.).unwrap();
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.set_coupled_resources(&[def]).unwrap();
    assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), None);
    graph.set_front_program_domain(true).unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0]),
        Some(expected)
    );
    graph.set_front_program_domain(false).unwrap();
    assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), None);
}

#[test]
fn near_slope_machine_pairs_do_not_claim_the_source_control_minimum() {
    // The source relation with identity arrival, no increment and free waiting
    // has minimum -step. Native secant merging retains the preceding zero
    // slope, so the declared MACHINE transition instead has minimum zero.
    let step = 0.00005;
    let def = descriptor(
        function(&[(0., 0.), (0., 1.), (-step, 2.)]),
        vec![SlopeRelationArc {
            arrival: affine(0., 0., 1., 2.),
            increment: affine(0., 0., 0., 2.),
            lower: affine(-10., 0., 0., 2.),
            upper: affine(10., 0., 0., 2.),
            waiting: 0,
        }],
        PairedCostNormalization::AbsoluteFinalValue,
    );
    assert_eq!(production_value(&def, &[0], 1.), Some((true, [0., 0.])));
    assert_eq!(machine_value(&def, &[0], 1.), Some((true, [0., 0.])));
    assert_ne!(0., -step);
    let mut source = def.clone();
    let CoupledResource::SlopePaired { policy, .. } = &mut source else {
        unreachable!()
    };
    *policy = STRICT;
    assert_eq!(
        production_value(&source, &[0], 1.),
        Some((true, [-step, -step]))
    );
}

#[test]
fn machine_composition_retains_short_remainder_anchors_from_reference_controls() {
    let fixture =
        include_str!("../../../../fixtures/coupled/metasolver-composition-boundaries.txt");
    let mut controls = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let (input, reference) = line.split_once('|').unwrap();
        let mut input = input.split_whitespace();
        let name = input.next().unwrap();
        let outer_count: usize = input.next().unwrap().parse().unwrap();
        let inner_count: usize = input.next().unwrap().parse().unwrap();
        let lo: f64 = input.next().unwrap().parse().unwrap();
        let hi: f64 = input.next().unwrap().parse().unwrap();
        assert_eq!(input.next(), Some("0"));
        let mut pairs = |count| {
            (0..=count)
                .map(|_| {
                    (
                        input.next().unwrap().parse::<f64>().unwrap(),
                        input.next().unwrap().parse::<f64>().unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let outer = pairs(outer_count);
        let inner = pairs(inner_count);
        let mut reference = reference.split_whitespace();
        assert_eq!(reference.next(), Some(name));
        assert_eq!(reference.next(), Some("1"));
        let count: usize = reference.next().unwrap().parse().unwrap();
        assert_eq!(reference.next(), Some("0"));
        let expected: Vec<_> = (0..=count)
            .map(|_| {
                (
                    u64::from_str_radix(reference.next().unwrap(), 16).unwrap(),
                    u64::from_str_radix(reference.next().unwrap(), 16).unwrap(),
                )
            })
            .collect();
        let mut work = 100_000;
        let actual = compose(&outer, &inner, lo, hi, &mut Machine::new(&mut work, None))
            .unwrap_or_else(|| panic!("unknown reference control {name}"))
            .unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            expected,
            "{name}"
        );
        controls += 1;
    }
    assert_eq!(controls, 74);
}

#[test]
fn short_remainder_paired_certificates_preserve_model_scope_and_work_limits() {
    let scale = 1.00010001;
    for end in [1., 1.5] {
        // The inverse arrival crosses the initial front's knot just before
        // an inner endpoint. Offset arrival by two so every arrival is later
        // than departure, satisfying the physical paired-arc precondition.
        let arrival = function(&[(2., 0.), (1. / scale, scale), (1., scale + end)]);
        let def = descriptor(
            function(&[(0., 0.), (1., 1.), (2., 4.)]),
            vec![SlopeRelationArc {
                arrival,
                increment: affine(0., 0., 0., 4.),
                lower: affine(-100., 2., 0., end + 3.),
                upper: affine(100., 2., 0., end + 3.),
                waiting: 10,
            }],
            PairedCostNormalization::AbsoluteFinalValue,
        );
        assert_eq!(production_value(&def, &[0], 1.), Some((true, [0., 0.])));
        assert_eq!(machine_value(&def, &[0], 1.), Some((true, [0., 0.])));
        let mut graph = crate::graph::Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.set_coupled_resources(&[def.clone()]).unwrap();
        let source_result = graph.coupled_kernel().unwrap().certify(&[0]);
        let alternate = graph
            .coupled_kernel()
            .unwrap()
            .certify_front_program(&[0], &mut 100_000, None)
            .unwrap();
        assert!(alternate.feasible);
        assert_eq!(alternate.objective, [0., 0.]);
        assert!(!graph.front_program_domain());
        assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), source_result);
        graph.set_front_program_domain(true).unwrap();
        assert_eq!(
            graph.coupled_kernel().unwrap().certify(&[0]),
            Some((true, [0., 0.]))
        );
        assert_eq!(certify(&def, &[0], 1., &mut 0, None), None);
        assert_eq!(
            certify(&def, &[0], 1., &mut 100_000, Some(Instant::now())),
            None
        );
        let mut source = def.clone();
        let CoupledResource::SlopePaired { policy, .. } = &mut source else {
            unreachable!()
        };
        *policy = STRICT;
        assert_eq!(machine_value(&source, &[0], 1.), None);
    }
}
