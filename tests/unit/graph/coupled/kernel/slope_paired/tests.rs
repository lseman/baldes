use super::*;
use crate::graph::coupled::MachineArcFunctionInput;
use descriptor::MachineRootInput;

fn function(value: f64, slope: f64) -> PreparedSlopeFunction {
    descriptor::RawAffine::new(value, slope)
        .unwrap()
        .prepare(0., 2.)
        .unwrap()
}
fn ranged(intercept: f64, slope: f64, lower: f64, upper: f64) -> PreparedSlopeFunction {
    descriptor::RawAffine::new(intercept, slope)
        .unwrap()
        .prepare(lower, upper)
        .unwrap()
}
fn window(lower: f64, upper: f64) -> Window {
    Window::new(lower, upper).unwrap()
}

/// One arc of an explicit paired front program. `base` is the consumption
/// (not the arrival) as a function of the tail base value.
#[derive(Clone)]
struct ProgramArc {
    tail: Window,
    head: Window,
    base: PreparedSlopeFunction,
    dependent: PreparedSlopeFunction,
    lower: PreparedSlopeFunction,
    upper: PreparedSlopeFunction,
    waiting: i32,
}
impl ProgramArc {
    /// Zero consumption, unit increment and dependent bounds `[0, 100]` on
    /// base window `[0, 2]`.
    fn unit() -> Self {
        Self {
            tail: window(0., 2.),
            head: window(0., 2.),
            base: function(0., 0.),
            dependent: function(1., 0.),
            lower: function(0., 0.),
            upper: function(100., 0.),
            waiting: 0,
        }
    }
}
fn install(
    graph: &mut Graph<2>,
    mode: PairedResourceMode,
    root: &PreparedSlopeFunction,
    source: Window,
    terminal: Window,
    arcs: &[ProgramArc],
) {
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront { mode }])
        .unwrap();
    let inputs: Vec<_> = arcs
        .iter()
        .enumerate()
        .map(|(physical_arc, arc)| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: arc.tail,
            waiting_bounds: arc.head,
            waiting_rate: arc.waiting,
            base: &arc.base,
            dependent: MachineRootInput::Vector(&arc.dependent),
            dependent_lower: MachineRootInput::Vector(&arc.lower),
            dependent_upper: MachineRootInput::Vector(&arc.upper),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        MachineRootInput::Vector(root),
        source,
        terminal,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
}
/// Chain `0 -> 1 -> 2` with only the paired declaration, for tests that
/// install machine operands directly.
fn declared_chain(mode: PairedResourceMode) -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront { mode }])
        .unwrap();
    graph
}
/// Chain `0 -> 1 -> 2` with initial value 5 and two unit arcs.
fn chain(mode: PairedResourceMode, arcs: [ProgramArc; 2]) -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    install(
        &mut graph,
        mode,
        &function(5., 0.),
        window(0., 2.),
        window(0., 2.),
        &arcs,
    );
    graph
}
fn minimum(state: &CoupledLabelState) -> f64 {
    match state
        .slope_relation_fronts()
        .next()
        .unwrap()
        .unwrap()
        .minimum(FrontierBudget::default())
    {
        descriptor::FrontierOutcome::Complete(witness) => witness.value,
        _ => panic!("expected complete raw paired front"),
    }
}

#[test]
fn machine_mode_distinguishes_bounded_and_unbounded_costs() {
    for (mode, expected) in [
        (PairedResourceMode::Feasibility, 4),
        (PairedResourceMode::CostUnbounded, 5),
        (PairedResourceMode::CostBounded, 6),
    ] {
        let graph = declared_chain(mode);
        assert_eq!(
            graph.coupled_kernel().unwrap().slope_paired[0].machine_mode(),
            expected
        );
    }
}

#[test]
fn backward_machine_labels_extend_from_the_installed_root() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::Feasibility,
        }])
        .unwrap();
    let backward = PreparedMachineBackwardInputs {
        tail_lower: 0.,
        tail_upper: 2.,
        forward_at_tail_lower: 0.,
        mapped_upper: 2.,
        dependent: function(1., 0.),
    };
    let input = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: Some(backward),
    };
    let other_sink = input.clone();
    graph
        .set_paired_machine_inputs(0, Some(&[input.clone(), input.clone(), other_sink]))
        .unwrap();
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    assert!(matches!(
        graph.install_paired_machine_backward_root(
            0,
            Window::new(0., 2.).unwrap(),
            descriptor::FrontierBudget::default(),
        ),
        descriptor::FrontierOutcome::Complete(())
    ));
    graph.set_front_program_domain(true).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let label = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    let history = label.slope_relations[0].machine_history.as_ref().unwrap();
    let descriptor::MachinePathSnapshot::Backward {
        front,
        waiting_bits,
        key,
        ceiling,
        ..
    } = history.last()
    else {
        panic!("expected backward machine snapshot");
    };
    assert_eq!(front.components()[0].pairs(), &[(99., 0.), (0., 2.)]);
    assert_eq!(*waiting_bits, 0);
    assert_eq!(*ceiling, 100.);
    assert_eq!(*key, 1e12);
    assert_eq!(label.slope_relations[0].minimum, 0.);
    let earlier = kernel.extend(&label, 0, false).unwrap().unwrap().0;
    let history = earlier.slope_relations[0].machine_history.as_ref().unwrap();
    assert_eq!(history.len(), 3);
    let descriptor::MachinePathSnapshot::Backward { front, ceiling, .. } = history.last() else {
        panic!("expected second backward machine snapshot");
    };
    assert_eq!(front.components()[0].pairs(), &[(98., 0.), (0., 2.)]);
    assert_eq!(*ceiling, 100.);

    let mut changed = input.clone();
    changed.dependent_upper = function(110., 0.);
    graph
        .set_paired_machine_inputs(0, Some(&[input.clone(), input, changed]))
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    assert!(root.slope_relations[0].machine_history.is_none());
    assert!(kernel.extend(&root, 1, false).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn backward_machine_arc_operands_round_trip_in_json() {
    let arc = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: Some(PreparedMachineBackwardInputs {
            tail_lower: -0.0,
            tail_upper: 2.,
            forward_at_tail_lower: 0.,
            mapped_upper: 2.,
            dependent: function(1., 0.),
        }),
    };
    let wire = serde_json::to_string(&arc).unwrap();
    let parsed: PreparedMachineArc = serde_json::from_str(&wire).unwrap();
    let original = arc.backward.unwrap();
    let restored = parsed.backward.unwrap();
    assert_eq!(restored.tail_lower.to_bits(), original.tail_lower.to_bits());
    assert_eq!(restored.dependent.pairs(), original.dependent.pairs());
}

#[test]
fn machine_root_without_arc_inputs_does_not_fall_back_to_source_extension() {
    let mut graph = declared_chain(PairedResourceMode::Feasibility);
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    assert!(graph.validate().is_err());
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel.extend(&kernel.root(true), 0, true).is_err());
}

#[test]
fn machine_label_root_uses_explicit_prepared_front() {
    let mut graph = declared_chain(PairedResourceMode::Feasibility);
    graph
        .set_paired_machine_root(0, Some(&function(7., 0.)))
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    let label = graph.coupled_kernel().unwrap().root(true);
    let root = &label.slope_relations[0];
    assert!(root.machine_history.is_some());
    assert_eq!(
        root.front.as_ref().unwrap().components()[0].pairs()[0].0,
        7.
    );
    assert_eq!(root.minimum.to_bits(), 0.0f64.to_bits());
}

#[test]
fn machine_inputs_require_root_and_reject_changed_arc_identity() {
    let mut graph = declared_chain(PairedResourceMode::Feasibility);
    let input = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: None,
    };
    let mut nonterminal = input.clone();
    nonterminal.dependent_upper = function(900., 0.);
    graph
        .set_paired_machine_inputs(0, Some(&[nonterminal, input]))
        .unwrap();
    let bounds = Window::new(0., 2.).unwrap();
    let budget = descriptor::FrontierBudget::default();
    let backward = match graph.prepare_paired_machine_backward_root(0, bounds, budget) {
        descriptor::FrontierOutcome::Complete(root) => root,
        other => panic!("unexpected backward root: {other:?}"),
    };
    assert_eq!(backward.key(), 100.);
    assert!(matches!(
        graph.install_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Complete(())
    ));
    let paired = &mut graph.coupled.as_mut().unwrap().slope_paired[0];
    let installed = paired.machine_backward_root.as_ref().unwrap().clone();
    assert!(matches!(
        paired.set_machine_backward_root(
            backward,
            descriptor::FrontierBudget {
                max_pairs: 1,
                ..budget
            },
        ),
        descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::PairLimit)
    ));
    assert!(Arc::ptr_eq(
        paired.machine_backward_root.as_ref().unwrap(),
        &installed
    ));
    graph.set_front_program_domain(true).unwrap();
    let label = graph.coupled_kernel().unwrap().root(false);
    let history = label.slope_relations[0].machine_history.as_ref().unwrap();
    let descriptor::MachinePathSnapshot::Root(root) = history.last() else {
        panic!("expected backward root");
    };
    assert_eq!(root.key(), 100.);
    assert_eq!(root.waiting_bits(), 0);
    assert!(matches!(
        history.backward_cost_comparison(history, 6, 0.25, budget),
        descriptor::FrontierOutcome::Complete(0.25)
    ));
    assert!(matches!(
        history.backward_feasibility_comparison(history, budget),
        descriptor::FrontierOutcome::Complete(())
    ));
    assert!(matches!(
        history.backward_cost_comparison(
            history,
            6,
            0.25,
            descriptor::FrontierBudget {
                max_operations: 0,
                ..budget
            }
        ),
        descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::WorkLimit)
    ));
    assert!(graph.validate().is_err());
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    assert!(graph.validate().is_ok());
    graph.sink = 1;
    assert!(matches!(
        graph.prepare_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Unknown(_)
    ));
    graph.sink = 2;
    graph.arcs[0].id += 10;
    assert!(graph.validate().is_err());
    assert!(matches!(
        graph.prepare_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Unknown(_)
    ));
}

#[test]
fn exact_cost_suffix_dominance_and_joins_survive_zero_increment_cycles() {
    for weight in [1., 2.] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (1, 1), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let arcs: Vec<_> = [0.25, 0., 0.25]
            .into_iter()
            .map(|increment| ProgramArc {
                tail: window(0., 1.),
                head: window(0., 1.),
                base: ranged(0., 0., 0., 1.),
                dependent: ranged(increment, 0., 0., 1.),
                lower: ranged(0., 0., 0., 1.),
                upper: ranged(8., 0., 0., 1.),
                waiting: 0,
            })
            .collect();
        install(
            &mut graph,
            PairedResourceMode::CostUnbounded,
            &ranged(0.5, 0., 0., 1.),
            window(0., 1.),
            window(0., 1.),
            &arcs,
        );
        graph
            .set_coupled_objective_coefficients(&[(0, weight)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let prefix = kernel
            .extend(&kernel.root(true), 0, true)
            .unwrap()
            .unwrap()
            .0;
        let direct = kernel
            .extend(&kernel.root(false), 2, false)
            .unwrap()
            .unwrap()
            .0;
        let mut cycled = direct.clone();
        for _ in 0..8 {
            cycled = kernel.extend(&cycled, 1, false).unwrap().unwrap().0;
        }
        assert_eq!(kernel.dominance_penalty(&direct, &cycled, false), Some(0.));
        assert_eq!(kernel.dominance_penalty(&cycled, &direct, false), Some(0.));
        // Independent recurrence: .5 + .25 + eight zeros + .25 = 1.
        let expected = weight;
        let mut path = vec![0];
        path.extend([1; 8]);
        path.push(2);
        assert_eq!(kernel.replay(&path).unwrap(), Some(expected));
        assert_eq!(
            kernel.join_objective_costs(&prefix, &direct).unwrap(),
            Some((1., expected))
        );
        assert_eq!(
            kernel.join_objective_costs(&prefix, &cycled).unwrap(),
            Some((1., expected))
        );
    }
}

#[test]
fn feasibility_fronts_compare_in_both_directions() {
    // Alternative suffixes allow dependent values up to 10 and 8.
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let arc = |upper| ProgramArc {
        upper: function(upper, 0.),
        ..ProgramArc::unit()
    };
    install(
        &mut graph,
        PairedResourceMode::Feasibility,
        &function(5., 0.),
        window(0., 2.),
        window(0., 2.),
        &[arc(10.), arc(8.)],
    );
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let wide = kernel.extend(&root, 0, false).unwrap().unwrap().0;
    let narrow = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    assert_eq!(kernel.dominance_penalty(&wide, &narrow, false), Some(0.));
    assert_eq!(kernel.dominance_penalty(&narrow, &wide, false), None);
}

fn shifted_prefixes(upper: f64) -> (Graph<2>, Box<CoupledLabelState>, Box<CoupledLabelState>) {
    let mut graph = Graph::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let arc = |dependent| ProgramArc {
        dependent,
        upper: function(upper, 0.),
        ..ProgramArc::unit()
    };
    // The falling prefix ends at 2 - x, the flat one at 1.
    install(
        &mut graph,
        PairedResourceMode::CostUnbounded,
        &function(0., 0.),
        window(0., 2.),
        window(0., 2.),
        &[
            arc(function(2., -1.)),
            arc(function(1., 0.)),
            arc(function(0., 0.)),
        ],
    );
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let prefix = |arc| {
        let mut state = kernel.extend(&root, arc, true).unwrap().unwrap().0;
        state.account_extension(&root, 0., 0.).unwrap();
        state
    };
    let (falling, flat) = (prefix(0), prefix(1));
    (graph, falling, flat)
}
#[test]
fn unbounded_paired_front_pays_its_excess() {
    let (graph, falling, flat) = shifted_prefixes(1e12);
    let kernel = graph.coupled_kernel().unwrap();
    // Falling exceeds flat by 1 at x=0: 1 + objective 1 - objective 0.
    let penalty = kernel.dominance_penalty(&falling, &flat, true).unwrap();
    assert!((2. ..=2. + 1e-12).contains(&penalty), "{penalty}");
    // Flat exceeds falling by 1 at x=2, already paid by its objective.
    assert_eq!(kernel.dominance_penalty(&flat, &falling, true), Some(0.));
    // An upper bound above every reachable value leaves the same native gaps.
    let (graph, falling, flat) = shifted_prefixes(100.);
    let kernel = graph.coupled_kernel().unwrap();
    let penalty = kernel.dominance_penalty(&falling, &flat, true).unwrap();
    assert!((2. ..=2. + 1e-12).contains(&penalty), "{penalty}");
    assert_eq!(kernel.dominance_penalty(&flat, &falling, true), Some(0.));
}

#[test]
fn paired_objective_keeps_root_zero_and_state_unscaled() {
    let graph = chain(
        PairedResourceMode::CostBounded,
        [ProgramArc::unit(), ProgramArc::unit()],
    );
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay(&[]).unwrap(), Some(0.));
    let root = kernel.root(true);
    assert_eq!(minimum(&root), 5.);
    assert_eq!(root.resource_objective, 0.);
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(cost, 7.);
    assert_eq!(minimum(&state), 7.);
    for split in 0..=2 {
        let f = kernel.replay_state(&[0, 1][..split]).unwrap().unwrap().0;
        let mut b = kernel.root(false);
        for &a in [0, 1][split..].iter().rev() {
            b = kernel.extend(&b, a, false).unwrap().unwrap().0;
        }
        assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), Some(7.));
    }
}

#[test]
fn paired_pricing_and_enumeration_charge_absolute_initial_value_in_all_modes() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v, cost) in [(0, 1, -10.), (1, 3, -10.), (0, 2, -4.), (2, 3, -4.)] {
        let a = graph.add_arc(u, v, cost).unwrap();
        graph.arc_mut(a).unwrap().consumption[0] = 1.;
    }
    install(
        &mut graph,
        PairedResourceMode::CostBounded,
        &function(5., 0.),
        window(0., 2.),
        window(0., 2.),
        &vec![ProgramArc::unit(); 4],
    );
    for mode in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = graph.clone();
        solver.enumeration_preserve_arc_paths = true;
        solver
            .set_option("BALDESuseBidirectionalSearch", mode)
            .unwrap();
        assert_eq!(
            solver
                .price_phase(0, 20, false)
                .unwrap()
                .minimum_reduced_cost,
            -13.
        );
        let mut costs: Vec<_> = solver
            .enumerate(0.)
            .unwrap()
            .into_iter()
            .map(|route| {
                assert_eq!(route.cost, route.original_cost);
                route.cost
            })
            .collect();
        costs.sort_by(f64::total_cmp);
        assert_eq!(costs, vec![-13., -1.]);
    }
}

#[test]
fn paired_feasibility_suffix_is_replayed_even_without_resource_objective() {
    let arc = ProgramArc {
        upper: function(6.5, 0.),
        ..ProgramArc::unit()
    };
    let graph = chain(PairedResourceMode::Feasibility, [arc.clone(), arc]);
    let kernel = graph.coupled_kernel().unwrap();
    assert!(!kernel.has_costs());
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), None);
    let f = kernel
        .extend(&kernel.root(true), 0, true)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(kernel.join(&f, &b).unwrap(), None);
}

#[cfg(feature = "clp")]
#[test]
fn paired_phase_i_preserves_physical_front_and_certifies_the_path() {
    let graph = chain(
        PairedResourceMode::CostBounded,
        [ProgramArc::unit(), ProgramArc::unit()],
    );
    let mut kernel = graph.coupled_kernel().unwrap().clone();
    kernel.suppress_costs();
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(cost, 0.);
    assert_eq!(minimum(&state), 7.);
    assert_eq!(kernel.certify(&[0, 1]), Some((true, [0., 0.])));
    let json = serde_json::to_string(kernel.definitions()).unwrap();
    let restored: Vec<CoupledResource> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, kernel.definitions());
}

fn physical_graph(vertices: usize, arcs: &[(usize, usize, f64)]) -> Graph<2> {
    let mut graph = Graph::new(vertices, 1).unwrap();
    for vertex in 0..vertices {
        graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for &(tail, head, cost) in arcs {
        let arc = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph
}
#[test]
fn composed_arrivals_variable_increments_and_signed_waiting_match_native_splits() {
    // Native joins from internal/reference/metasolver_paired_chain_splits.py
    // (cases composed-waiting-*). With initial coordinate t, first head x and
    // second head y the continuous optimum is 9 (k=1) or 1 (k=-1); the
    // native joins do not attain it and depend on the split.
    for (waiting, native) in [(-1, [3.6666666666666714, 3.5, 3.5]), (1, [9., 9.5, 9.5])] {
        let mut graph = physical_graph(3, &[(0, 1, -10.), (1, 2, -10.)]);
        // Arrival 2t is consumption t on each tail window.
        let arc = |tail: Window, head: Window| ProgramArc {
            tail,
            head,
            base: ranged(0., 1., 0., 4.),
            dependent: ranged(1., 1., 0., 4.),
            lower: ranged(0., 0., head.lower, head.upper),
            upper: ranged(100., 0., head.lower, head.upper),
            waiting,
        };
        let (source, first, second) = (window(0., 2.), window(1., 4.), window(0., 8.));
        install(
            &mut graph,
            if waiting < 0 {
                PairedResourceMode::CostUnbounded
            } else {
                PairedResourceMode::CostBounded
            },
            &ranged(5., 2., 0., 2.),
            source,
            second,
            &[arc(source, first), arc(first, second)],
        );
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(
            kernel.replay(&[0, 1]).unwrap(),
            Some(native[0]),
            "waiting={waiting}"
        );
        for split in 0..=2 {
            let f = kernel.replay_state(&[0, 1][..split]).unwrap().unwrap().0;
            let mut b = kernel.root(false);
            for &a in [0, 1][split..].iter().rev() {
                b = kernel.extend(&b, a, false).unwrap().unwrap().0;
            }
            assert_eq!(
                kernel.join_resource_objective(&f, &b).unwrap(),
                Some(native[split]),
                "waiting={waiting} split={split}"
            );
        }
        for search in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(3, 1).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", search)
                .unwrap();
            let priced = solver
                .price_phase(0, 20, false)
                .unwrap()
                .minimum_reduced_cost;
            assert!(
                native.contains(&(priced + 20.)),
                "waiting={waiting} search={search}: {priced}"
            );
            let routes = solver.enumerate(0.).unwrap();
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0].cost, priced);
            assert_eq!(routes[0].original_cost, priced);
        }
    }
}
fn composed_bounds_program() -> (Window, Window, Vec<ProgramArc>) {
    let (source, middle, sink) = (window(0., 2.), window(1., 4.), window(2., 4.));
    let first = ProgramArc {
        tail: source,
        head: middle,
        base: ranged(0., 1., 0., 4.),
        dependent: ranged(1., 1., 0., 4.),
        lower: ranged(0., 1., 1., 4.),
        upper: ranged(100., 0., 1., 4.),
        waiting: 0,
    };
    let second = ProgramArc {
        tail: middle,
        head: sink,
        base: ranged(0., 1., 0., 4.),
        dependent: ranged(1., 1., 0., 4.),
        lower: ranged(0., 0., 2., 4.),
        upper: ranged(5., -1., 2., 4.),
        waiting: 0,
    };
    (source, sink, vec![first, second])
}

#[test]
fn composed_function_bounds_retain_the_native_zero_width_nan_slope() {
    let mut graph = physical_graph(3, &[(0, 1, -3.), (1, 2, -3.)]);
    let (source, sink, arcs) = composed_bounds_program();
    // Cost factories reject the nonconstant FIRST lower bound. The raw
    // cost-mode operator behavior is checked separately, not used to bypass
    // that actual factory restriction.
    install(
        &mut graph,
        PairedResourceMode::Feasibility,
        &ranged(0., 0., 0., 2.),
        source,
        sink,
        &arcs,
    );
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay_state(&[0, 1]).unwrap(), None);
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(0.));
}

#[test]
fn retained_nan_suffix_is_consumed_by_a_factory_valid_leading_arc_in_every_mode() {
    for (mode, expected) in [
        (
            PairedResourceMode::Feasibility,
            [None, Some(0.), None, None],
        ),
        (
            PairedResourceMode::CostUnbounded,
            [Some(3.5), Some(3.), Some(3.5), None],
        ),
        (
            PairedResourceMode::CostBounded,
            [None, Some(3.), None, None],
        ),
    ] {
        let mut graph = physical_graph(4, &[(0, 1, 0.), (1, 2, 0.), (2, 3, 0.)]);
        let (source, sink, mut arcs) = composed_bounds_program();
        // The native initializer's first-arc checks accept these constant
        // bounds. Later arcs retain their original nonconstant bounds.
        arcs.insert(
            0,
            ProgramArc {
                tail: source,
                head: source,
                base: function(0., 0.),
                dependent: function(0., 0.),
                lower: function(-100., 0.),
                upper: function(100., 0.),
                waiting: 0,
            },
        );
        install(&mut graph, mode, &function(0., 0.), source, sink, &arcs);
        let kernel = graph.coupled_kernel().unwrap();
        for (split, expected) in expected.into_iter().enumerate() {
            let follow = |range: Vec<usize>, forward| {
                let mut state = kernel.root(forward);
                for arc in range {
                    state = kernel.extend(&state, arc, forward).unwrap()?.0;
                }
                Some(state)
            };
            let forward = follow((0..split).collect(), true);
            let backward = follow((split..3).rev().collect(), false);
            let joined = match (forward, backward) {
                (Some(f), Some(b)) => kernel.join_original_objective(&f, &b).unwrap(),
                _ => None,
            };
            assert_eq!(joined, expected, "mode={mode:?} split={split}");
        }
    }
}

#[test]
fn joined_machine_reconstruction_preserves_consumptions_and_rejects_replaced_roots() {
    use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::PairedFront { mode: mode }])
            .unwrap();
        let input = PreparedMachineArc {
            tail_limit: 2.,
            head_limit: 2.,
            waiting_lower: 0.,
            waiting_upper: 2.,
            waiting_rate: 0,
            forward_mapping: function(0., 1.),
            inverse: function(0., 1.),
            sum_vector: function(1., 0.),
            dependent_lower: function(0., 0.),
            dependent_upper: function(100., 0.),
            backward: Some(PreparedMachineBackwardInputs {
                tail_lower: 0.,
                tail_upper: 2.,
                forward_at_tail_lower: 0.,
                mapped_upper: 2.,
                dependent: function(1., 0.),
            }),
        };
        graph
            .set_paired_machine_inputs(0, Some(&[input.clone(), input]))
            .unwrap();
        graph
            .set_paired_machine_root(0, Some(&function(5., 0.)))
            .unwrap();
        let budget = FrontierBudget::default();
        assert!(matches!(
            graph.install_paired_machine_backward_root(0, Window::new(0., 2.).unwrap(), budget,),
            FrontierOutcome::Complete(())
        ));
        graph.set_front_program_domain(true).unwrap();
        if mode == PairedResourceMode::CostUnbounded {
            let kernel = graph.coupled_kernel().unwrap();
            let first = kernel
                .extend(&kernel.root(true), 0, true)
                .unwrap()
                .unwrap()
                .0;
            assert_eq!(first.resource_objective, 6.);
            let backward = kernel
                .extend(&kernel.root(false), 1, false)
                .unwrap()
                .unwrap()
                .0;
            // The backward label charges its own suffix increment.
            assert_eq!(backward.resource_objective, 1.);
            assert_eq!(
                kernel.join_resource_objective(&first, &backward).unwrap(),
                Some(7.)
            );
            // This test reconstructs raw consumptions; it must not bypass the
            // still-unproved nonzero backward objective accounting contract.
            graph.coupled.as_mut().unwrap().suppress_costs();
        }
        let kernel = graph.coupled_kernel().unwrap();
        let forward = kernel
            .extend(&kernel.root(true), 0, true)
            .unwrap()
            .unwrap()
            .0;
        let backward = kernel
            .extend(&kernel.root(false), 1, false)
            .unwrap()
            .unwrap()
            .0;
        if mode == PairedResourceMode::Feasibility {
            assert_eq!(kernel.join(&forward, &backward).unwrap(), Some(0.));
        }
        let FrontierOutcome::Complete(joined) =
            graph.reconstruct_paired_machine_join(0, &forward, &backward, budget)
        else {
            panic!("joined machine path should reconstruct");
        };
        // Independent exact recurrence: initial 5 plus one unit on each arc.
        assert_eq!(joined.path.base, vec![0., 0., 0.]);
        assert_eq!(joined.path.dependent, vec![5., 6., 7.]);
        assert_eq!(joined.path.waiting, vec![0., 0.]);
        assert_eq!(joined.initial_waiting, 0.);
        assert_eq!(joined.selection.coordinate, 0.);
        assert_eq!(
            joined.selection.dependent,
            (mode == PairedResourceMode::CostUnbounded).then_some(7.)
        );
        let FrontierOutcome::Complete(label_join) =
            joined.selection.join_labels(-20., 1., 0., 1, 2)
        else {
            panic!("raw reconstructed label costs should combine");
        };
        assert!(label_join.compatible);
        assert_eq!(
            label_join.reduced_cost,
            if mode == PairedResourceMode::CostUnbounded {
                -12.
            } else {
                -19.
            }
        );
        assert!(matches!(
            graph.reconstruct_paired_machine_join(
                0,
                &forward,
                &backward,
                FrontierBudget {
                    max_operations: 0,
                    ..budget
                },
            ),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
        assert!(matches!(
            graph.install_paired_machine_backward_root(0, Window::new(0., 1.5).unwrap(), budget,),
            FrontierOutcome::Complete(())
        ));
        assert!(matches!(
            graph.reconstruct_paired_machine_join(0, &forward, &backward, budget,),
            FrontierOutcome::Unknown(_)
        ));
    }
}

#[test]
fn constant_identity_suffixes_discard_positive_cycles_without_bounded_cache() {
    for mode in [
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (1, 1), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let arc = ProgramArc {
            tail: window(0., 8.),
            head: window(0., 8.),
            base: ranged(0., 0., 0., 8.),
            dependent: ranged(1., 0., 0., 8.),
            lower: ranged(0., 0., 0., 8.),
            upper: ranged(8., 0., 0., 8.),
            waiting: 0,
        };
        install(
            &mut graph,
            mode,
            &ranged(0., 0., 0., 8.),
            window(0., 8.),
            window(0., 8.),
            &vec![arc; 3],
        );
        let kernel = graph.coupled_kernel().unwrap();
        let direct = kernel
            .extend(&kernel.root(false), 2, false)
            .unwrap()
            .unwrap()
            .0;
        let mut cycled = direct.clone();
        for _ in 0..3 {
            cycled = kernel.extend(&cycled, 1, false).unwrap().unwrap().0;
            assert_eq!(kernel.dominance_penalty(&direct, &cycled, false), Some(0.));
            // The native scan returns a gap rather than rejecting a lower
            // ceiling; with its cost the cycled suffix never prunes direct.
            let penalty = kernel.dominance_penalty(&cycled, &direct, false);
            assert!(penalty.map_or(true, |penalty| {
                cycled.resource_objective + penalty > direct.resource_objective
            }));
        }
        // Independent recurrence for the full paths: each traversed
        // arc adds exactly one to the constant initial value.
        assert_eq!(kernel.replay(&[0, 2]).unwrap().unwrap(), 2.);
        assert_eq!(kernel.replay(&[0, 1, 1, 1, 2]).unwrap().unwrap(), 5.);
        // An associated owner coefficient does not reverse the raw front order.
        graph
            .set_coupled_objective_coefficients(&[(0, -1.)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.replay_pricing(&[0, 2]).unwrap(), Some(2.));
        assert_eq!(kernel.replay(&[0, 2]).unwrap(), Some(-2.));
    }
}

#[test]
fn retained_nan_suffix_lower_bounds_match_the_native_cleared_storage_lifecycle() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        for lower in [-0.5, -0.25, 0., 0.25, 1.] {
            if mode == PairedResourceMode::CostBounded && lower > 0. {
                continue;
            }
            let mut graph = physical_graph(4, &[(0, 1, 0.), (1, 2, 0.), (2, 3, 0.)]);
            let (source, sink, mut arcs) = composed_bounds_program();
            arcs.insert(
                0,
                ProgramArc {
                    tail: source,
                    head: source,
                    base: function(0., 0.),
                    dependent: function(0., 0.),
                    lower: function(lower, 0.),
                    upper: function(100., 0.),
                    waiting: 0,
                },
            );
            install(&mut graph, mode, &function(0., 0.), source, sink, &arcs);
            let expected = if lower == -0.5 {
                (mode == PairedResourceMode::CostUnbounded).then_some(3.5)
            } else {
                Some(if mode == PairedResourceMode::Feasibility {
                    0.
                } else {
                    3.
                })
            };
            // These captured values belong to split zero. Route replay
            // considers other splits, including a successful split one when
            // split zero rejects the negative lower-bound point.
            let kernel = graph.coupled_kernel().unwrap();
            let mut backward = Some(kernel.root(false));
            for arc in (0..3).rev() {
                backward = backward.and_then(|state| {
                    kernel
                        .extend(&state, arc, false)
                        .unwrap()
                        .map(|next| next.0)
                });
            }
            let joined = backward.and_then(|state| {
                kernel
                    .join_original_objective(&kernel.root(true), &state)
                    .unwrap()
            });
            assert_eq!(joined, expected, "mode={mode:?} lower={lower}");
        }
    }
}
