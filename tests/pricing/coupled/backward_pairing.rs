//! Native paired joins at every split, and search contracts on the same paths.
//! The fixtures' control oracles are diagnostic; native joins need not attain them.
use baldes::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
};
use baldes::graph::coupled::{
    CoupledKernel, CoupledLabelState, CoupledResource, MachineArcFunctionInput, PairedResourceMode,
};
use baldes::{Graph, Window};
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    reference_sha256: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    kind: i32,
    initial: f64,
    bounds: Vec<(f64, f64)>,
    arcs: Vec<Arc>,
    reference: Vec<Observation>,
}
#[derive(Deserialize)]
struct Arc {
    waiting: i32,
    consumption: Vec<(f64, f64)>,
    increment: Vec<(f64, f64)>,
    lower: Vec<(f64, f64)>,
    upper: Vec<(f64, f64)>,
}
#[derive(Deserialize)]
struct Observation {
    split: usize,
    joined: bool,
    cost: f64,
    cost_bits: String,
}
fn corpus() -> Corpus {
    let corpus: Corpus = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-backward-paths.json"
    ))
    .unwrap();
    assert_eq!(
        corpus.reference_sha256,
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(corpus.cases.len(), 170);
    corpus
}
fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
}
/// Chain graph with one explicit paired front program; functions are given on
/// their tail windows and dependent bounds on head windows.
fn chain_graph(
    name: &str,
    kind: i32,
    root: MachineRootInput<'_>,
    bounds: &[(f64, f64)],
    arcs: &[Arc],
    scalar_cost: f64,
) -> Graph<2> {
    let n = arcs.len();
    assert_eq!(bounds.len(), n + 1);
    let mut graph = Graph::<2>::new(n + 1, 0).unwrap();
    for arc in 0..n {
        graph
            .add_arc(arc, arc + 1, if arc == 0 { scalar_cost } else { 0. })
            .unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: match kind {
                4 => PairedResourceMode::Feasibility,
                5 => PairedResourceMode::CostUnbounded,
                6 => PairedResourceMode::CostBounded,
                _ => panic!("invalid fixture mode"),
            },
        }])
        .unwrap();
    let functions: Vec<_> = arcs
        .iter()
        .map(|arc| {
            (
                function(&arc.consumption),
                function(&arc.increment),
                function(&arc.lower),
                function(&arc.upper),
            )
        })
        .collect();
    let window = |(lower, upper)| Window::new(lower, upper).unwrap();
    let inputs: Vec<_> = functions
        .iter()
        .enumerate()
        .map(
            |(physical_arc, (base, dependent, lower, upper))| MachineArcFunctionInput {
                physical_arc,
                tail_bounds: window(bounds[physical_arc]),
                waiting_bounds: window(bounds[physical_arc + 1]),
                waiting_rate: arcs[physical_arc].waiting,
                base,
                dependent: MachineRootInput::Vector(dependent),
                dependent_lower: MachineRootInput::Vector(lower),
                dependent_upper: MachineRootInput::Vector(upper),
            },
        )
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        root,
        window(bounds[0]),
        window(bounds[n]),
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{name}: {result:?}"
    );
    graph
}
impl Case {
    fn graph(&self, scalar_cost: f64) -> Graph<2> {
        let root = MachineRootInput::Affine(RawAffine::new(self.initial, 0.).unwrap());
        chain_graph(
            &self.name,
            self.kind,
            root,
            &self.bounds,
            &self.arcs,
            scalar_cost,
        )
    }
}
/// Join value at `split`: forward labels over the prefix, backward labels
/// over the suffix. `None` when an extension or the join is infeasible.
fn split_join(kernel: &CoupledKernel, n: usize, split: usize, context: &str) -> Option<f64> {
    match (
        follow(kernel, 0..split, true, context),
        follow(kernel, (split..n).rev(), false, context),
    ) {
        (Some((forward, forward_cost)), Some((backward, backward_cost))) => kernel
            .join(&forward, &backward)
            .unwrap()
            .map(|cross| forward_cost + backward_cost + cross),
        _ => None,
    }
}
fn follow(
    kernel: &CoupledKernel,
    arcs: impl Iterator<Item = usize>,
    forward: bool,
    context: &str,
) -> Option<(Box<CoupledLabelState>, f64)> {
    let mut state = kernel.root(forward);
    let mut cost = 0.;
    for arc in arcs {
        let (next, delta) = kernel
            .extend(&state, arc, forward)
            .unwrap_or_else(|error| panic!("{context}: arc={arc} forward={forward}: {error}"))?;
        state = next;
        cost += delta;
    }
    Some((state, cost))
}
fn check(actual: Option<f64>, expected: Option<f64>, context: &str) {
    match (actual, expected) {
        (Some(a), Some(b)) => assert!((a - b).abs() < 1e-9, "{context}: actual={a}, expected={b}"),
        (None, None) => (),
        _ => panic!("{context}: actual={actual:?}, expected={expected:?}"),
    }
}

#[test]
fn backward_paired_joins_match_native_at_every_recorded_split() {
    let mut controls = 0;
    let mut failures = Vec::new();
    for case in corpus().cases {
        let result = std::panic::catch_unwind(|| {
            let mut controls = 0;
            let graph = case.graph(0.);
            let kernel = graph.coupled_kernel().unwrap();
            let n = case.arcs.len();
            assert_eq!(case.reference.len(), n + 1);
            for reference in &case.reference {
                assert_eq!(
                    reference.cost.to_bits(),
                    u64::from_str_radix(&reference.cost_bits, 16).unwrap()
                );
                let split = reference.split;
                let joined = split_join(kernel, n, split, &case.name);
                check(
                    joined,
                    reference.joined.then_some(reference.cost),
                    &format!("{} split={split}", case.name),
                );
                controls += 1;
            }
            controls
        });
        match result {
            Ok(count) => controls += count,
            Err(error) => {
                let message = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                failures.push(format!("{}: {message}", case.name));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} native cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(controls, 582);
}

/// Native joins are split-dependent, so a search need not return the
/// cheapest split or the source oracle. Contract: every search direction
/// prices and enumerates the same single path value, and that value is one
/// of the native join values recorded for the path.
#[test]
fn paired_pricing_and_enumeration_agree_across_search_directions_on_reference_paths() {
    let mut searches = 0;
    for case in corpus().cases {
        let graph = case.graph(-64.);
        let native: Vec<_> = case
            .reference
            .iter()
            .map(|r| r.joined.then_some(r.cost))
            .collect();
        let mut observed = None;
        for direction in [0., 1., 2.] {
            let mut solver = baldes::pricing::Solver::<2>::new(case.arcs.len() + 1, 0).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let priced = solver.price_phase(0, 64, false).unwrap();
            assert!(priced.exact, "{} direction={direction}", case.name);
            let value = (!priced.columns.is_empty()).then(|| priced.minimum_reduced_cost + 64.);
            assert!(
                native.contains(&value),
                "{} direction={direction}: priced {value:?} is not a native split value {native:?}",
                case.name
            );
            let routes = solver.enumerate(0.).unwrap();
            assert_eq!(
                routes.len(),
                usize::from(value.is_some()),
                "{} direction={direction}",
                case.name
            );
            for route in &routes {
                assert_eq!(route.arcs.len(), case.arcs.len());
                assert_eq!(
                    Some(route.cost + 64.),
                    value,
                    "{} direction={direction}",
                    case.name
                );
                assert_eq!(route.cost, route.original_cost);
            }
            assert_eq!(
                *observed.get_or_insert(value),
                value,
                "{} direction={direction}",
                case.name
            );
            searches += 1;
        }
    }
    assert_eq!(searches, 510);
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Initial {
    Constant(f64),
    Function(Vec<(f64, f64)>),
}
#[derive(Deserialize)]
struct ChainCorpus {
    reference_sha256: String,
    cases: Vec<ChainCase>,
}
#[derive(Deserialize)]
struct ChainCase {
    name: String,
    kind: i32,
    initial: Initial,
    bounds: Vec<(f64, f64)>,
    arcs: Vec<Arc>,
    native: Option<Vec<Option<f64>>>,
    native_failure: Option<String>,
}

/// Native joins of constant, nonconstant-initial and composed chain programs
/// at every split. Recorded by internal/reference/metasolver_paired_chain_splits.py,
/// which also stores a continuous control oracle: native splits need not
/// attain it, and the forward and backward splits can differ.
#[test]
fn paired_chain_joins_match_native_at_every_split() {
    let corpus: ChainCorpus = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-chain-splits.json"
    ))
    .unwrap();
    assert_eq!(
        corpus.reference_sha256,
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let (mut splits, mut aborted) = (0, 0);
    for case in &corpus.cases {
        let Some(native) = &case.native else {
            assert!(case.native_failure.is_some(), "{}", case.name);
            aborted += 1;
            continue;
        };
        let initial;
        let root = match &case.initial {
            Initial::Constant(value) => {
                MachineRootInput::Affine(RawAffine::new(*value, 0.).unwrap())
            }
            Initial::Function(pairs) => {
                initial = function(pairs);
                MachineRootInput::Vector(&initial)
            }
        };
        let graph = chain_graph(&case.name, case.kind, root, &case.bounds, &case.arcs, 0.);
        let kernel = graph.coupled_kernel().unwrap();
        let n = case.arcs.len();
        assert_eq!(native.len(), n + 1);
        for (split, &expected) in native.iter().enumerate() {
            let context = format!("{} split={split}", case.name);
            check(split_join(kernel, n, split, &context), expected, &context);
            splits += 1;
        }
    }
    assert_eq!((splits, aborted), (492, 4));
}
