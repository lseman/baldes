//! Full path/split controls with retained modern paired-state observations.
use baldes::graph::coupled::descriptor::{
    FrontOperationPolicy, PairedCostNormalization, PreparedSlopeFunction, SlopeRelationArc,
};
use baldes::graph::coupled::{
    CoupledKernel, CoupledLabelState, CoupledResource, PairedResourceMode,
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
    source_oracle: Option<f64>,
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
impl Case {
    fn graph(&self, policy: FrontOperationPolicy, scalar_cost: f64) -> Graph<2> {
        let n = self.arcs.len();
        assert_eq!(self.bounds.len(), n + 1);
        let mut graph = Graph::<2>::new(n + 1, 0).unwrap();
        for arc in 0..n {
            graph
                .add_arc(arc, arc + 1, if arc == 0 { scalar_cost } else { 0. })
                .unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::SlopePaired {
                initial: function(&[(self.initial, self.bounds[0].0), (0., self.bounds[0].1)]),
                initial_base: Window::new(self.bounds[0].0, self.bounds[0].1).unwrap(),
                arcs: self
                    .arcs
                    .iter()
                    .map(|arc| {
                        // Reference preparation takes consumption; BALDES takes the
                        // resulting arrival map G(t) = t + consumption(t).
                        let mut arrival = arc.consumption.clone();
                        arrival[0].0 += arrival[0].1;
                        for (slope, _) in &mut arrival[1..] {
                            *slope += 1.;
                        }
                        SlopeRelationArc {
                            arrival: function(&arrival),
                            increment: function(&arc.increment),
                            lower: function(&arc.lower),
                            upper: function(&arc.upper),
                            waiting: arc.waiting,
                        }
                    })
                    .collect(),
                mode: match self.kind {
                    4 => PairedResourceMode::Feasibility,
                    5 => PairedResourceMode::CostUnbounded,
                    6 => PairedResourceMode::CostBounded,
                    _ => panic!("invalid fixture mode"),
                },
                normalization: PairedCostNormalization::AbsoluteFinalValue,
                policy,
            }])
            .unwrap();
        graph
    }
}
fn follow(
    kernel: &CoupledKernel,
    arcs: impl Iterator<Item = usize>,
    forward: bool,
) -> Option<(Box<CoupledLabelState>, f64)> {
    let mut state = kernel.root(forward);
    let mut cost = 0.;
    for arc in arcs {
        let (next, delta) = kernel.extend(&state, arc, forward).unwrap()?;
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
fn backward_paired_joins_preserve_source_controls_at_every_reference_split() {
    let mut controls = 0;
    let mut reference_disagreements = 0;
    for case in corpus().cases {
        for reference in &case.reference {
            assert_eq!(
                reference.cost.to_bits(),
                u64::from_str_radix(&reference.cost_bits, 16).unwrap()
            );
            if reference.joined != case.source_oracle.is_some()
                || case
                    .source_oracle
                    .is_some_and(|cost| reference.joined && (cost - reference.cost).abs() >= 1e-9)
            {
                reference_disagreements += 1;
            }
        }
        for policy in [
            FrontOperationPolicy::FrontAlgebra,
            FrontOperationPolicy::StrictDomain,
        ] {
            let graph = case.graph(policy, 0.);
            let kernel = graph.coupled_kernel().unwrap();
            let n = case.arcs.len();
            check(
                kernel.replay(&(0..n).collect::<Vec<_>>()).unwrap(),
                case.source_oracle,
                &case.name,
            );
            assert_eq!(case.reference.len(), n + 1);
            for reference in &case.reference {
                let split = reference.split;
                let joined = match (
                    follow(kernel, 0..split, true),
                    follow(kernel, (split..n).rev(), false),
                ) {
                    (Some((f, fc)), Some((b, bc))) => {
                        kernel.join(&f, &b).unwrap().map(|cross| fc + bc + cross)
                    }
                    _ => None,
                };
                check(
                    joined,
                    case.source_oracle,
                    &format!("{} split={split} policy={policy:?}", case.name),
                );
                // Preserve ordinary reference agreement, while an independent
                // source oracle remains authoritative for reference anomalies.
                if reference.joined == case.source_oracle.is_some()
                    && case
                        .source_oracle
                        .map_or(true, |cost| (cost - reference.cost).abs() < 1e-9)
                {
                    check(
                        joined,
                        reference.joined.then_some(reference.cost),
                        &case.name,
                    );
                }
                controls += 1;
            }
        }
    }
    assert_eq!(controls, 1164);
    assert_eq!(reference_disagreements, 16);
}

#[test]
fn paired_pricing_and_enumeration_agree_across_search_directions_on_reference_paths() {
    let mut searches = 0;
    for case in corpus().cases {
        let graph = case.graph(FrontOperationPolicy::FrontAlgebra, -64.);
        for direction in [0., 1., 2.] {
            let mut solver = baldes::pricing::Solver::<2>::new(case.arcs.len() + 1, 0).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let priced = solver.price_phase(0, 64, false).unwrap();
            assert!(priced.exact, "{} direction={direction}", case.name);
            let routes = solver.enumerate(0.).unwrap();
            match case.source_oracle {
                Some(cost) => {
                    let expected = cost - 64.;
                    assert!(
                        (priced.minimum_reduced_cost - expected).abs() < 1e-9,
                        "{} direction={direction}: {priced:?}",
                        case.name
                    );
                    assert_eq!(routes.len(), 1, "{} direction={direction}", case.name);
                    assert_eq!(routes[0].arcs.len(), case.arcs.len());
                    assert!((routes[0].cost - expected).abs() < 1e-9);
                    assert!((routes[0].original_cost - expected).abs() < 1e-9);
                }
                None => {
                    assert!(
                        priced.columns.is_empty(),
                        "{} direction={direction}",
                        case.name
                    );
                    assert!(routes.is_empty(), "{} direction={direction}", case.name);
                }
            }
            searches += 1;
        }
    }
    assert_eq!(searches, 510);
}
