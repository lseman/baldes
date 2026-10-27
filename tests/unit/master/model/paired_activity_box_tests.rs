use super::*;
use crate::graph::coupled::{
    CoupledResource, FrontOperationPolicy, PairedCostNormalization, PairedResourceMode,
    PreparedSlopeFunction, SlopeRelationArc,
};

fn graph(integral: bool) -> Graph<2> {
    let file = if integral {
        include_str!("../../../fixtures/coupled/metasolver-paired-objective-mapping.json")
    } else {
        include_str!("../../../fixtures/coupled/metasolver-paired-preparation.json")
    };
    let corpus: serde_json::Value = serde_json::from_str(file).unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
    crate::master::model::format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap()
        .networks
        .remove(0)
        .graph
}

#[test]
fn paired_integral_endpoints_do_not_prove_integral_interior_controls() {
    let g = graph(false);
    let extent = range(&g, 0, false).unwrap().unwrap();
    assert!(extent.lower <= 1. && extent.upper >= 8.);
    assert!(integer_program_range(&g, 0).is_none());
    assert!(range(&g, 0, true).is_none());
    assert!(super::super::integral_activity(&g, 0).is_none());
    let g = graph(true);
    assert!(integer_program_range(&g, 0).is_some());
    assert!(range(&g, 0, true).is_some());
}

#[test]
fn paired_integer_programs_use_terminal_windows_on_cycles_and_decline_fractional_or_oversized_steps(
) {
    let function = |value, lo, hi| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, lo), (0., hi)]).unwrap()
    };
    let mut g = Graph::<2>::new(3, 0).unwrap();
    for (a, b) in [(0, 1), (1, 1), (1, 2)] {
        g.add_arc(a, b, 0.).unwrap();
    }
    let definition = |increment, waiting, upper| CoupledResource::SlopePaired {
        initial: function(0., 0., 7.),
        initial_base: Window::new(0., 7.).unwrap(),
        arcs: vec![
            SlopeRelationArc {
                arrival: PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 7.)])
                    .unwrap(),
                increment: function(increment, 0., 7.),
                lower: function(0., 1., 8.),
                upper: function(upper, 1., 8.),
                waiting
            };
            3
        ],
        mode: PairedResourceMode::CostUnbounded,
        normalization: PairedCostNormalization::AbsoluteFinalValue,
        policy: FrontOperationPolicy::StrictDomain,
    };
    g.set_coupled_resources(&[definition(1., 0, 8.)]).unwrap();
    assert!(!g.is_acyclic() && g.max_route_arcs().is_none());
    let extent = integer_program_range(&g, 0).unwrap();
    assert_eq!((extent.lower, extent.upper), (0., 8.));
    for (increment, waiting, upper) in [
        (0.5, 0, 8.),
        (1., -1, 8.),
        (9_007_199_254_740_992., 0, 9_007_199_254_740_992.),
    ] {
        g.set_coupled_resources(&[definition(increment, waiting, upper)])
            .unwrap();
        assert!(integer_program_range(&g, 0).is_none());
    }
}

#[test]
fn paired_terminal_certificates_respect_limits_and_decline_the_machine_domain() {
    let mut g = graph(false);
    let kernel = g.coupled_kernel().unwrap();
    assert!(kernel
        .paired_terminal_bounds(0, &[1, 2], &mut 0, None)
        .is_none());
    assert!(kernel
        .paired_terminal_bounds(0, &[1, 2], &mut 1_000_000, Some(Instant::now()))
        .is_none());
    assert_eq!(
        kernel.paired_terminal_bounds(0, &[1, 2], &mut 1_000_000, None),
        Some([0., 8.])
    );
    let mut definitions = g.coupled_kernel().unwrap().definitions().to_vec();
    if let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] {
        *policy = FrontOperationPolicy::FrontAlgebra;
    }
    g.set_coupled_resources(&definitions).unwrap();
    g.set_front_program_domain(true).unwrap();
    assert!(range(&g, 0, false).is_none());
    assert!(integer_program_range(&g, 0).is_none());
}
