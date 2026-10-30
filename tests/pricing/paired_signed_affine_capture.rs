#[test]
fn prepared_signed_affine_owner_preserves_native_enumerated_paths_after_fixing() {
    use baldes::enumeration::{Controls, Enumeration};
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use baldes::graph::coupled::{CoupledResource, MachineArcFunctionInput, PairedResourceMode};
    use baldes::Window;

    // The owner and its prepared domains are native. Pricing costs, fixing
    // cutoff and mode are explicitly controlled inputs; see the audit fixture.
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-signed-affine-prepared-owner.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let prepared = &captured["preparation"];
    let windows: Vec<_> = prepared["vertex_lower"]
        .as_array()
        .unwrap()
        .iter()
        .zip(prepared["vertex_upper"].as_array().unwrap())
        .map(|(lo, hi)| Window::new(lo.as_f64().unwrap(), hi.as_f64().unwrap()).unwrap())
        .collect();
    let mut solver = baldes::pricing::Solver::<2>::new(windows.len(), 1).unwrap();
    let sink = windows.len() - 1;
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[sink].elements.clear();
    for (vertex, window) in windows.iter().enumerate() {
        solver
            .graph
            .set_bounds(vertex, 0, window.lower, window.upper)
            .unwrap();
    }
    let arcs = prepared["arcs"].as_array().unwrap();
    let base: Vec<_> = arcs
        .iter()
        .map(|arc| {
            PreparedSlopeFunction::from_prepared_pairs(
                arc["functions"]["0x30"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    for (index, arc) in arcs.iter().enumerate() {
        assert_eq!(arc["id"].as_u64().unwrap() as usize, index);
        let cost = captured["physical_arc_reduced_costs"][index]
            .as_f64()
            .unwrap();
        let id = solver
            .graph
            .add_arc(
                arc["tail"].as_u64().unwrap() as usize,
                arc["head"].as_u64().unwrap() as usize,
                cost,
            )
            .unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = base[index].pairs()[0].0;
    }
    solver
        .graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let affine = |value, slope| MachineRootInput::Affine(RawAffine::new(value, slope).unwrap());
    let inputs: Vec<_> = arcs
        .iter()
        .enumerate()
        .map(|(physical_arc, arc)| {
            let dependent = &captured["dependent_affine"][physical_arc];
            MachineArcFunctionInput {
                physical_arc,
                tail_bounds: windows[arc["tail"].as_u64().unwrap() as usize],
                waiting_bounds: windows[arc["head"].as_u64().unwrap() as usize],
                waiting_rate: 0,
                base: &base[physical_arc],
                dependent: affine(
                    dependent[0].as_f64().unwrap(),
                    dependent[1].as_f64().unwrap(),
                ),
                dependent_lower: affine(0., 0.),
                dependent_upper: affine(10., 0.),
            }
        })
        .collect();
    let installed = solver.graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(0., 0.),
        windows[0],
        windows[sink],
        FrontierBudget::default(),
    );
    assert!(
        matches!(installed, FrontierOutcome::Complete(())),
        "{installed:?}"
    );
    solver
        .graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    solver.bucket_steps[0] = 1.;
    solver.complete_labels(true).unwrap();
    let cutoff = captured["cutoff"].as_f64().unwrap();
    let report = solver.fix(cutoff).unwrap();
    assert!(report.eliminated_forward > 0 && report.eliminated_backward > 0);
    let enumeration = Enumeration::run_directional(
        &solver.graph,
        &solver.engine.as_ref().unwrap().grid,
        &Controls {
            cutoff,
            ..Default::default()
        },
        |_, _| true,
    )
    .unwrap();
    let mut actual: Vec<Vec<_>> = enumeration
        .paths
        .partitions
        .iter()
        .flatten()
        .map(|route| route.arcs.iter().map(|&arc| arc % 10).collect())
        .collect();
    actual.sort();
    actual.dedup();
    let mut expected: Vec<Vec<_>> = captured["enumerated_paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|path| {
            path["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arc| arc.as_u64().unwrap() as usize % 10)
                .collect()
        })
        .collect();
    expected.sort();
    expected.dedup();
    assert_eq!(expected, vec![vec![0, 4, 8], vec![1, 8]]);
    assert_eq!(actual, expected);
}
