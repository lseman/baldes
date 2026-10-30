use baldes::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, RawAffine,
};
use baldes::graph::coupled::{
    CoupledResource, MachineArcFunctionInput, PairedResourceMode, PreparedSlopeFunction,
};
use baldes::{graph::Window, Graph};

fn main() -> baldes::Result<()> {
    let mut graph = Graph::<2>::new(3, 1)?;
    let bounds = Window::new(0., 4.)?;
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 4.)?;
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let arc = graph.add_arc(tail, head, -1.)?;
        graph.arc_mut(arc)?.consumption[0] = 1.;
    }
    graph.set_coupled_resources(&[CoupledResource::PairedFront {
        mode: PairedResourceMode::CostUnbounded,
    }])?;
    graph.set_paired_waiting_windows_from_resource(0, 0)?;
    let base = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 4.)]).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = (0..2)
        .map(|physical_arc| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: bounds,
            waiting_bounds: bounds,
            waiting_rate: 0,
            base: &base,
            dependent: affine(0.25),
            dependent_lower: affine(0.),
            dependent_upper: affine(8.),
        })
        .collect();
    let domains = match graph.preprocess_paired_base_domains(0, &inputs, FrontierBudget::default())
    {
        FrontierOutcome::Complete(domains) => domains,
        other => panic!("raw base preprocessing unavailable: {other:?}"),
    };
    assert_eq!(domains.vertex_bounds()[1], Window::new(1., 3.)?);
    assert!(matches!(
        domains.install_front_program(&mut graph, &inputs, affine(0.), FrontierBudget::default(),),
        FrontierOutcome::Complete(())
    ));
    println!("base vertex domains: {:?}", domains.vertex_bounds());
    Ok(())
}
