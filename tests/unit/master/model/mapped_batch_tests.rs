use super::*;
use crate::master::model::{self, FixedRoute};

fn point() -> (Vec<MappedBranchVariable>, Output<2>) {
    let problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let variables = problem.arc_flow_variables(1.);
    let mut options = model::Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.enumeration = None;
    options.gap_enumeration = false;
    options.reduced_cost_fixing = false;
    let mut output = model::solve(problem, options).unwrap();
    let base = output.routes[0].clone();
    while output.routes.len() < 16 {
        output.routes.push(base.clone());
        output.values.push(0.);
    }
    assert!(variables.len() >= 8);
    assert!(output.routes.len() + output.snapshot.state.fixed_routes.len() >= 16);
    (variables, output)
}

fn assert_scalar<const D: usize>(variables: &[MappedBranchVariable], output: &Output<D>) {
    let expected: Vec<_> = variables
        .iter()
        .map(|variable| variable.value(output).map(f64::to_bits))
        .collect();
    let actual: Vec<_> = values(variables, output)
        .into_iter()
        .map(|value| value.map(f64::to_bits))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn shared_unit_arc_values_match_scalar_at_a_solved_point() {
    let (variables, output) = point();
    assert_scalar(&variables, &output);
}

#[test]
fn unit_arc_batch_preserves_repeated_arcs_fixed_copies_and_ieee_cancellation() {
    let (variables, mut output) = point();
    let base = output
        .routes
        .iter()
        .find(|route| route.network == 7)
        .unwrap()
        .clone();
    let arc = variables
        .iter()
        .find(|v| v.networks[0].network == 7)
        .unwrap()
        .networks[0]
        .arcs[0]
        .0;
    let other = variables
        .iter()
        .find(|v| v.networks[0].network == 7 && v.networks[0].arcs[0].0 != arc)
        .unwrap()
        .networks[0]
        .arcs[0]
        .0;
    for large in [
        0.,
        -0.,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.1,
        -0.1,
        1e16,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        // These are scalar activity fixtures, deliberately independent of the
        // LP feasibility of the edited paths and quantities.
        output.routes = vec![base.clone(); 16];
        output.values = vec![0.; 16];
        output.routes[0].route.arcs = vec![arc, arc, other];
        output.routes[1].route.arcs = vec![arc];
        output.routes[2].route.arcs = vec![arc, arc, other];
        output.values[..3].copy_from_slice(&[large, f64::from_bits(1), -large]);
        let mut fixed = base.clone();
        fixed.route.arcs = vec![arc];
        output.snapshot.state.fixed_routes = vec![FixedRoute {
            route: fixed,
            quantity: 0.25,
        }];
        assert_scalar(&variables, &output);
        if !large.is_finite() {
            for (variable, value) in variables.iter().zip(values(&variables, &output)) {
                if variable.networks[0].network == 7 {
                    assert!(value.is_none());
                } else {
                    assert_eq!(value, Some(0.));
                }
            }
        }
    }
}

#[test]
fn generic_and_small_registrations_keep_the_scalar_path() {
    let (mut variables, mut output) = point();
    let mut generic = variables[0].clone();
    generic.name = "weighted".into();
    generic.networks[0].arcs[0].1 = 2.;
    variables.push(generic.clone());
    generic.name = "negative".into();
    generic.networks[0].arcs[0].1 = -1.;
    variables.push(generic.clone());
    generic.name = "per-route".into();
    generic.networks[0].per_route = 1.;
    variables.push(generic.clone());
    generic.name = "several-networks".into();
    generic.networks.push(
        variables
            .iter()
            .find(|v| v.networks[0].network != generic.networks[0].network)
            .unwrap()
            .networks[0]
            .clone(),
    );
    variables.push(generic);
    let mut activity = variables[0].clone();
    activity.name = "activity".into();
    activity.networks[0].activities = vec![(0, 1.)];
    variables.push(activity);
    let mut master = variables[0].clone();
    master.name = "static-master".into();
    master.networks.clear();
    master.masters = vec![(901, 2.)];
    output.master_values.push((901, -0.125));
    variables.push(master);
    assert_scalar(&variables, &output);
    assert_scalar(&variables[..1], &output);
    assert_scalar(&[], &output);
    output.routes.truncate(4);
    output.values.truncate(4);
    assert_scalar(&variables, &output);
}

#[test]
fn unit_arc_batch_matches_seeded_signed_walk_activity_points() {
    let (variables, mut output) = point();
    let base = output
        .routes
        .iter()
        .find(|route| route.network == 7)
        .unwrap()
        .clone();
    let arcs: Vec<_> = variables
        .iter()
        .filter(|v| v.networks[0].network == 7)
        .map(|v| v.networks[0].arcs[0].0)
        .collect();
    let mut state = 20261101_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..256 {
        let large = f64::from_bits(((next() % 2046 + 1) << 52) | (next() & ((1_u64 << 52) - 1)));
        let small = f64::from_bits(next() & ((1_u64 << 63) - 1));
        output.routes = vec![base.clone(); 16];
        output.values = vec![0.; 16];
        output.values[..5].copy_from_slice(&[large, small, -large, 0.1, -0.1]);
        for route in output.routes.iter_mut().take(5) {
            route.route.arcs = (0..(next() % 17))
                .map(|_| arcs[next() as usize % arcs.len()])
                .collect();
        }
        output.snapshot.state.fixed_routes = vec![FixedRoute {
            route: base.clone(),
            quantity: 0.25,
        }];
        assert_scalar(&variables, &output);
    }
}

#[test]
fn skipped_mappings_keep_positions_and_do_not_affect_other_exact_values() {
    let (mut variables, mut output) = point();
    let mut generic = variables[0].clone();
    generic.name = "skip-nonfinite-generic".into();
    generic.networks[0].arcs[0].1 = f64::INFINITY;
    variables.insert(2, generic);
    variables[0].name = "skip-unit".into();
    let skip = |v: &MappedBranchVariable| v.name.starts_with("skip-");
    let expected: Vec<_> = variables
        .iter()
        .map(|v| {
            if skip(v) {
                None
            } else {
                v.value(&output).map(f64::to_bits)
            }
        })
        .collect();
    let actual: Vec<_> = values_except(&variables, &output, skip)
        .into_iter()
        .map(|v| v.map(f64::to_bits))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(actual[0], None);
    assert_eq!(actual[2], None);
    assert_eq!(
        values(&variables, &output)
            .into_iter()
            .map(|v| v.map(f64::to_bits))
            .collect::<Vec<_>>(),
        values_except(&variables, &output, |_| false)
            .into_iter()
            .map(|v| v.map(f64::to_bits))
            .collect::<Vec<_>>()
    );
    // Exercise both scalar fallbacks: tiny point and no eligible unit slots.
    output.routes.truncate(4);
    output.values.truncate(4);
    assert_eq!(values_except(&variables[..3], &output, skip)[0], None);
    assert_eq!(values_except(&variables[2..3], &output, skip), vec![None]);
    let (mut generic_only, full_output) = point();
    for variable in &mut generic_only {
        variable.networks[0].arcs[0].1 = 2.;
        variable.name = "skip-generic".into();
    }
    assert_eq!(
        values_except(&generic_only, &full_output, skip),
        vec![None; generic_only.len()]
    );
}
