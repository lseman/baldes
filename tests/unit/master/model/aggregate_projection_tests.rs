use super::*;
use crate::master::model::{self, FixedRoute};
use std::sync::Arc;

fn point() -> Output<2> {
    let problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = model::Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.enumeration = None;
    options.gap_enumeration = false;
    options.reduced_cost_fixing = false;
    model::solve(problem, options).unwrap()
}

fn member(network: i32, id: i32, arcs: Vec<(i32, f64)>, integer: bool) -> OriginalPricingVariable {
    OriginalPricingVariable {
        network,
        id,
        cost: 0.,
        entries: vec![],
        arcs,
        accumulated_resource: None,
        integer,
        priority: 1.,
    }
}

fn raw(output: &mut Output<2>, arcs: Vec<i32>, quantities: &[f64]) -> (i32, i32) {
    let mut route = output.snapshot.state.routes[0].as_ref().clone();
    let network = route.network;
    let first = arcs.first().copied().unwrap_or(route.route.arcs[0]);
    route.route.arcs = arcs;
    Arc::make_mut(&mut output.snapshot.state.routes)[0] = Arc::new(route);
    output.master.column_tags = vec![0; quantities.len()];
    output.snapshot.master.values = quantities.to_vec();
    // The candidate must read physical master columns rather than this coalesced view.
    output.values.fill(0.);
    (network, first)
}

#[test]
fn aliases_are_gated_before_coalescing_and_fixed_copies_are_not_appended() {
    let mut output = point();
    let arc = output.snapshot.state.routes[0].route.arcs[0];
    let (network, _) = raw(&mut output, vec![arc], &[0.25, 0.75]);
    output.snapshot.state.fixed_routes.push(FixedRoute {
        route: output.snapshot.state.routes[0].as_ref().clone(),
        quantity: 1.,
    });
    let projection = AggregateProjection::new(vec![member(network, 3, vec![(arc, 1.)], true)]);
    assert_eq!(projection.value(&output, true), Some(1.));
    assert_eq!(projection.value(&output, false), Some(0.));
    output.snapshot.master.values = vec![1., 0.];
    assert_eq!(projection.value(&output, true), Some(0.));
    output.snapshot.master.values = vec![6e-7, 6e-7];
    assert_eq!(projection.value(&output, true), Some(0.));
    output.master.column_tags = vec![0];
    output.snapshot.master.values = vec![1.2e-6];
    assert_eq!(projection.value(&output, true), Some(1.2e-6));
}

#[test]
fn only_integer_members_and_fractional_column_quantities_contribute() {
    let mut output = point();
    let arc = output.snapshot.state.routes[0].route.arcs[0];
    let (network, _) = raw(&mut output, vec![arc], &[0., 1., 1e-8, 0.25, -0.25]);
    let integer = member(network, 3, vec![(arc, 2.)], true);
    let continuous = member(network, 4, vec![(arc, 17.)], false);
    assert_eq!(
        AggregateProjection::new(vec![integer.clone(), continuous.clone()]).value(&output, true),
        Some(0.)
    );
    assert_eq!(
        AggregateProjection::new(vec![continuous]).value(&output, true),
        Some(0.)
    );
    output.snapshot.master.values[4] = 0.25;
    assert_eq!(
        AggregateProjection::new(vec![integer]).value(&output, true),
        Some(1.)
    );
}

#[test]
fn arc_coefficients_materialize_before_quantity_multiplication() {
    let mut output = point();
    let existing = output.snapshot.state.routes[0].route.arcs.clone();
    assert!(existing.len() >= 2);
    let (a, b) = (existing[0], existing[1]);
    let (network, _) = raw(&mut output, vec![a, b, a], &[0.25]);
    // Last setter wins; route occurrence order gives (1e16 + .5) + 1e16.
    let v = member(network, 3, vec![(a, 7.), (a, 1e16), (b, 0.5)], true);
    assert_eq!(
        AggregateProjection::new(vec![v]).value(&output, true),
        Some(5e15)
    );
    let c = output
        .snapshot
        .state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .unwrap()
        .network
        .graph
        .arcs
        .iter()
        .map(|arc| arc.id)
        .find(|id| *id != a && *id != b)
        .unwrap();
    raw(&mut output, vec![a, b, c], &[0.25]);
    let v = member(network, 3, vec![(a, 1e16), (b, 0.5), (c, -1e16)], true);
    assert_eq!(
        AggregateProjection::new(vec![v]).value(&output, true),
        Some(0.)
    );
}

#[test]
fn registration_order_and_column_order_preserve_scalar_cancellation() {
    let mut output = point();
    let arc = output.snapshot.state.routes[0].route.arcs[0];
    let (network, _) = raw(&mut output, vec![arc], &[0.25]);
    let a = member(network, 30, vec![(arc, 4e16)], true);
    let b = member(network, 10, vec![(arc, 2.)], true);
    let c = member(network, 20, vec![(arc, -4e16)], true);
    assert_eq!(
        AggregateProjection::new(vec![a.clone(), b.clone(), c.clone()]).value(&output, true),
        Some(0.)
    );
    assert_eq!(
        AggregateProjection::new(vec![a, c, b]).value(&output, true),
        Some(0.5)
    );
    let ids: Vec<_> = output
        .snapshot
        .state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .unwrap()
        .network
        .graph
        .arcs
        .iter()
        .take(3)
        .map(|arc| arc.id)
        .collect();
    assert_eq!(ids.len(), 3);
    let base = output.snapshot.state.routes[0].as_ref().clone();
    let routes = Arc::make_mut(&mut output.snapshot.state.routes);
    routes.clear();
    for id in &ids {
        let mut route = base.clone();
        route.route.arcs = vec![*id];
        routes.push(Arc::new(route));
    }
    output.snapshot.master.values = vec![0.25; 3];
    output.master.column_tags = vec![0, 1, 2];
    let projection = AggregateProjection::new(vec![member(
        network,
        1,
        vec![(ids[0], 4e16), (ids[1], 2.), (ids[2], -4e16)],
        true,
    )]);
    assert_eq!(projection.value(&output, true), Some(0.));
    output.master.column_tags = vec![0, 2, 1];
    assert_eq!(projection.value(&output, true), Some(0.5));
}

#[test]
fn full_activity_keeps_continuous_and_integer_column_contributions() {
    let mut output = point();
    let arc = output.snapshot.state.routes[0].route.arcs[0];
    let (network, _) = raw(&mut output, vec![arc], &[3., 0.25]);
    let continuous = AggregateProjection::new(vec![member(network, 1, vec![(arc, 1.)], false)]);
    assert_eq!(continuous.value(&output, true), Some(0.));
    assert_eq!(continuous.full_value(&output), Some(3.25));
    let mixed = AggregateProjection::new(vec![
        member(network, 1, vec![(arc, 1.)], true),
        member(network, 2, vec![(arc, 2.)], false),
        member(network, 3, vec![(arc, 3.)], true),
    ]);
    assert_eq!(mixed.value(&output, true), Some(1.));
    assert_eq!(mixed.full_value(&output), Some(19.5));
}

#[test]
fn full_activity_matches_creation_order_scalar_cancellation_for_all_member_types() {
    let mut output = point();
    let arc = output.snapshot.state.routes[0].route.arcs[0];
    let (network, _) = raw(&mut output, vec![arc], &[0.25]);
    for integer in [
        [true, true, true],
        [true, false, true],
        [false, false, false],
    ] {
        let a = member(network, 30, vec![(arc, 4e16)], integer[0]);
        let b = member(network, 10, vec![(arc, 2.)], integer[1]);
        let c = member(network, 20, vec![(arc, -4e16)], integer[2]);
        assert_eq!(
            AggregateProjection::new(vec![a.clone(), b.clone(), c.clone()]).full_value(&output),
            Some(0.)
        );
        assert_eq!(
            AggregateProjection::new(vec![a, c, b]).full_value(&output),
            Some(0.5)
        );
    }
}

#[test]
fn full_activity_folds_columns_before_members_and_omits_fixed_copies() {
    let mut output = point();
    let base = output.snapshot.state.routes[0].as_ref().clone();
    let network = base.network;
    let ids: Vec<_> = output
        .snapshot
        .state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .unwrap()
        .network
        .graph
        .arcs
        .iter()
        .take(3)
        .map(|arc| arc.id)
        .collect();
    assert_eq!(ids.len(), 3);
    let routes = Arc::make_mut(&mut output.snapshot.state.routes);
    routes.clear();
    for id in &ids {
        let mut route = base.clone();
        route.route.arcs = vec![*id];
        routes.push(Arc::new(route));
    }
    let projection = AggregateProjection::new(vec![
        member(network, 1, vec![(ids[0], 4e16)], true),
        member(network, 2, vec![(ids[1], 2.)], true),
        member(network, 3, vec![(ids[2], -4e16)], true),
    ]);
    output.snapshot.master.values = vec![0.25; 3];
    output.master.column_tags = vec![0, 2, 1];
    assert_eq!(projection.value(&output, true), Some(0.));
    assert_eq!(projection.full_value(&output), Some(0.5));
    output.snapshot.state.fixed_routes.push(FixedRoute {
        route: base,
        quantity: 1.,
    });
    assert_eq!(projection.full_value(&output), Some(0.5));
    output.master.column_tags = vec![0, 1, 2];
    assert_eq!(projection.full_value(&output), Some(0.));
}

#[test]
fn full_activity_matches_all_original_runtime_scalar_fixtures() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        flags: String,
        ids: [i32; 3],
        order: [usize; 3],
        quantities: [f64; 3],
        coefficients: [[f64; 3]; 3],
        native_bits: u64,
    }
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("aggregate_full_lhs_fixtures.json")).unwrap();
    assert_eq!(fixtures.len(), 432);
    let mut output = point();
    let base = output.snapshot.state.routes[0].as_ref().clone();
    let network = base.network;
    let ids: Vec<_> = output
        .snapshot
        .state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .unwrap()
        .network
        .graph
        .arcs
        .iter()
        .take(3)
        .map(|arc| arc.id)
        .collect();
    assert_eq!(ids.len(), 3);
    let routes = Arc::make_mut(&mut output.snapshot.state.routes);
    routes.clear();
    // These are controlled stored-column expressions matching the original
    // computeLhs probe, not claims about feasible LP routes or registrations.
    for id in &ids {
        let mut route = base.clone();
        route.route.arcs = vec![*id];
        routes.push(Arc::new(route));
    }
    for (case, fixture) in fixtures.into_iter().enumerate() {
        let mut original_order = [0, 1, 2];
        original_order.sort_by_key(|&index| fixture.ids[index]);
        let members = original_order
            .into_iter()
            .map(|index| {
                member(
                    network,
                    fixture.ids[index],
                    ids.iter()
                        .enumerate()
                        .map(|(column, &arc)| (arc, fixture.coefficients[column][index]))
                        .collect(),
                    fixture.flags.as_bytes()[index] == b'I',
                )
            })
            .collect();
        output.master.column_tags = fixture.order.iter().map(|&index| index as i32).collect();
        output.snapshot.master.values = fixture
            .order
            .iter()
            .map(|&index| fixture.quantities[index])
            .collect();
        let actual = AggregateProjection::new(members)
            .full_value(&output)
            .unwrap();
        assert_eq!(
            actual.to_bits(),
            fixture.native_bits,
            "original runtime fixture {case}"
        );
    }
}
