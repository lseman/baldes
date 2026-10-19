use super::*;

fn problem(costs: [[f64; 2]; 2]) -> Problem {
    let mut networks = vec![];
    for (id, costs) in costs.into_iter().enumerate() {
        let mut graph = Graph::new(4, 1).unwrap();
        graph.element_count = 2;
        graph.elementary = true;
        graph.main_resource = Some(0);
        for vertex in 0..4 {
            graph.vertices[vertex].elements.clear();
            graph.vertices[vertex].memory.clear();
            graph.set_bounds(vertex, 0, 0., 1.).unwrap();
        }
        for customer in [1, 2] {
            graph.vertices[customer].elements = vec![customer - 1];
            graph.vertices[customer].memory = vec![0, 1];
        }
        for (tail, head, cost) in [(0, 1, costs[0]), (1, 3, 0.), (0, 2, costs[1]), (2, 3, 0.)] {
            let arc = graph.add_arc(tail, head, cost).unwrap();
            graph.arc_mut(arc).unwrap().consumption[0] = f64::from(head != 3);
        }
        networks.push(Network {
            id: id as i32 + 1,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![(1, 1.)], vec![]],
            route_entries: vec![],
        });
    }
    Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks,
    }
}

#[test]
fn shared_covering_rows_choose_routes_from_both_networks() {
    for costs in [[[1., 4.], [3., 1.]], [[4., 1.], [1., 3.]]] {
        let output = solve(problem(costs), Options::default()).unwrap();
        assert_eq!(output.master.status, cg::Status::Optimal);
        assert_eq!(output.master.lp, 2.);
        assert_eq!(output.master.bound, 2.);
        let mut covered = [0.; 2];
        let mut used = [0.; 2];
        let mut value = 0.;
        for (priced, &quantity) in output.routes.iter().zip(&output.values) {
            assert_eq!(priced.route.nodes.len(), 3);
            let customer = priced.route.nodes[1] as usize - 1;
            covered[customer] += quantity;
            used[priced.network as usize - 1] += quantity;
            value += costs[priced.network as usize - 1][customer] * quantity;
        }
        assert_eq!(covered, [1., 1.]);
        assert_eq!(used, [1., 1.]);
        // Cover duals (1,1), with zero multiplicity duals, certify value 2
        // against every one of the four possible paths.
        assert!(costs.into_iter().flatten().all(|cost| cost >= 1.));
        assert_eq!(value, 2.);
        assert!(output.oracle_calls.iter().all(|(_, calls)| *calls > 0));
    }
}

#[test]
fn unequal_fractional_multiplicities_have_a_valid_shared_bound() {
    let mut input = problem([[0., 0.], [5., 5.]]);
    input.networks[0].multiplicity.upper = 0.25;
    input.networks[1].multiplicity.upper = 1.75;
    let output = solve(input, Options::default()).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    // Two units must be covered and at most .25 can use a zero-cost path.
    assert_eq!(output.master.lp, 8.75);
    assert_eq!(output.master.bound, 8.75);
}

#[test]
fn required_networks_and_duplicate_local_arc_ids_remain_distinct() {
    let mut input = problem([[2., 4.], [3., 1.]]);
    for network in &mut input.networks {
        network.multiplicity.lower = 1.;
    }
    let output = solve(input, Options::default()).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 3.);
    assert!(output.master.bound <= 3. && 3. - output.master.bound < 1e-12);
    assert!(output.routes.iter().any(|route| route.network == 1));
    assert!(output.routes.iter().any(|route| route.network == 2));
}

#[test]
fn finite_penalties_do_not_certify_original_infeasibility() {
    let output = solve(problem([[1e8, 1e8], [1e8, 1e8]]), Options::default()).unwrap();
    assert_eq!(output.master.status, cg::Status::Stalled);
    assert!(output.master.bound <= 2e8);
}

#[test]
fn invalid_network_mappings_and_identities_are_rejected() {
    let mut input = problem([[1., 4.], [3., 1.]]);
    input.networks[1].id = input.networks[0].id;
    assert!(solve(input, Options::default()).is_err());
    let mut input = problem([[1., 4.], [3., 1.]]);
    input.networks[0].arc_entries[0] = vec![(2, 1.)];
    assert!(solve(input, Options::default()).is_err());
    let mut controls = Options::default();
    controls.cg.artificial_cost = 0.;
    assert!(solve(problem([[1., 4.], [3., 1.]]), controls).is_err());
}

#[test]
fn a_disabled_network_does_not_contribute_routes_or_bounds() {
    let mut input = problem([[0., 0.], [3., 1.]]);
    input.networks[0].multiplicity.upper = 0.;
    input.networks[1].multiplicity.upper = 2.;
    let output = solve(input, Options::default()).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 4.);
    assert_eq!(output.master.bound, 4.);
    assert_eq!(output.oracle_calls[0], (1, 0));
    assert!(output.routes.iter().all(|route| route.network == 2));
}

#[test]
fn shared_route_coefficients_enforce_network_use() {
    let mut input = problem([[1., 2.], [4., 1.]]);
    input.rows.push(Row {
        sense: RowSense::Equal,
        rhs: 4.,
    });
    for (index, network) in input.networks.iter_mut().enumerate() {
        network.multiplicity.upper = 2.;
        network.route_entries = vec![(2, if index == 0 { 2. } else { 1. })];
    }
    let output = solve(input, Options::default()).unwrap();
    // Covering requires two routes, and 2*q1+q2=4 forces both into network 1.
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 3.);
    assert!(output.master.bound <= 3. && 3. - output.master.bound < 1e-12);
    let mut used = [0.; 2];
    for (route, &quantity) in output.routes.iter().zip(&output.values) {
        used[route.network as usize - 1] += quantity;
    }
    assert_eq!(used, [2., 0.]);
}

#[test]
fn an_active_lower_multiplicity_forces_an_expensive_route() {
    let mut input = problem([[0., 0.], [5., 6.]]);
    for network in &mut input.networks {
        network.multiplicity = Window::new(1., 2.).unwrap();
    }
    let output = solve(input, Options::default()).unwrap();
    // Network 2 must supply at least one route, costing at least 5. Taking
    // its A route and network 1's B route attains that bound.
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 5.);
    assert!(output.master.bound <= 5. && 5. - output.master.bound < 1e-12);
}

#[test]
fn a_negative_walk_cycle_leaves_the_certificate_unknown() {
    let mut input = problem([[1., 4.], [3., 1.]]);
    let network = &mut input.networks[0];
    for (tail, head) in [(1, 2), (2, 1)] {
        let id = network.graph.add_arc(tail, head, -10.).unwrap();
        network.graph.arc_mut(id).unwrap().consumption[0] = 1.;
        network.arc_entries.push(vec![]);
    }
    // Resource upper 1 excludes both new arcs after reaching a customer.
    // The resource-free certificate relaxation sees the negative cycle.
    let output = solve(input, Options::default()).unwrap();
    assert_eq!(output.master.lp, 2.);
    assert_eq!(output.master.status, cg::Status::Stalled);
    assert_eq!(output.master.bound, f64::NEG_INFINITY);
}

#[test]
fn generated_assignment_models_have_strict_original_lower_bounds() {
    for seed in 0..16 {
        let cost = |shift| ((seed * (shift + 3) + shift * 7) % 97) as f64 / 8.;
        let costs = [[cost(1), cost(2)], [cost(3), cost(4)]];
        // Enumerate the two feasible perfect assignments. The bipartite
        // assignment LP is integral, so this also gives its original LP optimum.
        let optimum = (costs[0][0] + costs[1][1]).min(costs[0][1] + costs[1][0]);
        let output = solve(problem(costs), Options::default()).unwrap();
        assert_eq!(output.master.status, cg::Status::Optimal, "seed {seed}");
        assert!((output.master.lp - optimum).abs() < 1e-9, "seed {seed}");
        assert!(
            output.master.bound <= optimum,
            "seed {seed}: {:?}",
            output.master
        );
        assert!(optimum - output.master.bound < 1e-9, "seed {seed}");
    }
}
