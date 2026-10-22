use super::*;
use crate::inputs::Customer;

fn make_test_instance() -> Instance {
    // Simple 5-customer instance (index 0 = depot)
    Instance {
        name: "test-5".to_string(),
        vehicles: 3,
        capacity: 100.0,
        customers: vec![
            Customer {
                x: 0.0,
                y: 0.0,
                demand: 0.0,
                ready: 0.0,
                due: i32::MAX as f64,
                service: 0.0,
            }, // depot
            Customer {
                x: 10.0,
                y: 0.0,
                demand: 10.0,
                ready: 0.0,
                due: 1000.0,
                service: 5.0,
            },
            Customer {
                x: 20.0,
                y: 0.0,
                demand: 20.0,
                ready: 0.0,
                due: 1000.0,
                service: 10.0,
            },
            Customer {
                x: 30.0,
                y: 0.0,
                demand: 15.0,
                ready: 0.0,
                due: 1000.0,
                service: 5.0,
            },
            Customer {
                x: 40.0,
                y: 0.0,
                demand: 25.0,
                ready: 0.0,
                due: 1000.0,
                service: 10.0,
            },
        ],
        rounding: crate::inputs::Rounding::None,
        distances: None,
    }
}

#[test]
fn create_params_from_instance() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);

    assert_eq!(params.nb_clients, 4);
    assert_eq!(params.nb_vehicles, 3);
    assert_eq!(params.vehicle_capacity, 100);
    assert_eq!(params.cli.len(), 5); // including depot
    assert_eq!(params.cli[0].cust_num, 0); // depot
    assert_eq!(params.time_cost.size(), 5);
}

#[test]
fn params_time_limit() {
    let instance = make_test_instance();
    let mut params = Params::from_instance(&instance);

    // No time limit → never exceeded
    assert!(!params.is_time_limit_exceeded());

    // Zero time limit → treated as infinity
    params.config.time_limit = 0.0;
    assert!(!params.is_time_limit_exceeded());
}

#[test]
fn penalty_update_increases_on_violation() {
    let instance = make_test_instance();
    let mut params = Params::from_instance(&instance);

    let old_cap = params.penalty_capacity.get();
    params.update_penalties(0, 10, 5, 0.); // infeasible due to capacity

    assert!(
        params.penalty_capacity.get() > old_cap,
        "capacity penalty should increase when there's excess load"
    );
}

#[test]
fn penalty_update_decreases_when_feasible() {
    let instance = make_test_instance();
    let mut params = Params::from_instance(&instance);

    let old_cap = params.penalty_capacity.get();
    // All feasible, above target ratio → decrease penalty
    params.update_penalties(10, 0, 0, 0.);

    assert!(
        params.penalty_capacity.get() < old_cap,
        "capacity penalty should decrease when all solutions are feasible"
    );
}

#[test]
fn savings_list_sorted_descending() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);

    for i in 1..params.savings_list.len() {
        assert!(
            params.savings_list[i].value <= params.savings_list[i - 1].value,
            "savings list should be sorted descending"
        );
    }
}

#[test]
fn rng_matches_cpp_reference() {
    // Values from BALDES's C++ math/RNG.h and libstdc++ 16 std::shuffle.
    let mut rng = Xoroshiro128Plus::new(0);
    assert_eq!(
        [rng.next(), rng.next(), rng.next()],
        [
            2836985337983649843,
            17383111449422238058,
            15852657771511707500
        ]
    );
    assert_eq!(rng.next_u64(100), 86);
    let mut shuffled: Vec<Vec<usize>> = [1, 2, 10].iter().map(|&n| (0..n).collect()).collect();
    for v in &mut shuffled {
        rng.shuffle(v);
    }
    assert_eq!(
        shuffled,
        [vec![0], vec![1, 0], vec![7, 5, 9, 2, 3, 4, 1, 0, 8, 6]]
    );
}

#[test]
fn polar_angles_computed() {
    let mut instance = make_test_instance();
    for (customer, (x, y)) in
        instance
            .customers
            .iter_mut()
            .skip(1)
            .zip([(10., 0.), (0., 10.), (-10., 0.), (0., -10.)])
    {
        customer.x = x;
        customer.y = y;
    }
    let params = Params::from_instance(&instance);
    let angles: Vec<_> = params.cli.iter().map(|client| client.polar_angle).collect();
    assert_eq!(angles, [0, 0, 16384, 32768, 49152]);
}
