use super::*;
use crate::extensions::hgs::Params;
use crate::inputs::{Customer, Instance};

fn make_test_instance() -> Instance {
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
            },
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
fn test_split_basic() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let mut indiv = Individual::new_random(&params);

    let mut split = Split::new(&params);
    split.general_split(&mut indiv, 3);

    // After split, routes should be populated.
    assert!(indiv.chrom_r.iter().any(|r| !r.is_empty()));
}

#[test]
fn test_split_preserves_all_customers() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let mut indiv = Individual::new_random(&params);

    let mut split = Split::new(&params);
    split.general_split(&mut indiv, 3);

    // All customers should appear exactly once across all routes.
    let mut all_customers: Vec<usize> = Vec::new();
    for route in &indiv.chrom_r {
        all_customers.extend(route);
    }
    all_customers.sort();
    assert_eq!(all_customers, vec![1, 2, 3, 4]);
}

#[test]
fn test_trivial_deque() {
    let mut dq = TrivialDeque::new(10, 0);
    assert_eq!(dq.size(), 1);
    assert_eq!(dq.get_front(), 0);

    dq.push_back(1);
    dq.push_back(2);
    assert_eq!(dq.size(), 3);
    assert_eq!(dq.get_front(), 0);
    assert_eq!(dq.get_back(), 2);

    dq.pop_front();
    assert_eq!(dq.size(), 2);
    assert_eq!(dq.get_front(), 1);

    dq.pop_back();
    assert_eq!(dq.size(), 1);
    assert_eq!(dq.get_back(), 1);
}
