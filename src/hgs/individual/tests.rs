use super::*;
use crate::inputs::Customer;
use crate::inputs::Instance;

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
fn test_random_individual() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let indiv = Individual::new_random(&params);

    assert_eq!(indiv.chrom_t.len(), 4); // 4 customers (depot excluded)
    assert!(indiv.chrom_t.iter().all(|&c| c >= 1 && c <= 4));
    assert_eq!(indiv.nb_vehicles, 3);
}

#[test]
fn test_from_string() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let indiv = Individual::from_string(&params, "1 0 2 3 4 0");

    assert_eq!(indiv.chrom_r[0], vec![1]);
    assert_eq!(indiv.chrom_r[1], vec![2, 3, 4]);
    assert_eq!(indiv.chrom_t, vec![1, 2, 3, 4]);
}

#[test]
fn test_evaluate_cost() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let indiv = Individual::from_string(&params, "1 0 2 3 4 0");

    assert!(indiv.my_cost_sol.nb_routes >= 1);
    assert!(indiv.my_cost_sol.distance > 0.);
}

#[test]
fn test_broken_pairs_distance() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);

    let mut indiv1 = Individual::new_random(&params);
    indiv1.evaluate_complete_cost();

    let mut indiv2 = Individual::new_random(&params);
    indiv2.evaluate_complete_cost();

    // Two different random individuals should have some broken pairs.
    let dist = indiv1.broken_pairs_distance(&indiv2);
    assert!(dist >= 0.0 && dist <= 2.0);
}

#[test]
fn test_shuffled_giant_tour_is_permutation() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let mut indiv = Individual::new_random(&params);

    indiv.shuffle_chrom_t();

    let mut sorted = indiv.chrom_t.clone();
    sorted.sort();
    assert_eq!(sorted, vec![1, 2, 3, 4]);
}
