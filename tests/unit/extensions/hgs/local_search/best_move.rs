use super::*;
use crate::inputs::{Customer, Instance, Rounding};

fn input() -> Instance {
    Instance {
        name: "best-customer".into(),
        vehicles: 3,
        capacity: 3.,
        customers: (0..=6)
            .map(|c| Customer {
                x: c as f64,
                y: 0.,
                demand: if c == 0 { 0. } else { 1. },
                ready: 0.,
                due: 1000.,
                service: 0.,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    }
}

fn configured(data: &Instance) -> Params {
    let mut params = Params::from_instance(data);
    params.config.local_search.best_move = true;
    params.config.nb_granular = 6;
    params.penalty_capacity.set(17.);
    params.penalty_time_warp.set(3.);
    params
}

fn links(ls: &HGSLocalSearch<'_>) -> Vec<(i32, i32, i32)> {
    ls.nodes
        .iter()
        .map(|n| (n.prev, n.next, n.route_idx))
        .collect()
}

#[test]
fn probing_all_move_families_preserves_links_and_matches_fractional_directed_replay() {
    let mut random = 918_u64;
    for trial in 0..12 {
        let mut data = input();
        data.distances = Some(
            (0..7)
                .map(|i| {
                    (0..7)
                        .map(|j| {
                            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                            if i == j {
                                0.
                            } else {
                                1. + ((random >> 32) % 100) as f64 / 8.
                            }
                        })
                        .collect()
                })
                .collect(),
        );
        for (c, node) in data.customers.iter_mut().enumerate().skip(1) {
            node.service = c as f64 / 8.;
            node.ready = (c % 3) as f64;
            node.due = 6. + ((c + trial) % 4) as f64 * 2.;
        }
        let mut params = configured(&data);
        params.config.local_search.max_deterioration = 1e6;
        let original = Individual::from_string(&params, "1 2 0 3 4 0 5 6");
        let mut ls = HGSLocalSearch::new(&params);
        ls.penalty_capacity_ls = params.penalty_capacity.get();
        ls.penalty_time_warp_ls = params.penalty_time_warp.get();
        let mut checked = [0; 10];
        for u in 1..=6 {
            for v in 1..=6 {
                ls.load_individual(&original);
                let a = ls.nodes[u].route_idx as usize;
                let b = ls.nodes[v].route_idx as usize;
                let mut candidates = vec![
                    (0, Move::Single(u, v)),
                    (1, Move::Pair(u, v)),
                    (2, Move::ReversedPair(u, v)),
                    (3, Move::Swap(u, v)),
                    (4, Move::PairSingle(u, v)),
                    (5, Move::PairPair(u, v)),
                ];
                if a == b {
                    candidates.push((6, Move::TwoOpt(u, v)));
                    candidates.push((6, Move::TwoOpt(ls.depot_start(a), v)));
                } else {
                    candidates.extend([
                        (7, Move::TwoOptStar(u, v)),
                        (8, Move::RelocateStar(u, b)),
                        (9, Move::SwapStar(u, b, true)),
                        (9, Move::SwapStar(u, b, false)),
                        (0, Move::Single(u, ls.depot_start(b))),
                    ]);
                }
                for (family, mv) in candidates {
                    ls.load_individual(&original);
                    ls.move_credit = Some(1e7);
                    let before = links(&ls);
                    let moves = ls.nb_moves;
                    let delta = ls.evaluate_move(mv, false);
                    assert_eq!(links(&ls), before, "probe changed links: {mv:?}");
                    assert_eq!(ls.nb_moves, moves);
                    assert_eq!(ls.move_credit, Some(1e7));
                    if let Some(expected) = delta {
                        assert_eq!(ls.evaluate_move(mv, true), Some(expected));
                        let mut after = original.clone();
                        ls.export_individual(&mut after);
                        let actual =
                            after.my_cost_sol.penalized_cost - original.my_cost_sol.penalized_cost;
                        assert!(
                            (actual - expected).abs() < 1e-7,
                            "{mv:?}: {actual} != {expected}"
                        );
                        let mut covered = after.chrom_t.clone();
                        covered.sort_unstable();
                        assert_eq!(covered, vec![1, 2, 3, 4, 5, 6]);
                        checked[family] += 1;
                    }
                }
            }
        }
        assert!(
            checked.iter().all(|&n| n > 0),
            "unexercised family: {checked:?}"
        );
    }
}

#[test]
fn first_pass_credit_is_earned_capped_spent_and_disabled_in_strict_passes() {
    let mut params = configured(&input());
    params.config.local_search.max_deterioration = 2.;
    let original = Individual::from_string(&params, "1 3 5 0 2 4 6");
    let mut ls = HGSLocalSearch::new(&params);
    ls.penalty_capacity_ls = params.penalty_capacity.get();
    ls.penalty_time_warp_ls = params.penalty_time_warp.get();
    ls.load_individual(&original);
    ls.move_credit = Some(0.);
    let gain = ls.evaluate_move(Move::Swap(5, 2), true).unwrap();
    assert_eq!(gain, -20.);
    assert_eq!(ls.move_credit, Some(20.));
    ls.move_credit = Some(0.);
    assert_eq!(ls.evaluate_move(Move::Swap(5, 2), false), None);
    ls.move_credit = Some(20.);
    assert_eq!(ls.evaluate_move(Move::Swap(5, 2), true), Some(20.));
    assert_eq!(ls.move_credit, Some(0.));
    assert_eq!(gain + 20., 0.);
    ls.evaluate_move(Move::Swap(5, 2), true).unwrap();
    ls.move_credit = None;
    assert_eq!(ls.evaluate_move(Move::Swap(5, 2), false), None);
}

#[test]
fn zero_delta_self_moves_are_rejected_even_with_positive_credit() {
    let params = configured(&input());
    let original = Individual::from_string(&params, "1 2 0 3 4 0 5 6");
    let mut ls = HGSLocalSearch::new(&params);
    ls.load_individual(&original);
    ls.move_credit = Some(1000.);
    for u in 1..=6 {
        for mv in [
            Move::Single(u, u),
            Move::Pair(u, u),
            Move::ReversedPair(u, u),
            Move::Swap(u, u),
            Move::PairSingle(u, u),
            Move::PairPair(u, u),
            Move::TwoOpt(u, u),
        ] {
            assert_eq!(ls.evaluate_move(mv, true), None, "{mv:?}");
        }
    }
}

#[test]
fn customer_selection_finds_the_best_of_competing_moves_before_applying() {
    let params = configured(&input());
    let original = Individual::from_string(&params, "1 3 5 0 2 4 6");
    let mut ls = HGSLocalSearch::new(&params);
    ls.penalty_capacity_ls = params.penalty_capacity.get();
    ls.penalty_time_warp_ls = params.penalty_time_warp.get();
    ls.load_individual(&original);
    let before = links(&ls);
    let (delta, mv) = ls.best_customer_move(5).unwrap();
    assert_eq!(links(&ls), before);
    for v in 1..=6 {
        for alternative in [Move::Single(5, v), Move::Swap(5, v), Move::PairSingle(5, v)] {
            if let Some(other) = ls.evaluate_move(alternative, false) {
                assert!(delta <= other + 1e-8);
            }
        }
    }
    assert_eq!(ls.evaluate_move(mv, true), Some(delta));
}

#[test]
fn education_with_credit_keeps_the_best_feasible_witness_and_is_deterministic() {
    let mut data = input();
    for c in 1..=6 {
        data.customers[c].due = 20. + c as f64;
    }
    let mut params = configured(&data);
    params.config.local_search.max_deterioration = 10.;
    params.set_seed(19);
    let mut a = Individual::from_string(&params, "1 3 5 0 2 4 6");
    let cost = a.my_cost_sol.distance;
    let mut b = a.clone();
    HGSLocalSearch::new(&params).run(&mut a, 17., 3.);
    params
        .rng
        .replace(crate::extensions::hgs::DefaultRng::new(19));
    HGSLocalSearch::new(&params).run(&mut b, 17., 3.);
    assert_eq!(a.chrom_r, b.chrom_r);
    assert!(a.is_feasible && a.my_cost_sol.distance <= cost);
    crate::extensions::hgs::solution::validate_plan(&data, &a.chrom_r).unwrap();
}
