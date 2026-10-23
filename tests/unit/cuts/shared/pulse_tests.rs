use super::*;

#[test]
fn bucket_conversion_matches_floor_at_boundaries_and_extremes() {
    let mut graph = PulseDigraph::new(10);
    graph.source = 0;
    graph.ready_time = vec![7.];
    for width in [0., 0.5, 1., 17., f64::INFINITY, f64::NAN] {
        graph.bucket_width = width;
        let mut times = vec![f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -1e100, 1e100];
        for index in -20..=20 {
            for offset in [-1e-5, -1e-7, 0., 1e-7, 1e-5] {
                times.push(7. + index as f64 * width + offset);
            }
        }
        for time in times {
            let expected = ((time - 7. + 1e-6) / width).floor() as i32 as usize;
            assert_eq!(graph.bucket(time), expected, "time={time}, width={width}");
        }
    }
}

fn fixture() -> PulseDigraph {
    let mut graph = PulseDigraph::new(10);
    assert!(graph.set_ready_time(vec![0., 0., 5., 12., 0., 0.]));
    assert!(graph.set_due_date(vec![6., 10., 18., 19., 30., 30.]));
    let distance = (0usize..6)
        .map(|i| {
            (0usize..6)
                .map(|j| {
                    if i == j {
                        0.
                    } else {
                        3. + i.abs_diff(j) as f64
                    }
                })
                .collect()
        })
        .collect();
    assert!(graph.set_distance_matrix(distance));
    graph
}

// Independent exhaustive route feasibility for this small, finite fixture.
fn feasible(graph: &PulseDigraph, subset: &[i32], node: usize, time: f64) -> bool {
    if subset.is_empty() {
        return time + graph.distance[node][graph.sink as usize]
            < graph.due_date[graph.sink as usize] + 1e-6;
    }
    for (index, &next) in subset.iter().enumerate() {
        let arrival =
            (time + graph.distance[node][next as usize]).max(graph.ready_time[next as usize]);
        if arrival > graph.due_date[next as usize] + 1e-6 {
            continue;
        }
        let mut remaining = subset.to_vec();
        remaining.remove(index);
        if feasible(graph, &remaining, next as usize, arrival) {
            return true;
        }
    }
    false
}

#[test]
fn repeated_pulse_subsets_match_fresh_search_and_exhaustive_feasibility() {
    let initial = fixture();
    let mut reused = initial.clone();
    // Reverse the second pass to exercise changing subsets and old bucket rows.
    let masks: Vec<_> = (1..16).chain((1..16).rev()).collect();
    for mask in masks {
        let subset: Vec<i32> = (0..4).filter(|i| mask & (1 << i) != 0).collect();
        let mut fresh = initial.clone();
        let expected = feasible(&initial, &subset, initial.source as usize, 0.);
        assert_eq!(
            reused.is_subset_time_feasible(&subset),
            expected,
            "mask={mask}"
        );
        assert_eq!(
            fresh.is_subset_time_feasible(&subset),
            expected,
            "mask={mask}"
        );
        assert_eq!(reused.num_pulses, fresh.num_pulses);
        assert_eq!(reused.bound, fresh.bound);
    }
    // Changing bucket count must not expose stale bounds.
    for buckets in [3, 12, 1] {
        reused.num_buckets = buckets;
        reused.prepare();
        let mut fresh = reused.clone();
        fresh.bound.clear();
        assert_eq!(reused.run(&[1, 2]), fresh.run(&[1, 2]));
        assert_eq!(reused.bound, fresh.bound);
    }
}

#[test]
fn changed_windows_and_distances_restore_newly_feasible_arcs() {
    fn prepared(ready: Vec<f64>, due: Vec<f64>, distance: Vec<Vec<f64>>) -> PulseDigraph {
        let mut graph = PulseDigraph::new(10);
        assert!(graph.set_ready_time(ready));
        assert!(graph.set_due_date(due));
        assert!(graph.set_distance_matrix(distance));
        graph
    }
    let final_ready = vec![0.; 3];
    let final_due = vec![10., 100., 100.];
    let final_distance = vec![vec![0., 2., 2.], vec![2., 0., 2.], vec![2., 2., 0.]];
    for change in 0..3 {
        let mut ready = final_ready.clone();
        let mut due = final_due.clone();
        let mut distance = final_distance.clone();
        match change {
            0 => due[0] = 1.,
            1 => ready[1] = 20.,
            _ => distance[1][0] = 20.,
        }
        let mut reused = prepared(ready, due, distance);
        assert!(!reused.is_subset_time_feasible(&[0]));
        let row = reused.arc_cost[0].as_ptr();
        match change {
            0 => assert!(reused.set_due_date(final_due.clone())),
            1 => assert!(reused.set_ready_time(final_ready.clone())),
            _ => assert!(reused.set_distance_matrix(final_distance.clone())),
        }
        let mut fresh = prepared(
            final_ready.clone(),
            final_due.clone(),
            final_distance.clone(),
        );
        assert_eq!(reused.arc_cost[0].as_ptr(), row);
        assert_eq!(reused.arc_cost, fresh.arc_cost);
        assert_eq!(reused.arc_cost[1][0], -1.);
        assert!(feasible(&fresh, &[0], fresh.source as usize, 0.));
        assert!(reused.is_subset_time_feasible(&[0]));
        assert_eq!(reused.run(&[0]), fresh.run(&[0]));
    }
}
