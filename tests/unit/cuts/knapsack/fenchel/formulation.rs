use super::*;
use crate::master::{clp::ClpMaster, LpMaster, RowSense};

// Enumerate every integer packing independently of the production DP oracle.
// Include unobserved loads: the sparse formulation must still describe valid
// future-column coefficients for the complete capacity domain.
fn patterns(capacity: usize) -> Vec<Vec<usize>> {
    fn visit(next: usize, left: usize, current: &mut [usize], out: &mut Vec<Vec<usize>>) {
        if next > current.len() {
            if current.iter().any(|&count| count != 0) {
                out.push(current.to_vec());
            }
            return;
        }
        for count in 0..=left / next {
            current[next - 1] = count;
            visit(next + 1, left - count * next, current, out);
        }
        current[next - 1] = 0;
    }
    let mut result = Vec::new();
    visit(1, capacity, &mut vec![0; capacity], &mut result);
    result
}

fn full_pattern_objective(demand: &[f64], all: &[Vec<usize>]) -> f64 {
    let mut lp = ClpMaster::new().unwrap();
    lp.set_tolerances(1e-9, 1e-9).unwrap();
    for &rhs in demand {
        lp.add_row(RowSense::GreaterEqual, rhs, &[]).unwrap();
    }
    for pattern in all {
        let entries: Vec<_> = pattern
            .iter()
            .enumerate()
            .filter_map(|(i, &count)| (count != 0).then_some((i, count as f64)))
            .collect();
        lp.add_column(1., &entries).unwrap();
    }
    assert!(lp.solve());
    lp.objective()
}

fn check_case(capacity: usize, demand: &[f64], expected_native: Option<f64>) {
    let all = patterns(capacity);
    let objective = full_pattern_objective(demand, &all);
    if let Some(expected) = expected_native {
        assert!((objective - expected).abs() <= 1e-8 * expected.abs().max(1.));
    }
    let scale = 1_000_000;
    let parameters = FenchelParameters {
        max_capacity: capacity,
        coefficient_scale: scale,
        time_limit: Duration::from_secs(30),
        ..FenchelParameters::default()
    };
    let tolerance = demand.iter().sum::<f64>() / f64::from(scale) + 1e-7;
    let loads: Vec<_> = (1..=capacity as i32).collect();
    let result = FenchelSeparator::with_parameters(parameters)
        .unwrap()
        .separate(&loads, demand, capacity as i32)
        .unwrap();
    match result.cut {
        Some(cut) => {
            assert_eq!(result.status, FenchelStatus::Separated);
            assert!(cut.rhs > 0);
            let score = cut.lhs / f64::from(cut.rhs);
            assert!(
                score <= objective + 1e-7 && score >= objective - tolerance,
                "B={capacity}, demand={demand:?}, score={score}, optimum={objective}"
            );
            for pattern in &all {
                let activity: i64 = pattern
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| n as i64 * i64::from(cut.coefficient(i as i32 + 1)))
                    .sum();
                assert!(activity <= i64::from(cut.rhs));
            }
        }
        None => {
            assert_eq!(result.status, FenchelStatus::NoCutFound);
            assert!(
                objective <= 1. + tolerance + 1e-6,
                "missed violated auxiliary: B={capacity}, demand={demand:?}, optimum={objective}"
            );
        }
    }
}

#[test]
fn full_integer_pattern_lp_checks_sparse_separation_and_quantized_validity() {
    // 220 deterministic dense, sparse, degenerate and zero-demand fixtures.
    // The objective oracle materializes every column; it does not call either
    // of the production pattern or certification DPs.
    let mut seed = 0x8a_53_17_u64;
    for capacity in 2..=12 {
        for case in 0..20 {
            let demand: Vec<_> = (1..=capacity)
                .map(|load| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if case == 0 || (case % 2 == 0 && (seed >> 32) % 3 != 0) {
                        0.
                    } else if case == 1 {
                        1. / (capacity / load) as f64 / capacity as f64
                    } else {
                        ((seed >> 32) % 36) as f64 / 20.
                    }
                })
                .collect();
            check_case(capacity, &demand, None);
        }
    }
}

#[test]
fn native_auxiliary_exact_oracle_fixture_agrees_with_full_pattern_lp() {
    // Native preparation/master/CG with a supplied exact pricing oracle,
    // not a successful run of the library's unsupported Clp integer backend.
    // Source-pinned runtime record: native-fenchel-followup-20261023.
    check_case(4, &[0., 0.1, 1.2, 0.], Some(1.25));

    #[derive(serde::Deserialize)]
    struct RecordedCase {
        capacity: usize,
        demand: Vec<f64>,
        native_objective: f64,
        native_duals: Vec<f64>,
    }
    #[derive(serde::Deserialize)]
    struct Recording {
        binary_sha256: String,
        runtime: String,
        source_record_sha256: String,
        cases: Vec<RecordedCase>,
    }
    let recording: Recording =
        serde_json::from_str(include_str!("auxiliary-runtime-cases.json")).unwrap();
    assert_eq!(
        recording.binary_sha256,
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(
        recording.source_record_sha256,
        "4e56bf473fbbbf3e407dfb52406d0d4dfece75a77f49cc3f7365a3d23ac3168c"
    );
    assert!(recording.runtime.contains("explicit exact-pattern oracle"));
    assert_eq!(recording.cases.len(), 220);
    for case in recording.cases {
        assert_eq!(case.demand.len(), case.capacity);
        assert_eq!(case.native_duals.len(), case.capacity);
        assert!(case.native_duals.iter().all(|&a| a >= -1e-9));
        let score: f64 = case
            .demand
            .iter()
            .zip(&case.native_duals)
            .map(|(d, a)| d * a)
            .sum();
        assert!((score - case.native_objective).abs() < 1e-8);
        for pattern in patterns(case.capacity) {
            let activity: f64 = pattern
                .iter()
                .zip(&case.native_duals)
                .map(|(&n, &a)| n as f64 * a)
                .sum();
            assert!(activity <= 1. + 1e-8);
        }
        check_case(case.capacity, &case.demand, Some(case.native_objective));
    }
}
