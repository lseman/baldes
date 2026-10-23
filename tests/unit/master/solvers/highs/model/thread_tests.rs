use super::*;

fn native_integer(master: &HighsMaster, name: &[u8]) -> HighsInt {
    let (mut current, mut minimum, mut maximum, mut default_value) = (0, 0, 0, 0);
    // SAFETY: borrowed live handle, NUL-terminated static option name,
    // and valid scalar output pointers.
    unsafe {
        HighsMaster::check(
            Highs_getIntOptionValues(
                master.highs.as_ptr(),
                name.as_ptr().cast(),
                &mut current,
                &mut minimum,
                &mut maximum,
                &mut default_value,
            ),
            "read configured option",
        )
        .unwrap();
    }
    current
}

#[test]
fn explicit_and_all_cpu_options_reach_native_solver_and_survive_clone() {
    for requested in [1, 4, 0] {
        let master = HighsMaster::with_threads(requested).unwrap();
        let expected = if requested == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            requested
        };
        assert_eq!(master.configured_threads(), Some(expected));
        assert_eq!(native_integer(&master, b"threads\0") as usize, expected);
        assert_eq!(
            native_integer(&master, b"simplex_strategy\0"),
            if expected > 1 { 3 } else { 1 }
        );
        let concurrency = master.simplex_concurrency().unwrap();
        assert!(concurrency > 0 && concurrency <= expected);
        assert_eq!(
            native_integer(&master, b"simplex_max_concurrency\0") as usize,
            concurrency
        );
        let copy = master.try_clone().unwrap();
        assert_eq!(copy.configured_threads(), master.configured_threads());
        assert_eq!(native_integer(&copy, b"threads\0"), expected as HighsInt);
    }
    assert!(HighsMaster::with_threads(usize::MAX).is_err());
}

#[test]
fn adaptive_worker_budget_preserves_native_algorithm_choice_in_forks() {
    for requested in [1, 4, 0] {
        let master = HighsMaster::with_worker_budget(requested).unwrap();
        let expected = if requested == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            requested
        };
        assert_eq!(native_integer(&master, b"threads\0") as usize, expected);
        assert_eq!(native_integer(&master, b"simplex_strategy\0"), 0);
        let copy = master.try_clone().unwrap();
        assert_eq!(native_integer(&copy, b"threads\0") as usize, expected);
        assert_eq!(native_integer(&copy, b"simplex_strategy\0"), 0);
    }
}
