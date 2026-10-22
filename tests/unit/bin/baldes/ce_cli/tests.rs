use super::*;

#[test]
fn invalid_backend_and_thread_arguments_fail_before_reading_input() {
    for args in [
        vec!["missing.vrp", "--solver", "unknown"],
        vec!["missing.vrp", "--master", "unknown"],
        vec!["missing.vrp", "--threads", "4"],
        vec!["missing.vrp", "--solver", "highs", "--threads", "-1"],
    ] {
        let error = run(args.into_iter().map(String::from).collect()).unwrap_err();
        assert!(
            error.contains("solver") || error.contains("threads"),
            "{error}"
        );
    }
}

#[test]
fn backend_alias_and_unsupported_cut_diagnostics() {
    let parse = |args: Vec<&str>| run(args.into_iter().map(String::from).collect()).unwrap_err();
    assert_eq!(
        parse(vec!["missing.vrp", "--master", "highs"]),
        parse(vec!["missing.vrp", "--solver", "highs"])
    );
    for flag in ["--rank1", "--capacity"] {
        let error = parse(vec!["missing.vrp", flag]);
        assert!(error.contains("cuts are not implemented"), "{error}");
        assert!(error.contains("capacity constraints are already enforced"));
    }
}
