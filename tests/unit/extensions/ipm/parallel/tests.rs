use super::*;
#[test]
fn mode_semantics_and_thread_budget() {
    for kind in [ParallelType::Tree, ParallelType::Node, ParallelType::Both] {
        let off = Execution::new(Parallel::Off, kind, 4).unwrap();
        assert_eq!(off.levels(true, true), (false, false));
        assert!(off.pool.is_none());
        let choose = Execution::new(Parallel::Choose, kind, 2).unwrap();
        assert_eq!(choose.levels(true, false), (true, false));
        assert_eq!(choose.levels(false, true), (false, true));
        assert_eq!(choose.pool.unwrap().current_num_threads(), 2);
        let single = Execution::new(Parallel::On, kind, 1).unwrap();
        assert_eq!(single.levels(true, true), (false, false));
    }
}
