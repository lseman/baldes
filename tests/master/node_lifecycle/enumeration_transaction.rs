use baldes::master::{
    cg::{self, Answer, Column, Controller, LpMaster, Options, Status},
    RowSense,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    Preparation,
    Regeneration,
    BackendInsertion,
    ColumnCap,
    Remapping,
    Synchronization,
    None,
}

struct Transition {
    failure: Failure,
    done: bool,
    routes: Vec<i32>,
    mapping: Vec<Option<usize>>,
    saved: Option<(bool, Vec<i32>, Vec<Option<usize>>)>,
    rollbacks: usize,
    commits: usize,
}

fn column(tag: i32) -> Column {
    Column {
        cost: 10. * (tag + 1) as f64,
        reduced_cost: 10. * tag as f64,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag,
        identity: format!("route-{tag}"),
    }
}

impl<M: LpMaster> Controller<M> for Transition {
    fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
        if self.done {
            return Ok(false);
        }
        self.saved = Some((self.done, self.routes.clone(), self.mapping.clone()));
        Ok(true)
    }
    fn finish_enumeration_transaction(&mut self, commit: bool) {
        let saved = self.saved.take().unwrap();
        if commit {
            self.commits += 1;
        } else {
            self.rollbacks += 1;
            (self.done, self.routes, self.mapping) = saved;
        }
    }
    fn enumerated_universe(&mut self, _: &[f64], _: f64) -> baldes::Result<Option<Vec<Column>>> {
        self.done = true;
        self.routes.extend([1, 2]);
        if self.failure == Failure::Preparation {
            return Err(baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "injected enumeration timeout".into(),
            });
        }
        let mut universe = vec![column(0), column(1), column(2)];
        if self.failure == Failure::BackendInsertion {
            // Both the regenerated route and one new route are inserted on
            // the private LP before this backend validation failure.
            universe[2].cost = f64::NAN;
        }
        Ok(Some(universe))
    }
    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[Column],
        _: &[f64],
        _: f64,
        _: &[Column],
    ) -> baldes::Result<Vec<Column>> {
        if self.failure == Failure::Regeneration {
            return Err(baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "injected regeneration failure".into(),
            });
        }
        Ok(pool.to_vec())
    }
    fn column_upper_bound(&self, tag: i32) -> Option<f64> {
        (self.failure == Failure::ColumnCap && tag == 2).then_some(f64::NAN)
    }
    fn remap_columns(&mut self, map: &[Option<usize>]) -> baldes::Result<()> {
        self.mapping = map.to_vec();
        if self.failure == Failure::Remapping {
            return Err(baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "injected remap failure".into(),
            });
        }
        Ok(())
    }
    fn synchronize_generated_columns(
        &mut self,
        _: &mut [Column],
        _: &[Option<Column>],
    ) -> baldes::Result<()> {
        if self.failure == Failure::Synchronization {
            self.mapping = vec![Some(999)];
            return Err(baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::Runtime,
                message: "injected pool synchronization failure".into(),
            });
        }
        Ok(())
    }
}

fn run<M: LpMaster>(new: impl Fn() -> baldes::Result<M>) {
    for failure in [
        Failure::Preparation,
        Failure::Regeneration,
        Failure::BackendInsertion,
        Failure::ColumnCap,
        Failure::Remapping,
        Failure::Synchronization,
    ] {
        let mut master = new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(10., &[(0, 1.)]).unwrap();
        let mut controller = Transition {
            failure,
            done: false,
            routes: vec![0],
            mapping: vec![Some(0)],
            saved: None,
            rollbacks: 0,
            commits: 0,
        };
        let options = || Options {
            stabilize: false,
            known: vec![column(0)],
            enumerated_columns_added: usize::MAX,
            ..Default::default()
        };
        let price = |_, _: &[f64]| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        };
        let result = cg::optimize(&mut master, &[], price, &mut controller, options()).unwrap();
        assert_eq!(result.status, Status::Optimal, "{failure:?}");
        assert!(result.enumeration_failure.is_some(), "{failure:?}");
        assert_eq!(result.lp, 10.);
        assert_eq!(result.column_tags, [0]);
        assert_eq!(master.costs(), [10.]);
        assert_eq!(master.column_entries(0).unwrap(), [(0, 1.)]);
        assert_eq!(master.values(), [1.]);
        assert_eq!(result.warm_start.columns.len(), 1);
        assert_eq!(result.warm_start.generated_pool.len(), 1);
        assert!(!controller.done);
        assert_eq!(controller.routes, [0]);
        assert_eq!(controller.mapping, [Some(0)]);
        assert_eq!(controller.rollbacks, 1);
        assert_eq!(controller.commits, 0);
        assert!(controller.saved.is_none());

        // The same restored controller and LP can retry successfully.
        controller.failure = Failure::None;
        let retried = cg::optimize(&mut master, &[], price, &mut controller, options()).unwrap();
        assert_eq!(retried.status, Status::Optimal);
        assert!(retried.enumeration_failure.is_none());
        assert_eq!(retried.lp, 10.);
        assert_eq!(master.costs(), [10., 20., 30.]);
        assert_eq!(retried.column_tags, [0, 1, 2]);
        assert_eq!(controller.routes, [0, 1, 2]);
        assert!(controller.done);
        assert_eq!(controller.rollbacks, 1);
        assert_eq!(controller.commits, 1);
    }
}

#[test]
fn clp_failed_transfer_restores_lp_controller_pool_and_supports_retry() {
    run(baldes::master::clp::ClpMaster::new);
}

#[cfg(feature = "highs")]
#[test]
fn highs_failed_transfer_restores_lp_controller_pool_and_supports_retry() {
    run(baldes::master::highs::HighsMaster::new);
}

#[test]
fn successful_handoff_repeats_only_after_formulation_changes() {
    use baldes::master::clp::ClpMaster;
    struct Repeated {
        calls: usize,
        change_row: bool,
        row_added: bool,
        saved: usize,
    }
    impl Controller<ClpMaster> for Repeated {
        fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
            self.saved = self.calls;
            Ok(true)
        }
        fn finish_enumeration_transaction(&mut self, commit: bool) {
            if !commit {
                self.calls = self.saved;
            }
        }
        fn pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> baldes::Result<Option<f64>> {
            // A deliberately loose valid bound keeps the converged fast path
            // from hiding duplicate handoffs behind its own stopping gate.
            Ok(Some(-100.))
        }
        fn enumerated_universe(
            &mut self,
            _: &[f64],
            _: f64,
        ) -> baldes::Result<Option<Vec<Column>>> {
            self.calls += 1;
            let mut current = column(0);
            if self.row_added {
                current.entries.push((1, 1.));
            }
            Ok(Some(vec![current]))
        }
        fn separates(&self) -> bool {
            self.change_row
        }
        fn separate(&mut self, master: &mut ClpMaster, _: &[i32]) -> baldes::Result<usize> {
            if self.row_added {
                return Ok(0);
            }
            master.add_row(RowSense::LessEqual, 2., &[(0, 1.)])?;
            self.row_added = true;
            Ok(1)
        }
    }
    for change_row in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(10., &[(0, 1.)]).unwrap();
        let mut controller = Repeated {
            calls: 0,
            change_row,
            row_added: false,
            saved: 0,
        };
        let output = cg::optimize(
            &mut master,
            &[],
            |_, _| {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut controller,
            Options {
                known: vec![column(0)],
                stabilize: false,
                max_iterations: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.status, Status::Optimal);
        assert_eq!(controller.calls, if change_row { 2 } else { 1 });
        assert_eq!(master.costs(), [10.]);
        assert_eq!(master.rhs().len(), if change_row { 2 } else { 1 });
        assert_eq!(output.lp, 10.);
        assert_eq!(output.column_tags, [0]);
        assert_eq!(output.warm_start.generated_pool.len(), 1);
        assert_eq!(
            output.warm_start.generated_pool[0].entries,
            master.column_entries(0).unwrap()
        );
    }
}

#[test]
fn handoff_forgets_removed_columns_so_they_can_be_loaded_again() {
    use baldes::master::clp::ClpMaster;
    struct Replace {
        done: bool,
        include_original: bool,
    }
    impl Controller<ClpMaster> for Replace {
        fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
            Ok(!self.done)
        }
        fn enumerated_universe(
            &mut self,
            _: &[f64],
            _: f64,
        ) -> baldes::Result<Option<Vec<Column>>> {
            self.done = true;
            let mut universe = vec![column(1)];
            if self.include_original {
                universe.push(column(0));
            }
            Ok(Some(universe))
        }
        fn regenerate_enumerated_columns(
            &mut self,
            _: &[Column],
            _: &[f64],
            _: f64,
            _: &[Column],
        ) -> baldes::Result<Vec<Column>> {
            Ok(vec![])
        }
    }
    // The old column can return either in the enumerated universe or in a
    // later pricing pass after the restricted universe has been abandoned.
    for include_original in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(10., &[(0, 1.)]).unwrap();
        let output = cg::optimize(
            &mut master,
            &[],
            |_, duals| {
                let mut original = column(0);
                original.reduced_cost = original.cost - duals[0];
                Ok(Answer {
                    minimum: original.reduced_cost,
                    columns: vec![original],
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut Replace {
                done: false,
                include_original,
            },
            Options {
                known: vec![column(0)],
                enumerated_columns_added: usize::MAX,
                stabilize: false,
                max_iterations: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.status, Status::Optimal);
        assert_eq!(output.lp, 10., "include_original={include_original}");
        assert!(output.column_tags.contains(&0));
        assert_eq!(output.bound, 10.);
    }
}
