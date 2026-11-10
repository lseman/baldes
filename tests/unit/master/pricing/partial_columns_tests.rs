use super::*;

#[derive(Default)]
struct PartialNetwork {
    accept: bool,
    minimum: Option<f64>,
    empty: bool,
    count: usize,
    requested_exact: Option<bool>,
}
impl Networks<2> for PartialNetwork {
    fn accepts_partial_columns(&self, _: usize) -> bool {
        self.accept
    }
    fn partial_minimum(&self, _: usize) -> Option<f64> {
        self.minimum
    }
    fn active(&self) -> Vec<usize> {
        (0..self.count.max(1)).collect()
    }
    fn prepare(&mut self, _: usize, _: usize, _: &[f64], _: bool) -> Result<Request> {
        Ok(Request {
            phase: 0,
            exact: self.requested_exact.unwrap_or(true),
            offset: 0.,
            lower: 0.,
            upper: 1.,
            envelopes: Default::default(),
        })
    }
    fn enter(&mut self, _: usize) -> &mut Solver<2> {
        panic!("mock overrides price")
    }
    fn price(
        &mut self,
        _: usize,
        _: usize,
        _: std::time::Duration,
    ) -> Result<crate::PricingResult> {
        let mut graph = crate::Graph::<2>::new(2, 0)?;
        graph.add_arc(0, 1, 1.)?;
        let mut route = graph.elementary_route(&[0])?.unwrap();
        route.cost = -1.;
        Ok(crate::PricingResult {
            columns: if self.empty { vec![] } else { vec![route] },
            minimum_reduced_cost: -1.,
            exact: false,
            statistics: Default::default(),
        })
    }
    fn collect(&mut self, _: usize, routes: Vec<Route>, _: f64, _: &[f64]) -> Result<Collected> {
        Ok(Collected {
            columns: routes
                .into_iter()
                .map(|route| cg::Column {
                    cost: route.original_cost,
                    reduced_cost: route.cost,
                    entries: vec![(0, 1.)],
                    acceptable: true,
                    tag: 0,
                    identity: "partial".into(),
                })
                .collect(),
            fresh: 1,
            best_entries: Some(vec![(0, 1.)]),
            best_route: None,
        })
    }
}
#[test]
fn unfinished_exact_column_batches_require_opt_in_and_never_publish_a_minimum() {
    for accept in [false, true] {
        let mut scheduler = Scheduler::new(NetworkPricingSchedule::FullSweep, 1, 1e-6);
        let result = price(
            &mut PartialNetwork {
                accept,
                ..Default::default()
            },
            &mut scheduler,
            Context {
                phase: 0,
                duals: &[2.],
                rhs: &[1.],
                deadline: None,
                admission: false,
                master: None,
                gap: (0., 0.),
            },
        );
        if accept {
            let batch = result.unwrap();
            assert_eq!(batch.answer.columns.len(), 1);
            assert!(!batch.answer.exact);
            assert!(!scheduler.complete);
            assert!(batch.correction.is_none());
            assert!(batch.gradient.is_none());
            assert!(batch.answer.bound_minimum.is_none());
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn completed_stage_minimum_survives_an_empty_unfinished_sweep_without_certifying_it() {
    for (empty, requested_exact) in [(false, true), (true, true), (false, false), (true, false)] {
        let mut scheduler = Scheduler::new(NetworkPricingSchedule::FullSweep, 1, 1e-6);
        let batch = price(
            &mut PartialNetwork {
                accept: true,
                minimum: Some(-7.),
                empty,
                requested_exact: Some(requested_exact),
                ..Default::default()
            },
            &mut scheduler,
            Context {
                phase: 0,
                duals: &[2.],
                rhs: &[1.],
                deadline: None,
                admission: false,
                master: None,
                gap: (0., 0.),
            },
        )
        .unwrap();
        assert_eq!(batch.correction, Some(-7.));
        assert!(!scheduler.complete);
        assert!(!batch.answer.exact);
        assert!(batch.answer.bound_minimum.is_none());
        assert!(batch.gradient.is_none());
    }
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::FullSweep, 2, 1e-6);
    let batch = price(
        &mut PartialNetwork {
            accept: true,
            minimum: Some(-7.),
            count: 2,
            ..Default::default()
        },
        &mut scheduler,
        Context {
            phase: 0,
            duals: &[2.],
            rhs: &[1.],
            deadline: None,
            admission: false,
            master: None,
            gap: (0., 0.),
        },
    )
    .unwrap();
    assert!(
        batch.correction.is_none(),
        "unpriced networks prevent a global correction"
    );
}
