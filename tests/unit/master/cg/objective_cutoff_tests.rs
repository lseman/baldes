use super::*;
use crate::master::clp::ClpMaster;

struct ExternalCutoff {
    certificate: Option<f64>,
    complete: bool,
}
impl Controller<ClpMaster> for ExternalCutoff {
    fn objective_cutoff(&self) -> f64 {
        5.
    }
    fn pricing_complete(&self) -> bool {
        self.complete
    }
    fn pricing_bound(&self, master: &ClpMaster, _: &[f64]) -> Result<Option<f64>> {
        Ok(Some(master.objective()))
    }
    fn certified_pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> Result<Option<f64>> {
        Ok(self.certificate)
    }
}

fn artificial_only(
    mode: BoundMode,
    certificate: Option<f64>,
    complete: bool,
) -> (ClpMaster, Output) {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    let phases = [
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: 10,
        },
        Phase {
            check_ng: false,
            jump_arcs: false,
            labels_per_bucket: 1,
        },
    ];
    let mut controller = ExternalCutoff {
        certificate,
        complete,
    };
    let out = optimize(
        &mut master,
        &phases,
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: false,
                bound_minimum: None,
            })
        },
        &mut controller,
        Options {
            bound_mode: mode,
            stabilize: false,
            artificial_cost: 10.,
            max_iterations: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(controller.incumbent(), f64::INFINITY);
    assert!(!controller.original_witness_available());
    (master, out)
}

#[test]
fn external_cutoff_closes_exact_penalized_bound_without_local_primal() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (master, out) = artificial_only(mode, Some(10.), true);
        assert_eq!(out.status, Status::Optimal);
        assert!(out.pruned_by_bound);
        assert_eq!(out.bound, 5.);
        assert_eq!(out.penalty_updates, 0);
        assert_eq!(
            out.history
                .iter()
                .map(|round| round.phase)
                .collect::<Vec<_>>(),
            [2, 1, 0]
        );
        assert!(out
            .artificials
            .iter()
            .any(|&column| master.values()[column] > 0.));
        assert!(out.feasibility.is_empty());
        assert_eq!(out.lp, 10.);
    }
}

#[test]
fn positive_artificial_cutoff_requires_selected_proof_and_complete_exact_domain() {
    let (_, unknown) = artificial_only(BoundMode::Certified, None, true);
    assert!(!unknown.pruned_by_bound);
    assert_eq!(unknown.certified_bound, f64::NEG_INFINITY);
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (_, partial) = artificial_only(mode, Some(10.), false);
        assert!(!partial.pruned_by_bound);
        assert_eq!(partial.numerical_bound, f64::NEG_INFINITY);
        assert_eq!(partial.certified_bound, f64::NEG_INFINITY);
    }
}
