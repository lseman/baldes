use super::*;
#[test]
fn interrupted_numerical_proofs_publish_no_partial_bound() {
    let columns = [Column {
        cost_terms: &[3.],
        entries: &[(0, 1.)],
    }];
    let model = FiniteMaster {
        rhs: &[1.],
        senses: &[RowSense::LessEqual],
        columns: &columns,
        offset_terms: &[],
        multiplicity_row: 0,
    };
    let mut checkpoints = 0usize;
    assert_eq!(
        model.lower_bound_checked(&[0.], || {
            checkpoints += 1;
            Some(())
        }),
        Some(0.)
    );
    for allowance in 0usize..checkpoints {
        let mut remaining = allowance;
        assert_eq!(
            model.lower_bound_checked(&[0.], || {
                remaining = remaining.checked_sub(1usize)?;
                Some(())
            }),
            None
        );
    }
    assert_eq!(model.lower_bound(&[0.]), Some(0.));
}
#[test]
fn fractional_primal_and_projected_duals_have_a_valid_bound() {
    let columns = [
        Column {
            cost_terms: &[3.],
            entries: &[(0, 1.), (1, 2.)],
        },
        Column {
            cost_terms: &[2.],
            entries: &[(0, 1.), (1, 1.)],
        },
    ];
    let model = FiniteMaster {
        rhs: &[2., 1.],
        senses: &[RowSense::LessEqual, RowSense::Equal],
        columns: &columns,
        offset_terms: &[],
        multiplicity_row: 0,
    };
    // x=(1/2,0) and these duals both attain 3/2.
    assert_eq!(model.lower_bound(&[0., 1.5]), Some(1.5));
    // An invalid positive <= dual must not manufacture a stronger bound.
    assert_eq!(model.lower_bound(&[1e6, 1.5]), Some(1.5));
}
#[test]
fn original_cost_terms_do_not_lose_negative_cancellation() {
    let columns = [Column {
        cost_terms: &[-1., 1e16, -1e16],
        entries: &[(0, 1.), (1, 1.)],
    }];
    let model = FiniteMaster {
        rhs: &[1., 1.],
        senses: &[RowSense::LessEqual, RowSense::Equal],
        columns: &columns,
        offset_terms: &[],
        multiplicity_row: 0,
    };
    // Exactly x=1, objective -1. Ordinary accumulation gives the false 0.
    assert!(model.lower_bound(&[0., 0.]).unwrap() <= -1.);
}
#[test]
fn fractional_multiplicity_and_fixed_objective_terms_are_supported() {
    let columns = [Column {
        cost_terms: &[-4.],
        entries: &[(0, 1.)],
    }];
    let model = FiniteMaster {
        rhs: &[0.25],
        senses: &[RowSense::LessEqual],
        columns: &columns,
        offset_terms: &[(3., 2.)],
        multiplicity_row: 0,
    };
    assert_eq!(model.lower_bound(&[0.]), Some(5.));
}
#[test]
fn an_unproved_multiplicity_row_cannot_certify_the_matrix() {
    for entries in [&[(0, 0.5)][..], &[(0, 1e16), (0, 1.), (0, -1e16)][..]] {
        let columns = [Column {
            cost_terms: &[-1.],
            entries,
        }];
        let model = FiniteMaster {
            rhs: &[1.],
            senses: &[RowSense::LessEqual],
            columns: &columns,
            offset_terms: &[],
            multiplicity_row: 0,
        };
        assert!(model.lower_bound(&[0.]).is_none());
    }
}
#[test]
fn invalid_and_overflowing_data_leave_the_bound_unknown() {
    for terms in [&[f64::NAN][..], &[f64::MAX, f64::MAX][..]] {
        let columns = [Column {
            cost_terms: terms,
            entries: &[(0, 1.)],
        }];
        let model = FiniteMaster {
            rhs: &[1.],
            senses: &[RowSense::LessEqual],
            columns: &columns,
            offset_terms: &[],
            multiplicity_row: 0,
        };
        assert!(model.lower_bound(&[0.]).is_none());
        assert!(model.lower_bound(&[]).is_none());
    }
}
