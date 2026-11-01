use super::*;

#[test]
fn equal_automatic_radii_keep_two_independent_activity_bands() {
    let state = CurvatureState {
        outer_radius: 120.,
        inner_radius: Some(120.),
        curvature: 120.,
    };
    let parameters = state.bands().unwrap().unwrap();
    assert_eq!(parameters.bands.len(), 2);
    assert!(parameters
        .bands
        .iter()
        .all(|band| band.half_width == 60. && band.penalty == 1.));
}

#[test]
fn configured_starting_curvature_uses_native_stored_float_precision() {
    let parameters = AutomaticCurvatureParameters {
        kappa: 0.,
        curvature: 0.3,
        ..Default::default()
    };
    assert!(parameters.validate().is_ok());
    assert_eq!(parameters.initial_state().curvature, 0.3_f64 as f32 as f64);
    assert!(AutomaticCurvatureParameters {
        curvature: 1e100,
        ..parameters
    }
    .validate()
    .is_err());
}
#[test]
fn coupled_radius_cases_and_float_rate_match_the_recovered_arithmetic() {
    let rate = 0.2_f64 as f32 as f64;
    for (mean, inner, outer) in [
        (2., 20. - rate * 10., 10. - rate * 8.),
        (30., 20. + rate * 10., 10. + rate * 10.),
        (15., 20., 10. + rate * 5.),
    ] {
        let mut state = CurvatureState {
            inner_radius: Some(20.),
            outer_radius: 10.,
            curvature: 2.,
        };
        assert!(state
            .update(CurvatureFeedback {
                mean_dual_distance: Some(mean),
                advance_rate: 0.2
            })
            .unwrap());
        assert_eq!(state.inner_radius, Some(inner));
        assert_eq!(state.outer_radius, outer);
        let bands = state.bands().unwrap().unwrap();
        assert_eq!(bands.bands[0].half_width, outer * 0.5);
        assert_eq!(bands.bands[0].penalty, outer / 2.);
    }
}
#[test]
fn unavailable_feedback_does_not_mutate_state_and_invalid_feedback_fails() {
    let mut state = CurvatureState {
        inner_radius: None,
        outer_radius: 10.,
        curvature: 2.,
    };
    let before = state;
    assert!(!state
        .update(CurvatureFeedback {
            mean_dual_distance: None,
            advance_rate: 0.2
        })
        .unwrap());
    assert_eq!(state, before);
    assert!(state
        .update(CurvatureFeedback {
            mean_dual_distance: Some(f64::NAN),
            advance_rate: 0.2
        })
        .is_err());
    assert_eq!(state, before);
    assert_eq!(mean_dual_distance(&[1., 4.], &[3., 8.], &[0, 1]), Some(3.));
    assert_eq!(mean_dual_distance(&[1.], &[3.], &[1]), None);
}

#[test]
fn automatic_initialization_waits_for_mean_and_widens_stored_kappa() {
    for five_piece in [false, true] {
        let parameters = AutomaticCurvatureParameters {
            initial: None,
            five_piece,
            kappa: 0.7,
            advance_rate: 0.2,
            ..Default::default()
        };
        let mut state = parameters.initial_state();
        let original = state;
        assert!(!parameters.update(&mut state, None).unwrap());
        assert_eq!(state, original);
        assert!(parameters.update(&mut state, Some(4.)).unwrap());
        assert_eq!(state.curvature, 4. / (0.7_f64 as f32 as f64));
        assert_eq!(state.outer_radius, 4.);
        assert_eq!(state.inner_radius, five_piece.then_some(4.));
    }
}

include!("native_capture.rs");

#[test]
fn curvature_updates_and_bands_match_captured_native_kernel() {
    for (index, &(five_piece, bits)) in CURVATURE_CAPTURE.iter().enumerate() {
        let values = bits.map(f64::from_bits);
        let mut state = CurvatureState {
            outer_radius: values[0],
            inner_radius: five_piece.then_some(values[1]),
            curvature: values[3],
        };
        state
            .update(CurvatureFeedback {
                mean_dual_distance: Some(values[2]),
                advance_rate: values[4],
            })
            .unwrap();
        assert_eq!(state.outer_radius.to_bits(), bits[5], "outer case {index}");
        if five_piece {
            assert_eq!(
                state.inner_radius.unwrap().to_bits(),
                bits[6],
                "inner case {index}"
            );
        }
        let mut expected = vec![];
        for (width, cap) in [(values[7], values[8]), (values[9], values[10])] {
            if cap > 0. {
                expected.push((width, cap));
            }
        }
        expected.sort_by(|a, b| a.0.total_cmp(&b.0));
        let parameters = state.bands().unwrap();
        let actual = parameters.as_ref().map_or(&[][..], |p| p.bands.as_slice());
        assert_eq!(actual.len(), expected.len(), "band count case {index}");
        for (band, (width, cap)) in actual.iter().zip(expected) {
            assert_eq!(
                band.half_width.to_bits(),
                width.to_bits(),
                "width case {index}"
            );
            assert_eq!(band.penalty.to_bits(), cap.to_bits(), "cap case {index}");
        }
    }
}
