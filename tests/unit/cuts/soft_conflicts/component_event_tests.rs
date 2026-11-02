use super::*;

// The original event-scan coefficient is the equivalence oracle. In particular,
// a coordinate with a zero emission is present, unlike an omitted coordinate.
fn reference(class: &Class, state: &PresenceState) -> Result<usize> {
    if state.values.len() != class.coordinates {
        return Err(invalid("component state domain"));
    }
    let present = |event: usize| state.words[event / 64] & (1 << (event % 64)) != 0;
    if !present(class.emissions.len()) {
        return Ok(0);
    }
    Ok(usize::from(class.sequence.satisfied_by(|coordinate| {
        class
            .emissions
            .iter()
            .enumerate()
            .any(|(event, &(c, _))| c == coordinate && present(event))
            .then_some(state.values[coordinate] as f64)
    })))
}

fn bounds() -> ComponentSequence {
    ComponentSequence {
        bounds: vec![
            ComponentBound {
                coordinate: 0,
                threshold: -1.,
                sense: ComponentSense::GreaterEqual,
            },
            ComponentBound {
                coordinate: 2,
                threshold: 0.,
                sense: ComponentSense::LessEqual,
            },
            ComponentBound {
                coordinate: 2,
                threshold: 0.,
                sense: ComponentSense::GreaterEqual,
            },
            ComponentBound {
                coordinate: 3,
                threshold: -10.,
                sense: ComponentSense::LessEqual,
            },
        ],
    }
}

#[test]
fn grouped_and_interleaved_component_events_match_every_small_presence_pattern() {
    for emissions in [
        vec![(0, 0), (0, 1), (1, -1), (2, 1), (2, -1), (2, 0)],
        vec![(0, 0), (2, 1), (1, -1), (0, 1), (2, -1), (2, 0)],
    ] {
        let row = PresenceRow::component_class(4, bounds(), emissions).unwrap();
        let class = row.component.as_ref().unwrap();
        for pattern in 0..(1_u64 << row.events) {
            let mut state = row.empty_state();
            state.words[0] = pattern;
            for values in [[0, 0, 0, 0], [-2, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0]] {
                state.values.copy_from_slice(&values);
                assert_eq!(
                    class.coefficient(&state).unwrap(),
                    reference(class, &state).unwrap()
                );
            }
        }
        let cloned = row.clone();
        if let Some(cache) = &class.coordinate_events {
            assert!(std::sync::Arc::ptr_eq(
                cache,
                cloned
                    .component
                    .as_ref()
                    .unwrap()
                    .coordinate_events
                    .as_ref()
                    .unwrap()
            ));
        } else {
            assert!(class.emissions.windows(2).any(|w| w[0].0 > w[1].0));
        }
    }
}

#[test]
fn component_ranges_mask_terminal_and_cross_word_boundaries_exactly() {
    let emissions: Vec<_> = (0..193)
        .map(|e| {
            (
                if e < 32 {
                    0
                } else if e < 160 {
                    1
                } else if e < 192 {
                    2
                } else {
                    3
                },
                if e % 2 == 0 { 1 } else { -1 },
            )
        })
        .collect();
    let sequence = ComponentSequence {
        bounds: (0..4)
            .map(|coordinate| ComponentBound {
                coordinate,
                threshold: 0.,
                sense: ComponentSense::LessEqual,
            })
            .collect(),
    };
    let row = PresenceRow::component_class(4, sequence, emissions).unwrap();
    let class = row.component.as_ref().unwrap();
    assert!(class.coordinate_events.is_some());
    let mut rng = 1202026_u64;
    for _ in 0..4096 {
        let mut state = row.empty_state();
        for word in &mut state.words {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *word = rng;
        }
        assert_eq!(
            class.coefficient(&state).unwrap(),
            reference(class, &state).unwrap()
        );
        for coordinate in 0..4 {
            assert_eq!(
                class.coordinate_present(&state, coordinate),
                class
                    .emissions
                    .iter()
                    .enumerate()
                    .any(|(e, &(c, _))| c == coordinate
                        && state.words[e / 64] & (1_u64 << (e % 64)) != 0)
            );
        }
    }
    let mut terminal = row.empty_state();
    class.advance(&mut terminal, 193).unwrap();
    assert!(!class.coordinate_present(&terminal, 3));
    // A missing <= coordinate accepts independently of its negative threshold.
    assert_eq!(class.coefficient(&terminal).unwrap(), 1);
    let mut wrong = row.empty_state();
    wrong.values.pop();
    assert_eq!(
        class.coefficient(&wrong).unwrap_err().message,
        reference(class, &wrong).unwrap_err().message
    );
}

#[test]
fn component_event_index_preserves_repetition_cancellation_join_and_overflow() {
    let row = PresenceRow::component_class(
        1,
        ComponentSequence {
            bounds: vec![ComponentBound {
                coordinate: 0,
                threshold: 0.,
                sense: ComponentSense::GreaterEqual,
            }],
        },
        vec![(0, 0), (0, 2), (0, -2), (0, 1_i64 << 53)],
    )
    .unwrap();
    let class = row.component.as_ref().unwrap();
    let mut state = row.empty_state();
    for event in [4, 0, 1, 2, 1, 2, 0] {
        class.advance(&mut state, event).unwrap();
        assert_eq!(
            class.coefficient(&state).unwrap(),
            reference(class, &state).unwrap()
        );
    }
    let joined = class.join(&state, &state).unwrap();
    assert_eq!(
        class.coefficient(&joined).unwrap(),
        reference(class, &joined).unwrap()
    );
    class.advance(&mut state, 3).unwrap();
    assert!(class.advance(&mut state, 3).is_err());
}
