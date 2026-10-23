use super::*;
use crate::graph::Window;

#[test]
fn native_preflight_matches_scalar_propagation_and_ng_rejection() {
    let mut g = Graph::<2>::new(4, 2).unwrap();
    for v in &mut g.vertices {
        v.bounds[0] = Window::new(-10., 30.).unwrap();
        v.bounds[1] = Window::new(0., 20.).unwrap();
    }
    g.initial[0] = Some(2.);
    for (tail, head, consumption) in [(0, 1, [1.5, 2.]), (1, 2, [2., 3.]), (2, 3, [0., 0.])] {
        g.add_arc(tail, head, 1.).unwrap();
        let id = g.arcs.len() - 1;
        g.arcs[id].consumption = consumption.into();
        g.arcs[id].bounds[0] = Some(Window::new(-5., 25.).unwrap());
    }
    g.vertices[1].elements = vec![0];
    g.vertices[2].elements = vec![1];
    let engine = LabelingEngine::prepare(
        &g,
        EngineControls {
            steps: [5., 5.].into(),
            use_symmetric_execution: false,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    assert!(ExtensionBatch::supported(&g));
    let mut batch = ExtensionBatch::default();
    for x in [
        -20.,
        -10.,
        -5.,
        -0.,
        0.,
        2.,
        10.,
        20.,
        25.,
        30.,
        40.,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        for y in [-1., -0., 0., 5., 20., 21., f64::MAX, f64::NEG_INFINITY] {
            for ng in [0, 1, 2, 3, u16::MAX] {
                batch.input.push(Input {
                    resources: [x, y],
                    ng,
                });
            }
        }
    }
    batch.output.resize(batch.input.len(), Output::default());
    let mut checked = 0;
    let mut jumps = 0;
    for forward in [false, true] {
        let ctx = engine.ctx();
        for bucket in ctx.grid.direction(forward).buckets.iter().flatten() {
            for arc in &bucket.arcs {
                batch.run(ctx, &g, arc, forward);
                jumps += usize::from(arc.jump.is_some());
                for (position, input) in batch.input.iter().enumerate() {
                    let mut q = input.resources;
                    if let Some(jump) = arc.jump {
                        let point = ctx.grid.direction(forward).corners[arc.from.vertex][jump];
                        for (axis, &slot) in ctx.grid.resource_slots.iter().enumerate() {
                            q[slot] = if forward {
                                q[slot].max(point[axis])
                            } else {
                                q[slot].min(point[axis])
                            };
                        }
                    }
                    let transition = if forward {
                        &ctx.ng.forward[arc.physical]
                    } else {
                        &ctx.ng.backward[arc.physical]
                    };
                    let expected = crate::labeling::ScalarResourceKernel::extend_active(
                        &g,
                        &g.arcs[arc.physical],
                        q,
                        forward,
                    )
                    .filter(|_| transition.extend(input.ng, 0).is_some());
                    let actual = batch.resources::<2>(position);
                    assert_eq!(
                        actual.is_some(),
                        expected.is_some(),
                        "direction={forward} physical={} q={:?}",
                        arc.physical,
                        input.resources
                    );
                    if let (Some(actual), Some(expected)) = (actual, expected) {
                        // Signed zeros have the same resource semantics; every
                        // other accepted value must preserve scalar arithmetic.
                        assert_eq!(actual.0, expected);
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 10_000);
    // Exercise forbidden physical arcs and disabled reduced-cost arcs as well.
    let ctx = engine.ctx();
    let arc = ctx
        .grid
        .forward
        .buckets
        .iter()
        .flatten()
        .flat_map(|b| &b.arcs)
        .next()
        .unwrap();
    let mut forbidden = vec![false; g.arcs.len()];
    forbidden[arc.physical] = true;
    batch.run(
        Context {
            forbidden_arcs: &forbidden,
            ..ctx
        },
        &g,
        arc,
        true,
    );
    assert!(batch.output.iter().all(|o| o.feasible == 0));
    let mut disabled = g.clone();
    disabled.arcs[arc.physical].reduced_cost = NO_ROUTE_COST;
    batch.run(ctx, &disabled, arc, true);
    assert!(batch.output.iter().all(|o| o.feasible == 0));
    // A synthetic jump on a valid bucket corner verifies that branch even
    // when this tiny graph's topology does not require generated jump arcs.
    let mut jumped = arc.clone();
    jumped.jump = Some(arc.from.bucket);
    batch.run(ctx, &g, &jumped, true);
    let mut without_jump = ctx;
    without_jump.phase.jump_arcs = false;
    batch.run(without_jump, &g, &jumped, true);
    let _ = jumps;
}

#[test]
fn unsupported_resource_shapes_use_scalar_path() {
    assert!(!ExtensionBatch::supported(&Graph::<1>::new(3, 1).unwrap()));
    assert!(!ExtensionBatch::supported(&Graph::<3>::new(3, 3).unwrap()));
    assert!(!ExtensionBatch::supported(&Graph::<2>::new(3, 1).unwrap()));
    let mut g = Graph::<2>::new(3, 2).unwrap();
    g.resources[1] = ResourceKind::Equality;
    assert!(!ExtensionBatch::supported(&g));
    #[cfg(feature = "multi-time-windows")]
    {
        g.resources[1] = ResourceKind::Disposable;
        g.set_time_windows(
            1,
            0,
            &[Window::new(0., 1.).unwrap(), Window::new(2., 3.).unwrap()],
        )
        .unwrap();
        assert!(!ExtensionBatch::supported(&g));
    }
}
