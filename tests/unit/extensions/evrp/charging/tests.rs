use super::*;
use crate::graph::coupled::{CoupledResource, PluginResource};
use crate::Graph;

fn fixture(close: f64, max_work: usize) -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (u, v) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(u, v, -1.).unwrap();
    }
    let curve = PiecewiseLinearFunction::new(vec![[0., 0.], [2., 4.], [4., 5.]]).unwrap();
    let arcs = (0..3)
        .map(|i| ChargingResourceArc {
            travel_time: 1.,
            energy: [3., 2., 1.][i],
            window: Window::new(
                if i == 1 { 6. } else { 0. },
                if i == 2 { close } else { 20. },
            )
            .unwrap(),
            service: 0.,
            charging: (i == 0).then(|| curve.clone()),
        })
        .collect();
    graph
        .set_coupled_resources(&[
            CoupledResource::Plugin(PluginResource::new(ChargingResource {
                battery: 4.,
                initial_energy: 4.,
                initial_time: 0.,
                arcs,
                max_work,
            })),
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![3, 2, 3],
                vertices: vec![],
                relaxed: false,
            },
            CoupledResource::SimultaneousPickupDelivery {
                capacity: 2.,
                deliveries: vec![0., 2., 0.],
                pickups: vec![0.; 3],
                tolerance: 0.,
            },
        ])
        .unwrap();
    graph
}

#[test]
fn every_native_split_matches_independent_charging_controls() {
    for close in [6., 7., 8., 10.] {
        // Independent quarter-energy controls. At least two units must
        // be charged; the nonconcave profile costs 2+(e-2)/2 above e=2.
        let feasible = (0..=12).any(|i| {
            let recharge = f64::from(i) / 4.;
            let target = 1. + recharge;
            let cumulative = if target <= 2. {
                2. * target
            } else {
                4. + (target - 2.) / 2.
            };
            recharge >= 2. && (1. + cumulative - 2. + 1.).max(6.) + 1. <= close
        });
        let graph = fixture(close, 10000);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.replay(&[0, 1, 2]).unwrap().is_some(), feasible);
        for split in 0..=3 {
            let mut f = kernel.root(true);
            for arc in 0..split {
                let Some((next, _)) = kernel.extend(&f, arc, true).unwrap() else {
                    break;
                };
                f = next;
            }
            // For infeasible complete prefixes, test the forward replay
            // separately above; an absent label has no split join.
            if kernel
                .replay_state(&(0..split).collect::<Vec<_>>())
                .unwrap()
                .is_none()
            {
                continue;
            }
            let mut b = kernel.root(false);
            for arc in (split..3).rev() {
                b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
            }
            assert_eq!(
                kernel.join(&f, &b).unwrap().is_some(),
                feasible,
                "close={close}, split={split}"
            );
        }
    }
}

#[test]
fn native_charging_front_exhaustion_is_an_error() {
    let graph = fixture(10., 0);
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel
        .replay(&[0, 1, 2])
        .unwrap_err()
        .message
        .contains("work limit"));
    let b = kernel
        .extend(&kernel.root(false), 2, false)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel.extend(&b, 1, false).unwrap().unwrap().0;
    let b = kernel.extend(&b, 0, false).unwrap().unwrap().0;
    assert!(kernel.join(&kernel.root(true), &b).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn phase_one_keeps_charging_feasibility_and_declines_certification() {
    for close in [6., 7.] {
        let graph = fixture(close, 10000);
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        kernel.suppress_costs();
        assert_eq!(kernel.replay(&[0, 1, 2]).unwrap().is_some(), close >= 7.);
        assert!(kernel.certify(&[0, 1, 2]).is_none());
    }
}
