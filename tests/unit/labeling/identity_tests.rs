use super::*;
#[test]
fn reserved_identity_preserves_arc_and_schedule_encoding() {
    use std::fmt::Write;
    for arcs in [vec![], vec![0], vec![i32::MIN, i32::MAX, -1, 10]] {
        for resource_profile in [
            vec![],
            vec![vec![]],
            vec![vec![
                -0.,
                0.,
                1.,
                -1.,
                f64::from_bits(1),
                f64::INFINITY,
                f64::from_bits(0x7ff8000000000012),
            ]],
        ] {
            let route = Route {
                nodes: vec![],
                arcs: arcs.clone(),
                cost: 0.,
                original_cost: 0.,
                resources: vec![],
                resource_profile,
            };
            let mut expected = arcs
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            for state in &route.resource_profile {
                expected.push('|');
                for &value in state {
                    let bits = if value == 0. { 0 } else { value.to_bits() };
                    write!(&mut expected, "{bits:016x},").unwrap();
                }
            }
            assert_eq!(route.identity(), expected);
        }
    }
}
