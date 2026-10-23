use super::*;
use crate::master::preprocessing::Bounds;
#[test]
fn open_import_certificates_do_not_use_known_column_projected_caps() {
    let mut options = Options::default();
    options.preprocessing.enabled = false;
    let result = imported::solve(
        Problem::<2> {
            rows: vec![],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: -1.,
            bounds: Bounds {
                lower: 0.,
                upper: f64::INFINITY,
            },
            integer: false,
            entries: vec![],
        }],
        vec![imported::Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: Bounds {
                lower: 0.,
                upper: 1.,
            },
            columns: vec![imported::Column {
                id: 0,
                cost: 2.,
                bounds: Bounds {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                integer: false,
                entries: vec![(0, 1.)],
            }],
        }],
        options,
    )
    .unwrap();
    let node = &result.output.snapshot;
    assert!(node.state.master_variables[0].bounds.upper.is_infinite());
    let duals = vec![0.; node.state.rhs.len()];
    // x<=1 follows only from currently known lambda vectors. An unseen
    // vector x=100, cost=0 could yield objective -100 at the same total cap.
    assert_eq!(certificate(&node.state, &node.master, &duals), Some(-1.));
    assert_eq!(
        extension_certificate(&node.state, &node.master, &duals),
        None
    );
}
