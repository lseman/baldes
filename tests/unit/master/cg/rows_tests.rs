use super::*;
use crate::master::clp::ClpMaster;

#[test]
fn artificial_signs_and_ids_match_initial_cut_and_branch_installation() {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::LessEqual, -2., &[]).unwrap();
    master.add_row(RowSense::GreaterEqual, 3., &[]).unwrap();
    master.add_row(RowSense::Equal, -4., &[]).unwrap();
    let installed = append_artificials(&mut master, 0..3, 10.).unwrap();
    assert_eq!(installed, vec![(0, 0), (1, 1), (2, 2), (2, 3)]);
    assert!(master.solve());
    assert_eq!(master.objective(), 90.);
    assert_eq!(master.values(), &[2., 3., 0., 4.]);
    let count = master.costs.len();
    assert!(append_artificials(&mut master, 0..3, 0.)
        .unwrap()
        .is_empty());
    assert!(append_artificials(&mut master, 3..4, 10.).is_err());
    assert_eq!(master.costs.len(), count);
}
