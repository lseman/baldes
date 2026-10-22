use super::*;

#[test]
fn create_and_access() {
    let mut m = Matrix::new(3);
    assert_eq!(m.get(0, 0), 0.);
    m.set(1, 2, 42);
    assert_eq!(m.get(1, 2), 42.);
    assert_eq!(m[(1, 2)], 42.);
    m[(1, 2)] = 99.;
    assert_eq!(m.get(1, 2), 99.);
}

#[test]
fn row_slice() {
    let mut m = Matrix::new(3);
    m.set(0, 0, 1);
    m.set(0, 1, 2);
    m.set(0, 2, 3);
    assert_eq!(m.row(0), &[1., 2., 3.]);
}
