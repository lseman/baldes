use super::format_g;
#[test]
// This decimal input is a formatting fixture, rather than a geometric constant.
#[allow(clippy::approx_constant)]
fn printf_g() {
    assert_eq!(format_g(0.5, 6), "0.5");
    assert_eq!(format_g(100000.0, 6), "100000");
    assert_eq!(format_g(1000000.0, 6), "1e+06");
    assert_eq!(format_g(1e-5, 6), "1e-05");
    assert_eq!(format_g(3.14159265, 6), "3.14159");
    assert_eq!(format_g(-2.0, 6), "-2");
}
