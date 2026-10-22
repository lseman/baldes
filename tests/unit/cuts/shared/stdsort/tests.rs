#[test]
fn stable_sorts() {
    let mut v: Vec<(i32, usize)> = (0..200).map(|i| ((i * 7919) % 13, i as usize)).collect();
    super::stable_sort(&mut v, |a, b| a.0 < b.0);
    assert!(v
        .windows(2)
        .all(|w| w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1)));
}
#[test]
fn sorts() {
    let mut v: Vec<i32> = (0..200).map(|i| (i * 7919) % 101).collect();
    super::sort(&mut v, |a, b| a < b);
    assert!(v.windows(2).all(|w| w[0] <= w[1]));
}
