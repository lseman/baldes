use super::*;
#[test]
fn hash_matches_libstdcxx() {
    let mut b = Bits1024::default();
    b.set(3);
    assert_eq!(hash(&b), 0x630d4f168d99e17b);
}
