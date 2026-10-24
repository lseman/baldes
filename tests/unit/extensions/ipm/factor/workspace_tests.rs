use super::*;

#[test]
fn consumed_schur_storage_is_reused_with_a_bounded_cache() {
    let mut pool = SchurPool::default();
    let entries = SCHUR_CACHE_BYTES / std::mem::size_of::<f64>() / 2;
    let first = vec![3.; entries];
    let second = vec![4.; entries];
    let pointers = [first.as_ptr(), second.as_ptr()];
    pool.put(first);
    pool.put(second);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    pool.put(vec![5.; entries]);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    let reused = pool.take(entries / 2);
    assert!(reused.is_empty());
    assert!(pointers.contains(&reused.as_ptr()));
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES / 2);
    pool.put(reused);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    // Empty roots and requests larger than the cache need no cached buffer.
    assert_eq!(pool.take(0).capacity(), 0);
    assert_eq!(pool.take(2 * entries + 1).capacity(), 0);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
}
