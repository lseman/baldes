//! `std::unordered_set<std::bitset<1024>>` as libstdc++ builds it: the
//! `_Hash_bytes` hash, prime bucket counts with maximum load factor 1, and
//! insertion at the front of a bucket (or of the whole list for an empty
//! bucket). Separators iterate these sets, so iteration order is part of
//! their behaviour.
use super::bits::Bits1024;

/// std::__detail::__prime_list (first entries).
const PRIMES: [u64; 160] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 103, 109, 113,
    127, 137, 139, 149, 157, 167, 179, 193, 199, 211, 227, 241, 257, 277, 293, 313, 337, 359, 383, 409, 439, 467,
    503, 541, 577, 619, 661, 709, 761, 823, 887, 953, 1031, 1109, 1193, 1289, 1381, 1493, 1613, 1741, 1879, 2029,
    2179, 2357, 2549, 2753, 2971, 3209, 3469, 3739, 4027, 4349, 4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123,
    8783, 9497, 10273, 11113, 12011, 12983, 14033, 15173, 16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411,
    30727, 33223, 35933, 38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307, 72817, 78779, 85229, 92203,
    99733, 107897, 116731, 126271, 136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897, 256279, 277261,
    299951, 324503, 351061, 379787, 410857, 444487, 480881, 520241, 562841, 608903, 658753, 712697, 771049, 834181,
    902483, 976369, 1056323, 1142821, 1236397, 1337629, 1447153, 1565659, 1693859, 1832561, 1982627, 2144977,
    2320627,
];
const FAST_BUCKETS: [u64; 14] = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13];

/// _Hash_bytes over the 128 bytes of the bitset, seed 0xc70f6907.
pub fn hash(bits: &Bits1024) -> u64 {
    const MUL: u64 = (0xc6a4a793u64 << 32) + 0x5bd1e995;
    let shift_mix = |v: u64| v ^ (v >> 47);
    let mut h = 0xc70f6907u64 ^ 128u64.wrapping_mul(MUL);
    for &word in &bits.0 {
        let data = shift_mix(word.wrapping_mul(MUL)).wrapping_mul(MUL);
        h ^= data;
        h = h.wrapping_mul(MUL);
    }
    h = shift_mix(h).wrapping_mul(MUL);
    shift_mix(h)
}

const BEFORE_BEGIN: usize = usize::MAX;
const NONE: usize = usize::MAX - 1;

#[derive(Clone, Debug)]
struct Node {
    value: Bits1024,
    code: u64,
    next: usize,
}
#[derive(Clone, Debug)]
pub struct UnorderedSet {
    nodes: Vec<Node>,
    /// Per bucket, the node before its first node (BEFORE_BEGIN or a node).
    buckets: Vec<usize>,
    first: usize,
    next_resize: u64,
}
impl Default for UnorderedSet {
    fn default() -> Self {
        Self::new()
    }
}
impl UnorderedSet {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            buckets: vec![NONE],
            first: NONE,
            next_resize: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
    /// clear(): elements go, the bucket array and resize threshold stay.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.buckets.iter_mut().for_each(|b| *b = NONE);
        self.first = NONE;
    }
    fn next_of(&self, at: usize) -> usize {
        if at == BEFORE_BEGIN {
            self.first
        } else {
            self.nodes[at].next
        }
    }
    fn set_next(&mut self, at: usize, next: usize) {
        if at == BEFORE_BEGIN {
            self.first = next;
        } else {
            self.nodes[at].next = next;
        }
    }
    pub fn contains(&self, value: &Bits1024) -> bool {
        let code = hash(value);
        let b = (code % self.buckets.len() as u64) as usize;
        let before = self.buckets[b];
        if before == NONE {
            return false;
        }
        let mut p = self.next_of(before);
        while p != NONE {
            if self.nodes[p].code % self.buckets.len() as u64 != b as u64 {
                return false;
            }
            if self.nodes[p].value == *value {
                return true;
            }
            p = self.nodes[p].next;
        }
        false
    }
    /// _Prime_rehash_policy::_M_next_bkt.
    fn next_bucket_count(&mut self, n: u64) -> u64 {
        if (n as usize) < FAST_BUCKETS.len() {
            if n == 0 {
                return 1;
            }
            let b = FAST_BUCKETS[n as usize];
            self.next_resize = b;
            return b;
        }
        let candidates = &PRIMES[6..PRIMES.len() - 1];
        let at = candidates.partition_point(|&p| p < n);
        let p = candidates.get(at).copied().unwrap_or(*PRIMES.last().unwrap());
        self.next_resize = p;
        p
    }
    /// _M_need_rehash(bucket count, element count, 1).
    fn need_rehash(&mut self) -> Option<u64> {
        let (n_bkt, n_elt) = (self.buckets.len() as u64, self.nodes.len() as u64);
        if n_elt + 1 > self.next_resize {
            let min_bkts = (n_elt + 1).max(if self.next_resize != 0 { 0 } else { 11 }) as f64;
            if min_bkts >= n_bkt as f64 {
                let target = (min_bkts.floor() as u64 + 1).max(n_bkt * 2);
                return Some(self.next_bucket_count(target));
            }
            self.next_resize = n_bkt;
        }
        None
    }
    fn rehash(&mut self, n: usize) {
        let mut buckets = vec![NONE; n];
        let mut p = self.first;
        self.first = NONE;
        let mut bbegin_bkt = 0usize;
        while p != NONE {
            let next = self.nodes[p].next;
            let b = (self.nodes[p].code % n as u64) as usize;
            if buckets[b] == NONE {
                self.nodes[p].next = self.first;
                self.first = p;
                buckets[b] = BEFORE_BEGIN;
                if self.nodes[p].next != NONE {
                    buckets[bbegin_bkt] = p;
                }
                bbegin_bkt = b;
            } else {
                let before = buckets[b];
                let after = self.next_of(before);
                self.nodes[p].next = after;
                self.set_next(before, p);
            }
            p = next;
        }
        self.buckets = buckets;
    }
    /// Inserts when absent; returns whether it was inserted.
    pub fn insert(&mut self, value: Bits1024) -> bool {
        if self.contains(&value) {
            return false;
        }
        let code = hash(&value);
        if let Some(n) = self.need_rehash() {
            self.rehash(n as usize);
        }
        let b = (code % self.buckets.len() as u64) as usize;
        let node = self.nodes.len();
        self.nodes.push(Node {
            value,
            code,
            next: NONE,
        });
        let before = self.buckets[b];
        if before != NONE {
            let after = self.next_of(before);
            self.nodes[node].next = after;
            self.set_next(before, node);
        } else {
            self.nodes[node].next = self.first;
            self.first = node;
            if self.nodes[node].next != NONE {
                let other = self.nodes[self.nodes[node].next].code % self.buckets.len() as u64;
                self.buckets[other as usize] = node;
            }
            self.buckets[b] = BEFORE_BEGIN;
        }
        true
    }
    /// Values in iteration order.
    pub fn iter(&self) -> Vec<Bits1024> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut p = self.first;
        while p != NONE {
            out.push(self.nodes[p].value);
            p = self.nodes[p].next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_matches_libstdcxx() {
        let mut b = Bits1024::default();
        b.set(3);
        assert_eq!(hash(&b), 0x630d4f168d99e17b);
    }
}
