//! A fixed 1024-bit set (`std::bitset<1024>`): out-of-range reads are false
//! and out-of-range writes are ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bits1024(pub [u64; 16]);
impl Bits1024 {
    pub fn test(&self, i: usize) -> bool {
        i < 1024 && self.0[i / 64] >> (i % 64) & 1 != 0
    }
    pub fn set(&mut self, i: usize) {
        if i < 1024 {
            self.0[i / 64] |= 1 << (i % 64);
        }
    }
    pub fn reset(&mut self, i: usize) {
        if i < 1024 {
            self.0[i / 64] &= !(1 << (i % 64));
        }
    }
    pub fn flip(&mut self, i: usize) {
        if i < 1024 {
            self.0[i / 64] ^= 1 << (i % 64);
        }
    }
    pub fn flip_all(&mut self) {
        for w in &mut self.0 {
            *w = !*w;
        }
    }
    pub fn clear(&mut self) {
        self.0 = [0; 16];
    }
}
