//! Replays insert/clear lines and prints the emulated unordered_set order.
use baldes::cuts::bits::Bits1024;
use baldes::cuts::stdhash::UnorderedSet;
use std::io::Read;
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut set = UnorderedSet::new();
    let flush = |set: &UnorderedSet| {
        println!("buckets {}", set.bucket_count());
        for b in set.iter() {
            println!("it {}", b.0[0]);
        }
    };
    let mut started = false;
    for line in text.lines() {
        if line == "trial" {
            if started {
                flush(&set);
            }
            started = true;
            println!("trial");
            set = UnorderedSet::new();
        } else if line == "clear" {
            set.clear();
            println!("clear");
        } else if let Some(v) = line.strip_prefix("ins ") {
            let mut b = Bits1024::default();
            b.0[0] = v.parse().unwrap();
            set.insert(b);
            println!("{line}");
        }
    }
    flush(&set);
}
