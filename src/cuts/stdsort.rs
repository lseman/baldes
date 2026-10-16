//! libstdc++ `std::sort` (introsort, median-of-three, final insertion sort).
//! Unstable: equivalent elements end in the order this exact algorithm
//! leaves them, which separators expose through their candidate order.

fn insertion_sort<T, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    if v.is_empty() {
        return;
    }
    for i in 1..v.len() {
        if less(&v[i], &v[0]) {
            v[..=i].rotate_right(1);
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}
fn unguarded_linear_insert<T, F: Fn(&T, &T) -> bool>(v: &mut [T], mut last: usize, less: &F) {
    // Unguarded in libstdc++: a sentinel stops the walk before index 0.
    while last > 0 && less(&v[last], &v[last - 1]) {
        v.swap(last, last - 1);
        last -= 1;
    }
}
fn adjust_heap<T: Clone, F: Fn(&T, &T) -> bool>(v: &mut [T], mut hole: usize, len: usize, value: T, less: &F) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second].clone();
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1].clone();
        hole = second - 1;
    }
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[parent], &value) {
        v[hole] = v[parent].clone();
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[hole] = value;
}
fn heap_sort<T: Clone, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    let len = v.len();
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let value = v[parent].clone();
            adjust_heap(v, parent, len, value, less);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    let mut last = len;
    while last > 1 {
        last -= 1;
        let value = v[last].clone();
        v[last] = v[0].clone();
        adjust_heap(&mut v[..last], 0, last, value, less);
    }
}
fn move_median_to_first<T, F: Fn(&T, &T) -> bool>(v: &mut [T], a: usize, b: usize, c: usize, less: &F) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(0, b);
        } else if less(&v[a], &v[c]) {
            v.swap(0, c);
        } else {
            v.swap(0, a);
        }
    } else if less(&v[a], &v[c]) {
        v.swap(0, a);
    } else if less(&v[b], &v[c]) {
        v.swap(0, c);
    } else {
        v.swap(0, b);
    }
}
fn unguarded_partition<T, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) -> usize {
    // Pivot at 0, range [1, len).
    let (mut first, mut last) = (1usize, v.len());
    loop {
        while less(&v[first], &v[0]) {
            first += 1;
        }
        last -= 1;
        while less(&v[0], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}
fn introsort_loop<T: Clone, F: Fn(&T, &T) -> bool>(v: &mut [T], mut depth: usize, less: &F) {
    let mut len = v.len();
    while len > 16 {
        if depth == 0 {
            heap_sort(&mut v[..len], less);
            return;
        }
        depth -= 1;
        let mid = len / 2;
        move_median_to_first(&mut v[..len], 1, mid, len - 1, less);
        let cut = unguarded_partition(&mut v[..len], less);
        introsort_loop(&mut v[cut..len], depth, less);
        len = cut;
    }
}
pub fn sort<T: Clone, F: Fn(&T, &T) -> bool>(v: &mut [T], less: F) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let lg = (usize::BITS - 1 - n.leading_zeros()) as usize;
    introsort_loop(v, lg * 2, &less);
    if n > 16 {
        insertion_sort(&mut v[..16], &less);
        for i in 16..n {
            unguarded_linear_insert(v, i, &less);
        }
    } else {
        insertion_sort(v, &less);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sorts() {
        let mut v: Vec<i32> = (0..200).map(|i| (i * 7919) % 101).collect();
        super::sort(&mut v, |a, b| a < b);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}
