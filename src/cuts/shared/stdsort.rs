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
fn adjust_heap<T: Clone, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    mut hole: usize,
    len: usize,
    value: T,
    less: &F,
) {
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
fn move_median_to_first<T, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    a: usize,
    b: usize,
    c: usize,
    less: &F,
) {
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

/// libstdc++ `std::stable_sort` with the temporary buffer it normally gets
/// ((len + 1) / 2 elements): chunked insertion sort, then buffered merges.
/// Equivalent elements under a tolerant comparator end where this algorithm
/// leaves them.
pub fn stable_sort<T: Clone, F: Fn(&T, &T) -> bool>(v: &mut [T], less: F) {
    let len = v.len();
    if len < 2 {
        return;
    }
    let half = (len + 1) / 2;
    let mut buffer: Vec<T> = v[..half].to_vec();
    let (left, right) = v.split_at_mut(half);
    merge_sort_with_buffer(left, &mut buffer, &less);
    merge_sort_with_buffer(right, &mut buffer, &less);
    merge_adaptive(v, half, &mut buffer, &less);
}
fn merge_sort_with_buffer<T: Clone, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    buffer: &mut Vec<T>,
    less: &F,
) {
    let len = v.len();
    const CHUNK: usize = 7;
    let mut start = 0;
    while len - start >= CHUNK {
        insertion_sort(&mut v[start..start + CHUNK], less);
        start += CHUNK;
    }
    insertion_sort(&mut v[start..], less);
    let mut step = CHUNK;
    buffer.clear();
    buffer.extend_from_slice(v);
    while step < len {
        merge_sort_loop(v, buffer, step, less);
        step *= 2;
        let snapshot = buffer.clone();
        merge_sort_loop(&snapshot, v, step, less);
        step *= 2;
    }
}
/// __merge_sort_loop: merges runs of `step` from `from` into `to`.
fn merge_sort_loop<T: Clone, F: Fn(&T, &T) -> bool>(
    from: &[T],
    to: &mut [T],
    step: usize,
    less: &F,
) {
    let two = 2 * step;
    let len = from.len();
    let mut first = 0;
    let mut out = 0;
    while len - first >= two {
        out = move_merge(
            &from[first..first + step],
            &from[first + step..first + two],
            to,
            out,
            less,
        );
        first += two;
    }
    let step = step.min(len - first);
    move_merge(
        &from[first..first + step],
        &from[first + step..len],
        to,
        out,
        less,
    );
}
fn move_merge<T: Clone, F: Fn(&T, &T) -> bool>(
    a: &[T],
    b: &[T],
    to: &mut [T],
    mut out: usize,
    less: &F,
) -> usize {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if less(&b[j], &a[i]) {
            to[out] = b[j].clone();
            j += 1;
        } else {
            to[out] = a[i].clone();
            i += 1;
        }
        out += 1;
    }
    for x in &a[i..] {
        to[out] = x.clone();
        out += 1;
    }
    for x in &b[j..] {
        to[out] = x.clone();
        out += 1;
    }
    out
}
/// __merge_adaptive with a sufficient buffer.
fn merge_adaptive<T: Clone, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    middle: usize,
    _buffer: &mut Vec<T>,
    less: &F,
) {
    let len1 = middle;
    let len2 = v.len() - middle;
    if len1 <= len2 {
        let left: Vec<T> = v[..middle].to_vec();
        // __move_merge_adaptive(buffer, buffer_end, middle, last, first)
        let (mut i, mut j, mut out) = (0, middle, 0);
        while i < left.len() && j < v.len() {
            if less(&v[j], &left[i]) {
                v[out] = v[j].clone();
                j += 1;
            } else {
                v[out] = left[i].clone();
                i += 1;
            }
            out += 1;
        }
        for x in &left[i..] {
            v[out] = x.clone();
            out += 1;
        }
    } else {
        let right: Vec<T> = v[middle..].to_vec();
        // __move_merge_adaptive_backward(first, middle, buffer, buffer_end, last)
        let mut result = v.len();
        let mut l1 = middle; // one past the last of the first range
        let mut l2 = right.len();
        if l2 == 0 {
            return;
        }
        l1 -= 1;
        l2 -= 1;
        loop {
            if less(&right[l2], &v[l1]) {
                result -= 1;
                v[result] = v[l1].clone();
                if l1 == 0 {
                    for k in (0..=l2).rev() {
                        result -= 1;
                        v[result] = right[k].clone();
                    }
                    return;
                }
                l1 -= 1;
            } else {
                result -= 1;
                v[result] = right[l2].clone();
                if l2 == 0 {
                    return;
                }
                l2 -= 1;
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/shared/stdsort/tests.rs"]
mod tests;
