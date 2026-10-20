//! Optional regularized least-squares starting point for the HSD iteration.
use super::{finite, Point, SparseSolver, StandardForm};

pub(super) fn least_squares(f: &StandardForm, solver: &mut SparseSolver) -> Option<Point> {
    let n = f.c.len();
    let m = f.b.len();
    if n == 0 || m == 0 || f.upper.iter().any(|&(_, upper)| upper < 1e-6) {
        return None;
    }
    // Solve regularized AAᵀ systems through the existing augmented/normal
    // backend. Failure keeps the unit starting point used by the caller.
    solver
        .factorize(&vec![1.; n], &vec![1e-8; n], &vec![1e-8; m])
        .ok()?;
    let mut rhs = vec![0.; n];
    rhs.extend_from_slice(&f.b);
    let primal = solver.solve(&rhs).ok()?;
    let mut rhs = f.c.clone();
    rhs.resize(n + m, 0.);
    let dual = solver.solve(&rhs).ok()?;
    let mut x = primal[..n].to_vec();
    let y = dual[n..].to_vec();
    // HSD requires positive primal variables, unlike HiPO's separate bound
    // slacks. Shift the estimate and keep each finite bound strictly interior.
    let shift = (-1.5 * x.iter().copied().fold(0., f64::min)).max(0.) + 1e-2;
    for value in &mut x {
        *value += shift;
    }
    let mut v = Vec::with_capacity(f.upper.len());
    let mut w = Vec::with_capacity(f.upper.len());
    for &(j, upper) in &f.upper {
        x[j] = x[j].min(upper * 0.5);
        v.push(upper - x[j]);
        // Preserve unit complementarity even for routing upper bounds of 1e12.
        w.push(1. / (upper - x[j]));
    }
    let mut s: Vec<_> = f.c.iter().zip(f.a.tmul(&y)).map(|(c, a)| c - a).collect();
    let shift = (-1.5 * s.iter().copied().fold(0., f64::min)).max(0.) + 1e-2;
    for value in &mut s {
        *value += shift;
    }
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        s[j] += w[i];
    }
    if !finite(&y)
        || [&x, &s, &v, &w]
            .iter()
            .any(|values| values.iter().any(|v| !v.is_finite() || *v <= 0.))
    {
        return None;
    }
    let mut theta: Vec<_> = s.iter().zip(&x).map(|(s, x)| s / x).collect();
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        theta[j] += w[i] / v[i];
    }
    if !finite(&theta) {
        return None;
    }
    Some(Point {
        x,
        y,
        s,
        v,
        w,
        tau: 1.,
        kappa: 1.,
    })
}
