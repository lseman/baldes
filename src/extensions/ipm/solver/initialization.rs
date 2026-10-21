//! Optional regularized least-squares starting point for the HSD iteration.
use super::super::sparse_solver::SolveWorkspace;
use super::{finite, Point, SparseSolver, StandardForm};

pub(super) fn unit(f: &StandardForm) -> Point {
    let n = f.c.len();
    let m = f.b.len();
    let nu = f.upper.len();
    let mut q = Point {
        x: vec![1.; n],
        y: vec![0.; m],
        s: vec![1.; n],
        v: vec![1.; nu],
        w: vec![1.; nu],
        tau: 1.,
        kappa: 1.,
    };
    for (i, &(_, upper)) in f.upper.iter().enumerate() {
        if upper > 2. {
            // x = 1 and tau = 1: satisfy wide upper bounds initially,
            // with v*w = 1 like the other complementarity products.
            // Unit slacks would introduce residuals of order upper and
            // collapse the homogeneous scale on routing masters (u=1e12).
            q.v[i] = upper - 1.;
            q.w[i] = 1. / q.v[i];
        }
    }
    q
}

pub(super) fn least_squares(
    f: &StandardForm,
    solver: &mut SparseSolver,
    workspace: &mut SolveWorkspace,
) -> Option<Point> {
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
    solver.solve_into(&rhs, workspace).ok()?;
    let mut x = workspace.solution[..n].to_vec();
    rhs.clear();
    rhs.extend_from_slice(&f.c);
    rhs.resize(n + m, 0.);
    solver.solve_into(&rhs, workspace).ok()?;
    let y = workspace.solution[n..].to_vec();
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
