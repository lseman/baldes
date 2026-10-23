mod initialization;
mod persistent;
pub use persistent::{Solver, SolverStatistics};

use super::model::StandardForm;
use super::{dot, finite, norm, Error, ModelData, Result, SolverType, SparseSolver};

#[derive(Debug, Clone)]
pub struct Options {
    /// Cooperative wall-time budget, checked between Newton iterations.
    pub time_limit: Option<std::time::Duration>,
    pub tolerance: f64,
    /// Optional optimality tolerance, independent of primal/dual feasibility.
    /// Uses |primal - dual| / (1 + |primal|), as in primal-dual CG.
    pub gap_tolerance: Option<f64>,
    pub max_iterations: usize,
    pub solver: SolverType,
    /// Ruiz equilibration passes on the standard-form matrix (zero disables).
    pub scaling_iterations: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            time_limit: None,
            tolerance: 1e-8,
            gap_tolerance: None,
            max_iterations: 500,
            solver: SolverType::HiPO,
            scaling_iterations: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Optimal,
    IterationLimit,
    TimeLimit,
    Infeasible,
    Unbounded,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub status: Status,
    pub primal: Vec<f64>,
    /// Multipliers in original row senses, using c - Aᵀy for reduced costs.
    pub dual: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub relative_gap: f64,
    pub solver: SolverType,
    /// Standard-form certificate, when detected: primal ray for Unbounded;
    /// stacked (row multipliers, upper-bound multipliers) for Infeasible.
    /// Coordinates refer to the equilibrated standard form, before recovery.
    pub certificate: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct Point {
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    v: Vec<f64>,
    w: Vec<f64>,
    tau: f64,
    kappa: f64,
}
impl Point {
    fn alpha(&self, d: &Self) -> f64 {
        let mut alpha = 1.0_f64;
        for (v, dv) in [
            (&self.x, &d.x),
            (&self.s, &d.s),
            (&self.v, &d.v),
            (&self.w, &d.w),
        ] {
            for (&v, &dv) in v.iter().zip(dv) {
                if dv < 0. {
                    alpha = alpha.min(-v / dv);
                }
            }
        }
        if d.tau < 0. {
            alpha = alpha.min(-self.tau / d.tau);
        }
        if d.kappa < 0. {
            alpha = alpha.min(-self.kappa / d.kappa);
        }
        alpha
    }
    fn update(&mut self, d: &Self, alpha: f64) -> Result<()> {
        for (v, dv) in [
            (&mut self.x, &d.x),
            (&mut self.y, &d.y),
            (&mut self.s, &d.s),
            (&mut self.v, &d.v),
            (&mut self.w, &d.w),
        ] {
            for (v, &dv) in v.iter_mut().zip(dv) {
                *v += alpha * dv;
            }
        }
        self.tau += alpha * d.tau;
        self.kappa += alpha * d.kappa;
        if [&self.x, &self.s, &self.v, &self.w]
            .iter()
            .any(|v| v.iter().any(|&x| !x.is_finite() || x <= 0.))
            || !finite(&self.y)
            || !self.tau.is_finite()
            || !self.kappa.is_finite()
            || self.tau <= 0.
            || self.kappa <= 0.
        {
            return Err(Error::NumericalFailure(
                "iterate left the strict interior".into(),
            ));
        }
        Ok(())
    }

    /// Evaluate a corrected direction without copying all five vectors.
    /// Addition order matches materializing `direction + correction` first.
    fn alpha_with_correction(&self, direction: &Self, correction: &Self) -> f64 {
        let mut alpha = 1.0_f64;
        for (values, direction, correction) in [
            (&self.x, &direction.x, &correction.x),
            (&self.s, &direction.s, &correction.s),
            (&self.v, &direction.v, &correction.v),
            (&self.w, &direction.w, &correction.w),
        ] {
            for ((&value, &delta), &change) in values.iter().zip(direction).zip(correction) {
                let delta = delta + change;
                if delta < 0. {
                    alpha = alpha.min(-value / delta);
                }
            }
        }
        for (value, delta) in [
            (self.tau, direction.tau + correction.tau),
            (self.kappa, direction.kappa + correction.kappa),
        ] {
            if delta < 0. {
                alpha = alpha.min(-value / delta);
            }
        }
        alpha
    }

    fn add_direction(&mut self, correction: &Self) {
        for (values, correction) in [
            (&mut self.x, &correction.x),
            (&mut self.y, &correction.y),
            (&mut self.s, &correction.s),
            (&mut self.v, &correction.v),
            (&mut self.w, &correction.w),
        ] {
            for (value, &change) in values.iter_mut().zip(correction) {
                *value += change;
            }
        }
        self.tau += correction.tau;
        self.kappa += correction.kappa;
    }
}

#[derive(Default)]
struct Residual {
    p: Vec<f64>,
    u: Vec<f64>,
    d: Vec<f64>,
    g: f64,
}

fn residual_into(f: &StandardForm, q: &Point, out: &mut Residual) {
    f.a.mul_into(&q.x, &mut out.p);
    for (p, b) in out.p.iter_mut().zip(&f.b) {
        *p = q.tau * b - *p;
    }
    out.u.clear();
    out.u.extend(
        f.upper
            .iter()
            .enumerate()
            .map(|(i, &(j, b))| q.tau * b - q.x[j] - q.v[i]),
    );
    f.a.tmul_into(&q.y, &mut out.d);
    for ((d, c), s) in out.d.iter_mut().zip(&f.c).zip(&q.s) {
        *d = q.tau * c - *d - s;
    }
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        out.d[j] += q.w[i];
    }
    out.g = q.kappa + dot(&f.c, &q.x) - dot(&f.b, &q.y) + bound_dot(f, &q.w);
}
fn bound_dot(f: &StandardForm, w: &[f64]) -> f64 {
    f.upper.iter().zip(w).map(|((_, u), w)| u * w).sum()
}
/// Solve an LP using the homogeneous self-dual predictor-corrector method.
/// `Optimal` requires primal feasibility, dual feasibility and a relative gap
/// below tolerance. Numerical failures are returned as errors.
pub fn solve(model: &ModelData, options: &Options) -> Result<Solution> {
    Solver::default().solve(model, options)
}

fn solve_prepared(
    model: &ModelData,
    options: &Options,
    prepared: &mut persistent::Prepared,
    initial: Option<Point>,
) -> Result<(Solution, Point)> {
    let started = std::time::Instant::now();
    let persistent::Prepared {
        f,
        row_scale,
        col_scale,
        cost_scale,
        unscaled_cost_norm,
        ls,
        solve_workspace,
        newton_scratch,
        iterations: work_iterations,
        ..
    } = prepared;
    let cost_scale = *cost_scale;
    let unscaled_cost_norm = *unscaled_cost_norm;
    let n = f.c.len();
    let m = f.b.len();
    let nu = f.upper.len();
    let mut q = initial
        .or_else(|| initialization::least_squares(f, ls, solve_workspace))
        .unwrap_or_else(|| initialization::unit(f));
    let mut rp = vec![1.0_f64; n];
    let mut rd = vec![1.0_f64; m];
    let mut rg = 1.0_f64;
    let floor = f64::EPSILON.sqrt();
    let tol = options.tolerance;
    let mut status = Status::IterationLimit;
    let mut iterations = 0;
    let mut cert = None;
    let mut metrics = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let ubnorm = f.upper.iter().fold(0.0_f64, |v, (_, u)| v.max(u.abs()));
    let rhsnorm = norm(&f.b);
    let upper: Vec<_> = f.upper.iter().map(|&(_, u)| u).collect();
    let mut original = Vec::with_capacity(f.recovery.len());
    let mut tvw = Vec::with_capacity(nu);
    let mut theta = Vec::with_capacity(n);
    let mut r = Residual::default();
    let mut original_activity = Vec::with_capacity(model.a.rows());
    let mut xs = Vec::with_capacity(n);
    let mut vw = Vec::with_capacity(nu);
    let mut cx = Vec::with_capacity(n);
    let mut cv = Vec::with_capacity(nu);
    let mut tx = Vec::with_capacity(n);
    let mut tv = Vec::with_capacity(nu);
    let mut damped = Residual::default();
    let mut zero: Option<Residual> = None;
    for k in 0..=options.max_iterations {
        iterations = k;
        *work_iterations = k;
        if options
            .time_limit
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            status = Status::TimeLimit;
            break;
        }
        residual_into(f, &q, &mut r);
        let dual_obj = dot(&f.b, &q.y) - bound_dot(f, &q.w);
        metrics = (
            (norm(&r.p) / (q.tau * (1. + rhsnorm))).max(norm(&r.u) / (q.tau * (1. + ubnorm))),
            r.d.iter()
                .zip(col_scale.iter())
                .fold(0.0_f64, |v, (d, s)| v.max((d / s).abs()))
                / (q.tau * cost_scale * (1. + unscaled_cost_norm)),
            (dot(&f.c, &q.x) - dual_obj).abs() / (cost_scale * q.tau + dual_obj.abs()),
        );
        // Verify original coordinates too, since equilibration can conceal
        // large original-unit row errors.
        original.clear();
        original.extend(f.recovery.iter().map(|(offset, terms)| {
            offset
                + terms
                    .iter()
                    .map(|&(j, s)| s * col_scale[j] * q.x[j] / q.tau)
                    .sum::<f64>()
        }));
        let original_p = primal_residual(model, &original, &mut original_activity);
        if options.gap_tolerance.is_some() {
            // Recovery includes objective contributions from shifted/fixed variables.
            metrics.2 = (dot(&f.c, &q.x) - dual_obj).abs()
                / (cost_scale * q.tau * (1. + dot(&model.c, &original).abs()));
        }
        if metrics.0 <= tol
            && original_p <= tol
            && metrics.1 <= tol
            && metrics.2 <= options.gap_tolerance.unwrap_or(tol)
        {
            status = Status::Optimal;
            break;
        }
        if q.tau < 1e-7 * q.kappa {
            if let Some((s, c)) = certificate(f, &q, tol.min(1e-8)) {
                status = s;
                cert = Some(c);
                break;
            }
        }
        if k == options.max_iterations {
            break;
        }
        tvw.clear();
        tvw.extend(q.w.iter().zip(&q.v).map(|(w, v)| w / v));
        theta.clear();
        theta.extend(q.s.iter().zip(&q.x).map(|(s, x)| s / x));
        for (i, &(j, _)) in f.upper.iter().enumerate() {
            theta[j] += tvw[i];
        }
        for x in rp.iter_mut().chain(&mut rd) {
            *x = (*x / 10.).max(floor);
        }
        rg = (rg / 10.).max(floor);
        let mut step = None;
        let mut last_error = Error::NotFactorized;
        // A factor can be finite yet unusable. Retry the entire Newton step
        // with stronger regularization when a solve or refinement fails.
        for _ in 0..8 {
            let result = (|| -> Result<(Point, f64)> {
                ls.factorize(&theta, &rp, &rd)?;
                let delta = augsys(ls, f, &tvw, &f.b, &f.c, &upper, solve_workspace)?;
                // Equivalent positive quadratic form avoids cancellation between the
                // three dot products in the original homogeneous Schur complement.
                let delta0 = rg
                    + q.kappa / q.tau
                    + delta
                        .0
                        .iter()
                        .enumerate()
                        .map(|(j, dx)| (q.s[j] / q.x[j] + rp[j]) * dx * dx)
                        .sum::<f64>()
                    + delta
                        .1
                        .iter()
                        .zip(&rd)
                        .map(|(dy, r)| r * dy * dy)
                        .sum::<f64>()
                    + f.upper
                        .iter()
                        .enumerate()
                        .map(|(i, &(j, u))| tvw[i] * (delta.0[j] - u).powi(2))
                        .sum::<f64>();
                if !delta0.is_finite() || delta0 <= 0. {
                    return Err(Error::NumericalFailure(
                        "invalid homogeneous Schur pivot".into(),
                    ));
                }
                let newton = Newton {
                    ls,
                    f,
                    q: &q,
                    tvw: &tvw,
                    delta,
                    delta0,
                };
                xs.clear();
                xs.extend(q.x.iter().zip(&q.s).map(|(x, s)| -x * s));
                vw.clear();
                vw.extend(q.v.iter().zip(&q.w).map(|(v, w)| -v * w));
                let mu = (q.tau * q.kappa - xs.iter().sum::<f64>() - vw.iter().sum::<f64>())
                    / (n + nu + 1) as f64;
                let affine = newton.solve(
                    &r,
                    &xs,
                    &vw,
                    -q.tau * q.kappa,
                    solve_workspace,
                    newton_scratch,
                )?;
                let one_minus = 1. - q.alpha(&affine);
                // alpha is in [0, 1], so the expression before clipping is at most
                // 0.1. A lower bound of 0.1 would make centering constant and damp
                // even near-full affine steps. Retain strong centering for short
                // steps, but permit faster feasibility progress near a full step.
                let gamma = (one_minus * one_minus * one_minus.min(0.1)).clamp(1e-4, 0.1);
                let damping = 1. - gamma;
                damped.p.clear();
                damped.p.extend(r.p.iter().map(|r| damping * r));
                damped.u.clear();
                damped.u.extend(r.u.iter().map(|r| damping * r));
                damped.d.clear();
                damped.d.extend(r.d.iter().map(|r| damping * r));
                damped.g = damping * r.g;
                cx.clear();
                cx.extend(
                    xs.iter()
                        .zip(&affine.x)
                        .zip(&affine.s)
                        .map(|((xs, x), s)| xs + gamma * mu - x * s),
                );
                cv.clear();
                cv.extend(
                    vw.iter()
                        .zip(&affine.v)
                        .zip(&affine.w)
                        .map(|((vw, v), w)| vw + gamma * mu - v * w),
                );
                let mut direction = newton.solve(
                    &damped,
                    &cx,
                    &cv,
                    -q.tau * q.kappa + gamma * mu - affine.tau * affine.kappa,
                    solve_workspace,
                    newton_scratch,
                )?;
                let mut alpha = q.alpha(&direction);
                // Gondzio centrality corrections are added to the existing direction.
                // The C++ path overwrites it, which can discard the feasibility step.
                for _ in 0..3 {
                    if alpha >= 0.9995 {
                        break;
                    }
                    let trial = (2. * alpha).min(1.);
                    let low = 0.1 * mu * gamma;
                    let high = mu * gamma / 0.1;
                    tx.clear();
                    tx.extend(
                        q.x.iter()
                            .zip(&direction.x)
                            .zip(&q.s)
                            .zip(&direction.s)
                            .map(|(((x, dx), s), ds)| {
                                let v = (x + trial * dx) * (s + trial * ds);
                                v.clamp(low, high) - v
                            }),
                    );
                    tv.clear();
                    tv.extend(
                        q.v.iter()
                            .zip(&direction.v)
                            .zip(&q.w)
                            .zip(&direction.w)
                            .map(|(((v, dv), w), dw)| {
                                let z = (v + trial * dv) * (w + trial * dw);
                                z.clamp(low, high) - z
                            }),
                    );
                    let z = (q.tau + trial * direction.tau) * (q.kappa + trial * direction.kappa);
                    let mut t0 = z.clamp(low, high) - z;
                    let mean = (tx.iter().sum::<f64>() + tv.iter().sum::<f64>() + t0)
                        / (n + nu + 1) as f64;
                    for t in tx.iter_mut().chain(&mut tv) {
                        *t -= mean;
                    }
                    t0 -= mean;
                    let zero = zero.get_or_insert_with(|| Residual {
                        p: vec![0.; m],
                        u: vec![0.; nu],
                        d: vec![0.; n],
                        g: 0.,
                    });
                    let correction =
                        newton.solve(zero, &tx, &tv, t0, solve_workspace, newton_scratch)?;
                    let improved = q.alpha_with_correction(&direction, &correction);
                    if improved > alpha {
                        direction.add_direction(&correction);
                        alpha = improved;
                    } else {
                        break;
                    }
                }
                Ok((direction, alpha))
            })();
            match result {
                Ok(value) => {
                    step = Some(value);
                    break;
                }
                Err(e @ Error::NumericalFailure(_)) => last_error = e,
                Err(e) => return Err(e),
            }
            for x in rp.iter_mut().chain(&mut rd) {
                *x *= 10.;
            }
            rg *= 10.;
        }
        let (direction, alpha) = step.ok_or(last_error)?;
        q.update(&direction, 0.9995 * alpha)?;
    }
    let primal: Vec<_> = f
        .recovery
        .iter()
        .map(|(offset, terms)| {
            offset
                + terms
                    .iter()
                    .map(|&(j, s)| s * col_scale[j] * q.x[j] / q.tau)
                    .sum::<f64>()
        })
        .collect();
    let dual: Vec<_> =
        q.y.iter()
            .zip(row_scale.iter())
            .zip(&f.row_sign)
            .map(|((y, r), s)| y * r * s / q.tau / cost_scale)
            .collect();
    let objective = dot(&model.c, &primal);
    if !finite(&primal) || !finite(&dual) || !objective.is_finite() {
        return Err(Error::NumericalFailure("solution recovery overflow".into()));
    }
    Ok((
        Solution {
            status,
            objective,
            primal_residual: primal_residual(model, &primal, &mut original_activity),
            dual_residual: metrics.1,
            relative_gap: metrics.2,
            primal,
            dual,
            iterations,
            solver: ls.active_solver_type(),
            certificate: cert,
        },
        q,
    ))
}

fn primal_residual(model: &ModelData, x: &[f64], activity: &mut Vec<f64>) -> f64 {
    let mut residual = 0.0_f64;
    model.a.mul_into(x, activity);
    for ((ax, &b), sense) in activity.iter().zip(&model.b).zip(&model.sense) {
        let v = match sense {
            super::Sense::Equal => (ax - b).abs(),
            super::Sense::LessEqual => (ax - b).max(0.),
            super::Sense::GreaterEqual => (b - ax).max(0.),
        };
        residual = residual.max(v / (1. + b.abs()));
    }
    for ((&x, &l), &u) in x.iter().zip(&model.lower).zip(&model.upper) {
        if l.is_finite() {
            residual = residual.max((l - x).max(0.) / (1. + l.abs()));
        }
        if u.is_finite() {
            residual = residual.max((x - u).max(0.) / (1. + u.abs()));
        }
    }
    residual
}

mod newton;
use newton::{augsys, Newton};

mod scaling;
use scaling::{certificate, scale};

#[cfg(test)]
mod direction_tests {
    use super::*;

    #[test]
    fn correction_step_bound_matches_materialized_directions() {
        let q = Point {
            x: vec![0.5, 2., 3.],
            y: vec![0., -1.],
            s: vec![2., 1., 5.],
            v: vec![1., 2.],
            w: vec![5., 1.],
            tau: 1.,
            kappa: 1.5,
        };
        let mut rng = crate::rng::Xoroshiro128Plus::from_seed(0x01f0_cafe);
        for _ in 0..1000 {
            let mut values = |n| {
                (0..n)
                    .map(|_| (rng.uniform_below(257) as f64 - 128.) / 16.)
                    .collect()
            };
            let direction = Point {
                x: values(3),
                y: values(2),
                s: values(3),
                v: values(2),
                w: values(2),
                tau: -0.25,
                kappa: 0.5,
            };
            let correction = Point {
                x: values(3),
                y: values(2),
                s: values(3),
                v: values(2),
                w: values(2),
                tau: 0.125,
                kappa: -2.,
            };
            let mut combined = direction.clone();
            combined.add_direction(&correction);
            assert_eq!(
                q.alpha_with_correction(&direction, &correction).to_bits(),
                q.alpha(&combined).to_bits()
            );
        }
    }
}
