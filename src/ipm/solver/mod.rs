mod initialization;

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

#[derive(Clone)]
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
}

struct Residual {
    p: Vec<f64>,
    u: Vec<f64>,
    d: Vec<f64>,
    g: f64,
}

fn residual(f: &StandardForm, q: &Point) -> Residual {
    let p =
        f.a.mul(&q.x)
            .iter()
            .zip(&f.b)
            .map(|(a, b)| q.tau * b - a)
            .collect();
    let u = f
        .upper
        .iter()
        .enumerate()
        .map(|(i, &(j, b))| q.tau * b - q.x[j] - q.v[i])
        .collect();
    let mut d: Vec<_> =
        f.a.tmul(&q.y)
            .iter()
            .zip(&f.c)
            .zip(&q.s)
            .map(|((a, c), s)| q.tau * c - a - s)
            .collect();
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        d[j] += q.w[i];
    }
    Residual {
        p,
        u,
        d,
        g: q.kappa + dot(&f.c, &q.x) - dot(&f.b, &q.y) + bound_dot(f, &q.w),
    }
}
fn bound_dot(f: &StandardForm, w: &[f64]) -> f64 {
    f.upper.iter().zip(w).map(|((_, u), w)| u * w).sum()
}
/// Solve an LP using the homogeneous self-dual predictor-corrector method.
/// `Optimal` requires primal feasibility, dual feasibility and a relative gap
/// below tolerance. Numerical failures are returned as errors.
pub fn solve(model: &ModelData, options: &Options) -> Result<Solution> {
    if options
        .gap_tolerance
        .is_some_and(|gap| !gap.is_finite() || gap <= 0. || gap > 1.)
    {
        return Err(Error::InvalidInput(
            "gap tolerance must be finite and in (0, 1]".into(),
        ));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0. || options.tolerance >= 1. {
        return Err(Error::InvalidInput(
            "tolerance must be finite and between zero and one".into(),
        ));
    }
    let started = std::time::Instant::now();
    let mut f = model.standard_form()?;
    let (row_scale, col_scale) = scale(&mut f, options.scaling_iterations)?;
    let unscaled_cost_norm =
        f.c.iter()
            .zip(&col_scale)
            .fold(0.0_f64, |v, (c, s)| v.max((c / s).abs()));
    // Keep the objective comparable to the equilibrated constraint matrix.
    // Residual checks and recovered multipliers below undo this scaling.
    let cost_scale = 1. / norm(&f.c).max(1.);
    for c in &mut f.c {
        *c *= cost_scale;
    }
    let n = f.c.len();
    let m = f.b.len();
    let nu = f.upper.len();
    let mut ls = SparseSolver::new(f.a.clone(), options.solver)?;
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
    if let Some(initial) = initialization::least_squares(&f, &mut ls) {
        q = initial;
    }
    let mut rp = vec![1.0_f64; n];
    let mut rd = vec![1.0_f64; m];
    let mut rg = 1.0_f64;
    let floor = f64::EPSILON.sqrt();
    let tol = options.tolerance;
    let mut status = Status::IterationLimit;
    let mut iterations = 0;
    let mut cert = None;
    let mut metrics = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for k in 0..=options.max_iterations {
        iterations = k;
        if options
            .time_limit
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            status = Status::TimeLimit;
            break;
        }
        let r = residual(&f, &q);
        let ubnorm = f.upper.iter().fold(0.0_f64, |v, (_, u)| v.max(u.abs()));
        let dual_obj = dot(&f.b, &q.y) - bound_dot(&f, &q.w);
        metrics = (
            (norm(&r.p) / (q.tau * (1. + norm(&f.b)))).max(norm(&r.u) / (q.tau * (1. + ubnorm))),
            r.d.iter()
                .zip(&col_scale)
                .fold(0.0_f64, |v, (d, s)| v.max((d / s).abs()))
                / (q.tau * cost_scale * (1. + unscaled_cost_norm)),
            (dot(&f.c, &q.x) - dual_obj).abs() / (cost_scale * q.tau + dual_obj.abs()),
        );
        // Verify original coordinates too, since equilibration can conceal
        // large original-unit row errors.
        let original: Vec<_> = f
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
        let original_p = primal_residual(model, &original);
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
            if let Some((s, c)) = certificate(&f, &q, tol.min(1e-8)) {
                status = s;
                cert = Some(c);
                break;
            }
        }
        if k == options.max_iterations {
            break;
        }
        let tvw: Vec<_> = q.w.iter().zip(&q.v).map(|(w, v)| w / v).collect();
        let mut theta: Vec<_> = q.s.iter().zip(&q.x).map(|(s, x)| s / x).collect();
        for (i, &(j, _)) in f.upper.iter().enumerate() {
            theta[j] += tvw[i];
        }
        let mut factored = false;
        let mut last_error = Error::NotFactorized;
        for _ in 0..3 {
            for x in rp.iter_mut().chain(&mut rd) {
                *x = (*x / 10.).max(floor);
            }
            rg = (rg / 10.).max(floor);
            match ls.factorize(&theta, &rp, &rd) {
                Ok(()) => {
                    factored = true;
                    break;
                }
                Err(e) => last_error = e,
            }
            for x in rp.iter_mut().chain(&mut rd) {
                *x *= 100.;
            }
            rg *= 100.;
        }
        if !factored {
            return Err(last_error);
        }
        let ub: Vec<_> = f.upper.iter().map(|&(_, u)| u).collect();
        let delta = augsys(&ls, &f, &tvw, &f.b, &f.c, &ub)?;
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
            ls: &ls,
            f: &f,
            q: &q,
            tvw: &tvw,
            delta,
            delta0,
        };
        let xs: Vec<_> = q.x.iter().zip(&q.s).map(|(x, s)| -x * s).collect();
        let vw: Vec<_> = q.v.iter().zip(&q.w).map(|(v, w)| -v * w).collect();
        let mu = (q.tau * q.kappa - xs.iter().sum::<f64>() - vw.iter().sum::<f64>())
            / (n + nu + 1) as f64;
        let affine = newton.solve(&r, &xs, &vw, -q.tau * q.kappa)?;
        let one_minus = 1. - q.alpha(&affine);
        // alpha is in [0, 1], so the expression before clipping is at most
        // 0.1. A lower bound of 0.1 would make centering constant and damp
        // even near-full affine steps. Retain strong centering for short
        // steps, but permit faster feasibility progress near a full step.
        let gamma = (one_minus * one_minus * one_minus.min(0.1)).clamp(1e-4, 0.1);
        let damping = 1. - gamma;
        let damped = Residual {
            p: r.p.iter().map(|r| damping * r).collect(),
            u: r.u.iter().map(|r| damping * r).collect(),
            d: r.d.iter().map(|r| damping * r).collect(),
            g: damping * r.g,
        };
        let cx: Vec<_> = xs
            .iter()
            .zip(&affine.x)
            .zip(&affine.s)
            .map(|((xs, x), s)| xs + gamma * mu - x * s)
            .collect();
        let cv: Vec<_> = vw
            .iter()
            .zip(&affine.v)
            .zip(&affine.w)
            .map(|((vw, v), w)| vw + gamma * mu - v * w)
            .collect();
        let mut direction = newton.solve(
            &damped,
            &cx,
            &cv,
            -q.tau * q.kappa + gamma * mu - affine.tau * affine.kappa,
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
            let mut tx: Vec<_> =
                q.x.iter()
                    .zip(&direction.x)
                    .zip(&q.s)
                    .zip(&direction.s)
                    .map(|(((x, dx), s), ds)| {
                        let v = (x + trial * dx) * (s + trial * ds);
                        v.clamp(low, high) - v
                    })
                    .collect();
            let mut tv: Vec<_> =
                q.v.iter()
                    .zip(&direction.v)
                    .zip(&q.w)
                    .zip(&direction.w)
                    .map(|(((v, dv), w), dw)| {
                        let z = (v + trial * dv) * (w + trial * dw);
                        z.clamp(low, high) - z
                    })
                    .collect();
            let z = (q.tau + trial * direction.tau) * (q.kappa + trial * direction.kappa);
            let mut t0 = z.clamp(low, high) - z;
            let mean = (tx.iter().sum::<f64>() + tv.iter().sum::<f64>() + t0) / (n + nu + 1) as f64;
            for t in tx.iter_mut().chain(&mut tv) {
                *t -= mean;
            }
            t0 -= mean;
            let zero = Residual {
                p: vec![0.; m],
                u: vec![0.; nu],
                d: vec![0.; n],
                g: 0.,
            };
            let correction = newton.solve(&zero, &tx, &tv, t0)?;
            let mut candidate = direction.clone();
            // Directions have unrestricted signs, so do not use Point::update.
            for (v, dv) in [
                (&mut candidate.x, &correction.x),
                (&mut candidate.y, &correction.y),
                (&mut candidate.s, &correction.s),
                (&mut candidate.v, &correction.v),
                (&mut candidate.w, &correction.w),
            ] {
                for (v, dv) in v.iter_mut().zip(dv) {
                    *v += dv;
                }
            }
            candidate.tau += correction.tau;
            candidate.kappa += correction.kappa;
            let improved = q.alpha(&candidate);
            if improved > alpha {
                direction = candidate;
                alpha = improved;
            } else {
                break;
            }
        }
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
            .zip(&row_scale)
            .zip(&f.row_sign)
            .map(|((y, r), s)| y * r * s / q.tau / cost_scale)
            .collect();
    let objective = dot(&model.c, &primal);
    if !finite(&primal) || !finite(&dual) || !objective.is_finite() {
        return Err(Error::NumericalFailure("solution recovery overflow".into()));
    }
    Ok(Solution {
        status,
        objective,
        primal_residual: primal_residual(model, &primal),
        dual_residual: metrics.1,
        relative_gap: metrics.2,
        primal,
        dual,
        iterations,
        solver: ls.active_solver_type(),
        certificate: cert,
    })
}

fn primal_residual(model: &ModelData, x: &[f64]) -> f64 {
    let mut residual = 0.0_f64;
    for ((ax, &b), sense) in model.a.mul(x).iter().zip(&model.b).zip(&model.sense) {
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
