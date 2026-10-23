use super::super::sparse_solver::SolveWorkspace;
use super::*;

#[derive(Debug, Clone, Default)]
pub(super) struct NewtonScratch {
    d: Vec<f64>,
    u: Vec<f64>,
}

pub(super) fn augsys(
    ls: &SparseSolver,
    f: &StandardForm,
    tvw: &[f64],
    p: &[f64],
    d: &[f64],
    u: &[f64],
    workspace: &mut SolveWorkspace,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut rhs = std::mem::take(&mut workspace.rhs);
    rhs.clear();
    rhs.extend_from_slice(d);
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        rhs[j] -= u[i] * tvw[i];
    }
    rhs.extend_from_slice(p);
    let result = ls.solve_into(&rhs, workspace);
    workspace.rhs = rhs;
    result?;
    let sol = &workspace.solution;
    let x = sol[..f.c.len()].to_vec();
    let y = sol[f.c.len()..].to_vec();
    let z = f
        .upper
        .iter()
        .enumerate()
        .map(|(i, &(j, _))| (x[j] - u[i]) * tvw[i])
        .collect();
    Ok((x, y, z))
}

pub(super) struct Newton<'a> {
    pub(super) ls: &'a SparseSolver,
    pub(super) f: &'a StandardForm,
    pub(super) q: &'a Point,
    pub(super) tvw: &'a [f64],
    pub(super) delta: (Vec<f64>, Vec<f64>, Vec<f64>),
    pub(super) delta0: f64,
}
impl Newton<'_> {
    pub(super) fn solve(
        &self,
        r: &Residual,
        xs: &[f64],
        vw: &[f64],
        tk: f64,
        workspace: &mut SolveWorkspace,
        scratch: &mut NewtonScratch,
    ) -> Result<Point> {
        let q = self.q;
        let f = self.f;
        scratch.d.clear();
        scratch
            .d
            .extend(r.d.iter().zip(xs).zip(&q.x).map(|((d, xs), x)| d - xs / x));
        scratch.u.clear();
        scratch
            .u
            .extend(r.u.iter().zip(vw).zip(&q.w).map(|((u, vw), w)| u - vw / w));
        let (mut x, mut y, mut w) = augsys(
            self.ls, f, self.tvw, &r.p, &scratch.d, &scratch.u, workspace,
        )?;
        let tau =
            (r.g + tk / q.tau + dot(&f.c, &x) - dot(&f.b, &y) + bound_dot(f, &w)) / self.delta0;
        let kappa = (tk - q.kappa * tau) / q.tau;
        for (xi, di) in x.iter_mut().zip(&self.delta.0) {
            *xi += tau * di;
        }
        for (yi, di) in y.iter_mut().zip(&self.delta.1) {
            *yi += tau * di;
        }
        for (wi, di) in w.iter_mut().zip(&self.delta.2) {
            *wi += tau * di;
        }
        let s = xs
            .iter()
            .zip(&q.s)
            .zip(&x)
            .zip(&q.x)
            .map(|(((xs, s), dx), x)| (xs - s * dx) / x)
            .collect();
        let v = vw
            .iter()
            .zip(&q.v)
            .zip(&w)
            .zip(&q.w)
            .map(|(((vw, v), dw), w)| (vw - v * dw) / w)
            .collect();
        let out = Point {
            x,
            y,
            s,
            v,
            w,
            tau,
            kappa,
        };
        if [&out.x, &out.y, &out.s, &out.v, &out.w]
            .iter()
            .any(|v| !finite(v))
            || !tau.is_finite()
            || !kappa.is_finite()
        {
            return Err(Error::NumericalFailure(
                "non-finite Newton direction".into(),
            ));
        }
        Ok(out)
    }
}
