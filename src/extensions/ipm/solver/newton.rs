use super::super::sparse_solver::SolveWorkspace;
use super::*;

#[derive(Debug, Clone, Default)]
pub(super) struct NewtonScratch {
    pub(super) rhs: NewtonRhs,
    pub(super) delta: (Vec<f64>, Vec<f64>, Vec<f64>),
    pub(super) affine: Point,
    pub(super) direction: Point,
    pub(super) correction: Point,
}

#[derive(Debug, Clone, Default)]
pub(super) struct NewtonRhs {
    d: Vec<f64>,
    u: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn augsys_into(
    ls: &SparseSolver,
    f: &StandardForm,
    tvw: &[f64],
    p: &[f64],
    d: &[f64],
    u: &[f64],
    workspace: &mut SolveWorkspace,
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    z: &mut Vec<f64>,
) -> Result<()> {
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
    x.clear();
    x.extend_from_slice(&sol[..f.c.len()]);
    y.clear();
    y.extend_from_slice(&sol[f.c.len()..]);
    z.clear();
    z.extend(
        f.upper
            .iter()
            .enumerate()
            .map(|(i, &(j, _))| (x[j] - u[i]) * tvw[i]),
    );
    Ok(())
}

pub(super) struct Newton<'a> {
    pub(super) ls: &'a SparseSolver,
    pub(super) f: &'a StandardForm,
    pub(super) q: &'a Point,
    pub(super) tvw: &'a [f64],
    pub(super) delta: &'a (Vec<f64>, Vec<f64>, Vec<f64>),
    pub(super) delta0: f64,
}
impl Newton<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_into(
        &self,
        r: &Residual,
        xs: &[f64],
        vw: &[f64],
        tk: f64,
        workspace: &mut SolveWorkspace,
        scratch: &mut NewtonRhs,
        out: &mut Point,
    ) -> Result<()> {
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
        augsys_into(
            self.ls, f, self.tvw, &r.p, &scratch.d, &scratch.u, workspace, &mut out.x, &mut out.y,
            &mut out.w,
        )?;
        let Point { x, y, w, s, v, .. } = out;
        let tau = (r.g + tk / q.tau + dot(&f.c, x) - dot(&f.b, y) + bound_dot(f, w)) / self.delta0;
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
        s.clear();
        s.extend(
            xs.iter()
                .zip(&q.s)
                .zip(x.iter())
                .zip(&q.x)
                .map(|(((xs, s), dx), x)| (xs - s * dx) / x),
        );
        v.clear();
        v.extend(
            vw.iter()
                .zip(&q.v)
                .zip(w.iter())
                .zip(&q.w)
                .map(|(((vw, v), dw), w)| (vw - v * dw) / w),
        );
        out.tau = tau;
        out.kappa = kappa;
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
        Ok(())
    }
}
