use super::*;

pub(super) fn scale(f: &mut StandardForm, passes: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut rows = vec![1.; f.b.len()];
    let mut cols = vec![1.; f.c.len()];
    for _ in 0..passes {
        let mut rn = vec![0.0_f64; rows.len()];
        let mut cn = vec![0.0_f64; cols.len()];
        for (j, cj) in cn.iter_mut().enumerate() {
            for (i, v) in f.a.column(j) {
                rn[i] = rn[i].max(v.abs());
                *cj = cj.max(v.abs());
            }
        }
        for v in rn.iter_mut().chain(&mut cn) {
            *v = if *v > 0. { 1. / v.sqrt() } else { 1. };
            // Powers of two make scaling and its recovery exact.
            *v = v.log2().round().exp2();
        }
        for (j, &cj) in cn.iter().enumerate() {
            for p in f.a.ptr[j]..f.a.ptr[j + 1] {
                f.a.val[p] *= rn[f.a.idx[p]] * cj;
            }
        }
        for ((b, r), &s) in f.b.iter_mut().zip(&mut rows).zip(&rn) {
            *b *= s;
            *r *= s;
        }
        for ((c, col), &s) in f.c.iter_mut().zip(&mut cols).zip(&cn) {
            *c *= s;
            *col *= s;
        }
        for (j, u) in &mut f.upper {
            *u /= cn[*j];
        }
    }
    if !finite(&f.a.val)
        || !finite(&f.b)
        || !finite(&f.c)
        || !finite(&rows)
        || !finite(&cols)
        || rows.iter().chain(&cols).any(|&v| v <= 0.)
        || f.upper.iter().any(|&(_, u)| !u.is_finite() || u <= 0.)
    {
        return Err(Error::NumericalFailure(
            "equilibration overflow or underflow".into(),
        ));
    }
    Ok((rows, cols))
}

pub(super) fn certificate(f: &StandardForm, q: &Point, tol: f64) -> Option<(Status, Vec<f64>)> {
    let mut aty = f.a.tmul(&q.y);
    for (i, &(j, _)) in f.upper.iter().enumerate() {
        aty[j] -= q.w[i];
    }
    let violation = aty.iter().fold(0.0_f64, |v, a| v.max(*a));
    let dual_obj = dot(&f.b, &q.y) - bound_dot(f, &q.w);
    let ynorm = norm(&q.y).max(norm(&q.w));
    if dual_obj > 0. && ynorm > 0. && violation <= tol * ynorm && dual_obj > tol * ynorm {
        let mut cert = q.y.clone();
        cert.extend(&q.w);
        for x in &mut cert {
            *x /= dual_obj;
        }
        return Some((Status::Infeasible, cert));
    }
    let ax = norm(&f.a.mul(&q.x));
    let cx = dot(&f.c, &q.x);
    let xn = norm(&q.x);
    let bounded = f
        .upper
        .iter()
        .fold(0.0_f64, |v, &(j, _)| v.max(q.x[j].abs()));
    if cx < 0. && xn > 0. && ax.max(bounded) <= tol * xn && -cx > tol * xn {
        return Some((Status::Unbounded, q.x.iter().map(|x| x / (-cx)).collect()));
    }
    None
}
