use super::*;

#[test]
fn augmented_refinement_controls_all_regularized_hsd_newton_equations() {
    let model = ModelData {
        a: super::super::SparseMatrix::from_triplets(
            2,
            3,
            &[(0, 0, 1.), (0, 1, 2.), (1, 1, 1.), (1, 2, 3.)],
        )
        .unwrap(),
        b: vec![3., 4.],
        c: vec![1., -2., 0.5],
        lower: vec![0.; 3],
        upper: vec![4., f64::INFINITY, 5.],
        sense: vec![super::super::Sense::Equal; 2],
    };
    let f = model.standard_form().unwrap();
    let q = initialization::unit(&f);
    let mut r = Residual::default();
    residual_into(&f, &q, &mut r);
    let xs: Vec<_> = q.x.iter().zip(&q.s).map(|(x, s)| -x * s).collect();
    let vw: Vec<_> = q.v.iter().zip(&q.w).map(|(v, w)| -v * w).collect();
    let tvw: Vec<_> = q.w.iter().zip(&q.v).map(|(w, v)| w / v).collect();
    let upper: Vec<_> = f.upper.iter().map(|&(_, u)| u).collect();
    let tk = -q.tau * q.kappa;
    let check = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 1e-10 * (1. + actual.abs() + expected.abs()),
            "actual={actual} expected={expected}"
        );
    };
    for backend in [SolverType::HiPO, SolverType::NormalEquations] {
        for rho in [1e-8, 1e-4] {
            let rp = vec![rho; f.c.len()];
            let rd = vec![2. * rho; f.b.len()];
            let rg = 3. * rho;
            let mut theta: Vec<_> = q.s.iter().zip(&q.x).map(|(s, x)| s / x).collect();
            for (i, &(j, _)) in f.upper.iter().enumerate() {
                theta[j] += tvw[i];
            }
            let mut ls = SparseSolver::new(f.a.clone(), backend).unwrap();
            ls.factorize(&theta, &rp, &rd).unwrap();
            let mut workspace = super::super::sparse_solver::SolveWorkspace::default();
            let mut delta = (Vec::new(), Vec::new(), Vec::new());
            augsys_into(
                &ls,
                &f,
                &tvw,
                &f.b,
                &f.c,
                &upper,
                &mut workspace,
                &mut delta.0,
                &mut delta.1,
                &mut delta.2,
            )
            .unwrap();
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
            let newton = Newton {
                ls: &ls,
                f: &f,
                q: &q,
                tvw: &tvw,
                delta: &delta,
                delta0,
            };
            let mut out = Point::default();
            newton
                .solve_into(
                    &r,
                    &xs,
                    &vw,
                    tk,
                    &mut workspace,
                    &mut newton::NewtonRhs::default(),
                    &mut out,
                )
                .unwrap();
            let ax = f.a.mul(&out.x);
            for i in 0..f.b.len() {
                check(ax[i] + rd[i] * out.y[i] - f.b[i] * out.tau, r.p[i]);
            }
            let mut stationarity = f.a.tmul(&out.y);
            for (i, &(j, u)) in f.upper.iter().enumerate() {
                check(out.x[j] + out.v[i] - u * out.tau, r.u[i]);
                check(q.w[i] * out.v[i] + q.v[i] * out.w[i], vw[i]);
                stationarity[j] -= out.w[i];
            }
            for j in 0..f.c.len() {
                check(
                    stationarity[j] + out.s[j] - f.c[j] * out.tau - rp[j] * out.x[j],
                    r.d[j],
                );
                check(q.s[j] * out.x[j] + q.x[j] * out.s[j], xs[j]);
            }
            check(
                dot(&f.c, &out.x) - dot(&f.b, &out.y) + bound_dot(&f, &out.w) + out.kappa
                    - rg * out.tau,
                -r.g,
            );
            check(q.kappa * out.tau + q.tau * out.kappa, tk);
        }
    }
}

#[test]
fn retained_newton_outputs_overwrite_poison_after_growth_and_shrink() {
    let mut scratch = newton::NewtonRhs::default();
    let mut out = Point::default();
    let mut workspace = super::super::sparse_solver::SolveWorkspace::default();
    for columns in [2, 7, 1, 4] {
        for bounded in [true, false] {
            let model = ModelData {
                a: super::super::SparseMatrix::from_triplets(
                    1,
                    columns,
                    &(0..columns).map(|j| (0, j, 1.)).collect::<Vec<_>>(),
                )
                .unwrap(),
                b: vec![1.],
                c: vec![2.; columns],
                lower: vec![0.; columns],
                upper: vec![if bounded { 10. } else { f64::INFINITY }; columns],
                sense: vec![super::super::Sense::Equal],
            };
            let f = model.standard_form().unwrap();
            let q = initialization::unit(&f);
            let mut ls = SparseSolver::new(f.a.clone(), SolverType::HiPO).unwrap();
            ls.factorize(&vec![2.; columns], &vec![1e-6; columns], &[1e-6])
                .unwrap();
            let tvw = vec![1.; f.upper.len()];
            let delta = (vec![0.; columns], vec![0.], vec![0.; f.upper.len()]);
            let newton = Newton {
                ls: &ls,
                f: &f,
                q: &q,
                tvw: &tvw,
                delta: &delta,
                delta0: 1.,
            };
            let mut r = Residual::default();
            residual_into(&f, &q, &mut r);
            let xs = vec![-1.; columns];
            let vw = vec![-1.; f.upper.len()];
            let mut expected = Point::default();
            newton
                .solve_into(
                    &r,
                    &xs,
                    &vw,
                    -1.,
                    &mut workspace,
                    &mut scratch,
                    &mut expected,
                )
                .unwrap();
            for values in [&mut out.x, &mut out.y, &mut out.s, &mut out.v, &mut out.w] {
                values.resize(20, f64::NAN);
                values.fill(f64::NAN);
            }
            out.tau = f64::NAN;
            out.kappa = f64::NAN;
            newton
                .solve_into(&r, &xs, &vw, -1., &mut workspace, &mut scratch, &mut out)
                .unwrap();
            for (actual, expected) in [
                (&out.x, &expected.x),
                (&out.y, &expected.y),
                (&out.s, &expected.s),
                (&out.v, &expected.v),
                (&out.w, &expected.w),
            ] {
                assert_eq!(actual.len(), expected.len());
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(actual.capacity() >= 20);
            }
            assert_eq!(out.tau.to_bits(), expected.tau.to_bits());
            assert_eq!(out.kappa.to_bits(), expected.kappa.to_bits());
        }
    }
}

#[test]
fn weighted_correction_step_bounds_match_materialized_directions() {
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
        let weights = [0.0625, 0.5, 1.];
        let alphas = q.alphas_with_weighted_correction(&direction, &correction, &weights);
        for (alpha, weight) in alphas.into_iter().zip(weights) {
            let mut combined = direction.clone();
            combined.add_scaled_direction(&correction, weight);
            assert_eq!(alpha.to_bits(), q.alpha(&combined).to_bits());
        }
    }
}
