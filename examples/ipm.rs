use baldes::ipm::{solve, ModelData, Options, Sense, SparseMatrix, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize -x - 2y, with x + y <= 3, x >= 1, and x,y >= 0.
    let model = ModelData {
        a: SparseMatrix::from_triplets(2, 2, &[(0, 0, 1.), (0, 1, 1.), (1, 0, 1.)])?,
        b: vec![3., 1.],
        c: vec![-1., -2.],
        lower: vec![0., 0.],
        upper: vec![f64::INFINITY; 2],
        sense: vec![Sense::LessEqual, Sense::GreaterEqual],
    };
    let result = solve(&model, &Options::default())?;
    println!(
        "{:?}: objective = {:.8}, x = {:?} ({} iterations)",
        result.status, result.objective, result.primal, result.iterations
    );
    if result.status != Status::Optimal {
        return Err("LP did not converge".into());
    }
    Ok(())
}
