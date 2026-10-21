//! Thread-bound owned node handles; no borrowed C pointers cross this API.
use baldes::master::multi_pricing::{
    self,
    format::{AddRow, Model, SolveOptions, SolveResult},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::time::Instant;

fn error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "_NetworkNode", module = "pybaldes._core", unsendable)]
struct NetworkNode {
    snapshot: multi_pricing::NodeSnapshot,
}

fn pack_root(output: multi_pricing::Output, started: Instant) -> PyResult<(String, NetworkNode)> {
    let json = SolveResult::root(&output, started.elapsed().as_secs_f64())
        .to_json()
        .map_err(error)?;
    Ok((
        json,
        NetworkNode {
            snapshot: output.snapshot,
        },
    ))
}

#[pymethods]
impl NetworkNode {
    fn solve(&self, options_json: &str) -> PyResult<(String, NetworkNode)> {
        let options = SolveOptions::from_json(options_json).map_err(error)?;
        if options.tree {
            return Err(error(
                "resume solves a node; tree=True is only supported for a fresh model",
            ));
        }
        let options = options.node_options().map_err(error)?;
        let started = Instant::now();
        let output = self.snapshot.solve(options).map_err(error)?;
        pack_root(output, started)
    }
    fn branch_assignment(&self, shared_row: usize, network: i32, assigned: bool) -> PyResult<Self> {
        Ok(Self {
            snapshot: self
                .snapshot
                .branch_assignment(shared_row, network, assigned)
                .map_err(error)?,
        })
    }
    fn with_row(&self, row_json: &str) -> PyResult<Self> {
        let input = AddRow::from_json(row_json).map_err(error)?;
        let coefficients = input
            .coefficients
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        Ok(Self {
            snapshot: self
                .snapshot
                .with_row(input.row.row(), &coefficients)
                .map_err(error)?,
        })
    }
    fn fix_route(&self, route_index: usize, quantity: f64) -> PyResult<Self> {
        Ok(Self {
            snapshot: self
                .snapshot
                .fix_route(route_index, quantity)
                .map_err(error)?,
        })
    }
    fn solve_enumerated_mip(
        &self,
        time_limit: f64,
        node_limit: usize,
        objective_cutoff: Option<f64>,
    ) -> PyResult<Option<String>> {
        let started = Instant::now();
        let options = baldes::master::solvers::mip::MipOptions {
            time_limit: std::time::Duration::try_from_secs_f64(time_limit).map_err(error)?,
            node_limit,
            objective_cutoff: objective_cutoff.unwrap_or(f64::INFINITY),
        };
        self.snapshot
            .solve_enumerated_mip(&options)
            .map_err(error)?
            .map(|output| {
                SolveResult::mip(
                    &output,
                    self.snapshot.enumeration_report(),
                    started.elapsed().as_secs_f64(),
                )
                .to_json()
                .map_err(error)
            })
            .transpose()
    }
    fn fractional_assignments(&self, tolerance: f64) -> PyResult<Vec<(usize, i32, f64)>> {
        Ok(self
            .snapshot
            .fractional_assignments(tolerance)
            .map_err(error)?
            .into_iter()
            .map(|c| (c.shared_row, c.network, c.value))
            .collect())
    }
}

#[pyfunction]
fn _solve_networks(
    py: Python<'_>,
    model_json: &str,
    options_json: &str,
) -> PyResult<(String, Option<NetworkNode>)> {
    let options = SolveOptions::from_json(options_json).map_err(error)?;
    let problem = Model::from_json(model_json)
        .and_then(Model::problem::<{ baldes::DEFAULT_RESOURCE_DIMENSIONS }>)
        .map_err(error)?;
    let started = Instant::now();
    if options.tree {
        let options = options.tree_options().map_err(error)?;
        let json = py
            .allow_threads(move || -> baldes::Result<String> {
                let output = multi_pricing::solve_assignment_tree(problem, options)?;
                SolveResult::tree(&output, started.elapsed().as_secs_f64()).to_json()
            })
            .map_err(error)?;
        Ok((json, None))
    } else {
        let output =
            multi_pricing::solve(problem, options.node_options().map_err(error)?).map_err(error)?;
        let (json, node) = pack_root(output, started)?;
        Ok((json, Some(node)))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NetworkNode>()?;
    m.add_function(wrap_pyfunction!(_solve_networks, m)?)?;
    Ok(())
}
