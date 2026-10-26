//! One Python solve transport with model compilation adapters.
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
#[cfg(feature = "clp")]
mod networks;
mod routing;

#[cfg(feature = "clp")]
#[pyfunction]
fn _solve(
    py: Python<'_>,
    model: &Bound<'_, PyDict>,
    options: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let kind: String = model
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("model kind is required"))?
        .extract()?;
    let data = model
        .get_item("data")?
        .ok_or_else(|| PyValueError::new_err("model data is required"))?
        .cast_into::<PyDict>()?;
    match kind.as_str() {
        "routing" => {
            let kwargs = data.copy()?;
            for (key, value) in options.iter() {
                if kwargs.contains(&key)? {
                    return Err(PyValueError::new_err(
                        "solve options cannot replace model data",
                    ));
                }
                kwargs.set_item(key, value)?;
            }
            Ok(wrap_pyfunction!(routing::solve_routing, py)?
                .call((), Some(&kwargs))?
                .unbind())
        }
        "networks" => {
            let json = PyModule::import(py, "json")?;
            let model_json: String = json.call_method1("dumps", (&data,))?.extract()?;
            let options_json: String = json.call_method1("dumps", (options,))?.extract()?;
            Ok(wrap_pyfunction!(networks::solve_networks, py)?
                .call1((model_json, options_json))?
                .unbind())
        }
        _ => Err(PyValueError::new_err("unknown model kind")),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    routing::register(m)?;
    #[cfg(feature = "clp")]
    {
        networks::register(m)?;
        m.add_function(wrap_pyfunction!(_solve, m)?)?;
    }
    Ok(())
}
