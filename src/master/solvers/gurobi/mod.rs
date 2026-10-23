//! Persistent LP master using Gurobi's C API. Requires a licensed installation.
//! Models retain their simplex basis across edits; forks copy the basis explicitly.
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{incomplete, invalid, Result};
use std::ffi::{c_char, c_void, CStr};
use std::ptr::{null, null_mut, NonNull};
use std::time::Duration;

// Signatures follow Gurobi's public gurobi_c.h (32-bit sparse indices).
#[allow(clippy::too_many_arguments)]
extern "C" {
    fn GRBemptyenv(env: *mut *mut c_void) -> i32;
    fn GRBstartenv(env: *mut c_void) -> i32;
    fn GRBfreeenv(env: *mut c_void);
    fn GRBgeterrormsg(env: *mut c_void) -> *const c_char;
    fn GRBnewmodel(
        env: *mut c_void,
        model: *mut *mut c_void,
        name: *const c_char,
        n: i32,
        obj: *const f64,
        lb: *const f64,
        ub: *const f64,
        types: *const c_char,
        names: *const *const c_char,
    ) -> i32;
    fn GRBfreemodel(model: *mut c_void);
    fn GRBgetenv(model: *mut c_void) -> *mut c_void;
    fn GRBcopymodel(model: *mut c_void) -> *mut c_void;
    fn GRBupdatemodel(model: *mut c_void) -> i32;
    fn GRBoptimize(model: *mut c_void) -> i32;
    fn GRBsetintparam(env: *mut c_void, name: *const c_char, value: i32) -> i32;
    fn GRBsetdblparam(env: *mut c_void, name: *const c_char, value: f64) -> i32;
    fn GRBgetdblparam(env: *mut c_void, name: *const c_char, value: *mut f64) -> i32;
    fn GRBgetintattr(model: *mut c_void, name: *const c_char, value: *mut i32) -> i32;
    fn GRBgetdblattr(model: *mut c_void, name: *const c_char, value: *mut f64) -> i32;
    fn GRBgetdblattrarray(
        model: *mut c_void,
        name: *const c_char,
        start: i32,
        len: i32,
        values: *mut f64,
    ) -> i32;
    fn GRBgetintattrarray(
        model: *mut c_void,
        name: *const c_char,
        start: i32,
        len: i32,
        values: *mut i32,
    ) -> i32;
    fn GRBsetintattrarray(
        model: *mut c_void,
        name: *const c_char,
        start: i32,
        len: i32,
        values: *const i32,
    ) -> i32;
    fn GRBsetdblattrarray(
        model: *mut c_void,
        name: *const c_char,
        start: i32,
        len: i32,
        values: *const f64,
    ) -> i32;
    fn GRBsetdblattrelement(model: *mut c_void, name: *const c_char, index: i32, value: f64)
        -> i32;
    fn GRBaddconstrs(
        model: *mut c_void,
        n: i32,
        nz: i32,
        starts: *const i32,
        ids: *const i32,
        values: *const f64,
        senses: *const c_char,
        rhs: *const f64,
        names: *const *const c_char,
    ) -> i32;
    fn GRBaddvars(
        model: *mut c_void,
        n: i32,
        nz: i32,
        starts: *const i32,
        ids: *const i32,
        values: *const f64,
        costs: *const f64,
        lb: *const f64,
        ub: *const f64,
        types: *const c_char,
        names: *const *const c_char,
    ) -> i32;
    fn GRBchgcoeffs(
        model: *mut c_void,
        n: i32,
        rows: *const i32,
        cols: *const i32,
        values: *const f64,
    ) -> i32;
    fn GRBgetvars(
        model: *mut c_void,
        nz: *mut i32,
        starts: *mut i32,
        ids: *mut i32,
        values: *mut f64,
        start: i32,
        len: i32,
    ) -> i32;
    fn GRBdelconstrs(model: *mut c_void, n: i32, ids: *const i32) -> i32;
    fn GRBdelvars(model: *mut c_void, n: i32, ids: *const i32) -> i32;
}
const COLUMN_UPPER: f64 = 1e12;

#[derive(Debug)]
pub struct GurobiMaster {
    model: NonNull<c_void>,
    threads: usize,
    // Shared only by copies made through GRBcopymodel; freed after the last model.
    environment: std::rc::Rc<Environment>,
    pub rhs: Vec<f64>,
    pub senses: Vec<RowSense>,
    pub costs: Vec<f64>,
    pub upper: Vec<f64>,
    pub duals: Vec<f64>,
    pub values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    row_status: Vec<i32>,
    primal_tolerance: f64,
    pub objective: f64,
    pub objective_offset: f64,
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    pub zero_tolerance: f64,
}
#[derive(Debug)]
struct Environment(NonNull<c_void>);
impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY: all models referencing this environment have been freed.
        unsafe { GRBfreeenv(self.0.as_ptr()) }
    }
}
impl Drop for GurobiMaster {
    fn drop(&mut self) {
        // SAFETY: exclusively owned handle, freed before environment field drops.
        unsafe { GRBfreemodel(self.model.as_ptr()) }
    }
}
fn count(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid("Gurobi index/count exceeds i32"))
}
fn sense_code(sense: RowSense) -> c_char {
    (match sense {
        RowSense::Equal => b'=',
        RowSense::LessEqual => b'<',
        RowSense::GreaterEqual => b'>',
    }) as c_char
}
fn entries_of(entries: &[(usize, f64)], size: usize) -> Result<(Vec<i32>, Vec<f64>)> {
    let mut merged = std::collections::BTreeMap::new();
    for &(id, value) in entries {
        if id >= size || !value.is_finite() {
            return Err(invalid("sparse master entry"));
        }
        *merged.entry(count(id)?).or_insert(0.) += value;
    }
    if merged.values().any(|x: &f64| !x.is_finite()) {
        return Err(invalid("nonfinite accumulated coefficient"));
    }
    Ok(merged.into_iter().filter(|&(_, v)| v != 0.).unzip())
}
fn mapping(size: usize, removed: &[usize]) -> Result<(Vec<Option<usize>>, Vec<i32>)> {
    let mut ids = removed.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.iter().any(|&i| i >= size) {
        return Err(invalid("master deletion index"));
    }
    let mut next = 0;
    let map = (0..size)
        .map(|i| {
            if ids.binary_search(&i).is_ok() {
                None
            } else {
                let j = next;
                next += 1;
                Some(j)
            }
        })
        .collect();
    Ok((map, ids.into_iter().map(count).collect::<Result<_>>()?))
}
impl GurobiMaster {
    /// Create a quiet LP environment using all available CPUs and automatic
    /// algorithm selection. Warm bases are retained across model edits.
    pub fn new() -> Result<Self> {
        Self::with_threads(0)
    }

    /// Zero uses all CPUs available to the process. Gurobi chooses the algorithm
    /// and its actual parallelism within this worker budget.
    pub fn with_threads(threads: usize) -> Result<Self> {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        };
        let native_threads = count(threads)?;
        let mut env = null_mut();
        // SAFETY: out-pointer receives an owned environment, managed immediately by RAII.
        let rc = unsafe { GRBemptyenv(&mut env) };
        let env = NonNull::new(env)
            .ok_or_else(|| incomplete(format!("Gurobi environment allocation failed ({rc})")))?;
        let environment = std::rc::Rc::new(Environment(env));
        if rc != 0 {
            return Err(incomplete(format!(
                "Gurobi environment allocation failed ({rc})"
            )));
        }
        // SAFETY: live environment, static NUL-terminated parameter names.
        unsafe {
            for (name, value) in [
                (b"OutputFlag\0".as_slice(), 0),
                (b"Threads\0".as_slice(), native_threads),
                (b"Method\0".as_slice(), -1),
                (b"DualReductions\0".as_slice(), 0),
            ] {
                let rc = GRBsetintparam(env.as_ptr(), name.as_ptr().cast(), value);
                if rc != 0 {
                    return Err(invalid(format!(
                        "Gurobi environment parameter failed ({rc})"
                    )));
                }
            }
            let rc = GRBstartenv(env.as_ptr());
            if rc != 0 {
                return Err(incomplete(format!(
                    "Gurobi start environment: {}",
                    CStr::from_ptr(GRBgeterrormsg(env.as_ptr())).to_string_lossy()
                )));
            }
        }
        let mut model = null_mut();
        // SAFETY: zero variables, null optional arrays; output owns the new model.
        let rc = unsafe {
            GRBnewmodel(
                env.as_ptr(),
                &mut model,
                b"baldes\0".as_ptr().cast(),
                0,
                null(),
                null(),
                null(),
                null(),
                null(),
            )
        };
        let model = NonNull::new(model)
            .ok_or_else(|| incomplete(format!("Gurobi model allocation failed ({rc})")))?;
        let result = Self {
            model,
            threads,
            environment,
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            duals: vec![],
            values: vec![],
            reduced_costs: vec![],
            row_status: vec![],
            primal_tolerance: 1e-7,
            dual_tolerance: 1e-7,
            objective: 0.,
            objective_offset: 0.,
            status: -1,
            solved: false,
            zero_tolerance: 0.,
        };
        result.check(rc, "new model")?;
        result.set_double(b"FeasibilityTol\0", result.primal_tolerance)?;
        result.set_double(b"OptimalityTol\0", result.dual_tolerance)?;
        Ok(result)
    }

    /// Configured worker budget; individual algorithms may use fewer threads.
    pub fn configured_threads(&self) -> usize {
        self.threads
    }
    fn check(&self, rc: i32, operation: &str) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: model and its environment are live; Gurobi owns the error string.
        let message = unsafe { CStr::from_ptr(GRBgeterrormsg(GRBgetenv(self.model.as_ptr()))) }
            .to_string_lossy();
        Err(incomplete(format!("Gurobi {operation} ({rc}): {message}")))
    }
    fn update(&self) -> Result<()> {
        // SAFETY: the model is exclusively owned and never shared between threads.
        unsafe { self.check(GRBupdatemodel(self.model.as_ptr()), "update") }
    }
    fn set_double(&self, name: &[u8], value: f64) -> Result<()> {
        // SAFETY: all private callers supply static NUL-terminated names.
        unsafe {
            self.check(
                GRBsetdblparam(GRBgetenv(self.model.as_ptr()), name.as_ptr().cast(), value),
                "set parameter",
            )
        }
    }
    fn get_parameter(&self, name: &[u8]) -> Result<f64> {
        let mut value = 0.;
        // SAFETY: valid output and static NUL-terminated name.
        unsafe {
            self.check(
                GRBgetdblparam(
                    GRBgetenv(self.model.as_ptr()),
                    name.as_ptr().cast(),
                    &mut value,
                ),
                "get parameter",
            )?;
        }
        Ok(value)
    }
    fn set_element(&self, name: &[u8], index: usize, value: f64) -> Result<()> {
        self.update()?;
        // SAFETY: callers validate index against the attribute dimensions.
        unsafe {
            self.check(
                GRBsetdblattrelement(
                    self.model.as_ptr(),
                    name.as_ptr().cast(),
                    count(index)?,
                    value,
                ),
                "set attribute",
            )
        }
    }
    fn int_attr(&self, name: &[u8]) -> Result<i32> {
        let mut value = 0;
        // SAFETY: live model, valid output and static NUL-terminated name.
        unsafe {
            self.check(
                GRBgetintattr(self.model.as_ptr(), name.as_ptr().cast(), &mut value),
                "integer attribute",
            )?;
        }
        Ok(value)
    }
    fn double_attr(&self, name: &[u8]) -> Result<f64> {
        let mut value = 0.;
        // SAFETY: live model, valid output and static NUL-terminated name.
        unsafe {
            self.check(
                GRBgetdblattr(self.model.as_ptr(), name.as_ptr().cast(), &mut value),
                "double attribute",
            )?;
        }
        Ok(value)
    }
    fn double_array(&self, name: &[u8], len: usize) -> Result<Vec<f64>> {
        let mut values = vec![0.; len];
        if len > 0 {
            // SAFETY: output has exactly len slots; callers use the attribute's dimension.
            unsafe {
                self.check(
                    GRBgetdblattrarray(
                        self.model.as_ptr(),
                        name.as_ptr().cast(),
                        0,
                        count(len)?,
                        values.as_mut_ptr(),
                    ),
                    "solution array",
                )?;
            }
        }
        Ok(values)
    }
    pub fn set_rhs(&mut self, rhs: &[f64]) -> Result<()> {
        if rhs.len() != self.rhs.len() || rhs.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Gurobi RHS dimensions/values"));
        }
        self.update()?;
        if !rhs.is_empty() {
            // SAFETY: one value per live constraint.
            unsafe {
                self.check(
                    GRBsetdblattrarray(
                        self.model.as_ptr(),
                        b"RHS\0".as_ptr().cast(),
                        0,
                        count(rhs.len())?,
                        rhs.as_ptr(),
                    ),
                    "set RHS",
                )?;
            }
        }
        self.rhs.copy_from_slice(rhs);
        self.solved = false;
        Ok(())
    }
    #[cfg(feature = "clp")]
    pub(crate) fn set_lp_iteration_limit(&mut self, iterations: usize) -> Result<()> {
        self.set_double(b"IterationLimit\0", iterations as f64)
    }
    pub fn simplex_iterations(&self) -> Result<usize> {
        Ok(self.double_attr(b"IterCount\0")?.max(0.) as usize)
    }
    /// Independent mathematical model and basis. Does not share mutable model state.
    pub fn try_clone(&self) -> Result<Self> {
        self.update()?;
        // SAFETY: pending updates flushed; copy owns its own native model.
        let model = NonNull::new(unsafe { GRBcopymodel(self.model.as_ptr()) })
            .ok_or_else(|| incomplete("Gurobi model copy failed"))?;
        let copy = Self {
            model,
            threads: self.threads,
            environment: self.environment.clone(),
            rhs: self.rhs.clone(),
            senses: self.senses.clone(),
            costs: self.costs.clone(),
            upper: self.upper.clone(),
            duals: self.duals.clone(),
            values: self.values.clone(),
            reduced_costs: self.reduced_costs.clone(),
            row_status: self.row_status.clone(),
            primal_tolerance: self.primal_tolerance,
            dual_tolerance: self.dual_tolerance,
            objective: self.objective,
            objective_offset: self.objective_offset,
            status: self.status,
            solved: self.solved,
            zero_tolerance: self.zero_tolerance,
        };
        // Copy available native basis even if an edit invalidated the solution cache.
        let mut vb = vec![0; self.costs.len()];
        let mut cb = vec![0; self.rhs.len()];
        // SAFETY: arrays match the live model dimensions. Missing basis is normal before first solve.
        unsafe {
            let vrc = if vb.is_empty() {
                0
            } else {
                GRBgetintattrarray(
                    self.model.as_ptr(),
                    b"VBasis\0".as_ptr().cast(),
                    0,
                    count(vb.len())?,
                    vb.as_mut_ptr(),
                )
            };
            let crc = if cb.is_empty() {
                0
            } else {
                GRBgetintattrarray(
                    self.model.as_ptr(),
                    b"CBasis\0".as_ptr().cast(),
                    0,
                    count(cb.len())?,
                    cb.as_mut_ptr(),
                )
            };
            if vrc == 0 && crc == 0 {
                if !vb.is_empty() {
                    copy.check(
                        GRBsetintattrarray(
                            copy.model.as_ptr(),
                            b"VBasis\0".as_ptr().cast(),
                            0,
                            count(vb.len())?,
                            vb.as_ptr(),
                        ),
                        "copy column basis",
                    )?;
                }
                if !cb.is_empty() {
                    copy.check(
                        GRBsetintattrarray(
                            copy.model.as_ptr(),
                            b"CBasis\0".as_ptr().cast(),
                            0,
                            count(cb.len())?,
                            cb.as_ptr(),
                        ),
                        "copy row basis",
                    )?;
                }
            }
        }
        Ok(copy)
    }
}
mod interface;
