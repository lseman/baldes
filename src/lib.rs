//! BALDES 2.0: resource-constrained shortest path pricing and the
//! column-and-cut generation root, in Rust.
//!
//! Resource/NG preparation, bucket labeling, pricing, fixing/enumeration,
//! limited-memory rank-one cuts (`cuts`), and the optional Clp root master
//! (`--features clp`). The reference implementation is the C++ OpenRSCP
//! RCSP engine.
// Scalar capacity used when a const dimension is not explicitly selected.
// Python bindings use this library build's capacity.
include!(concat!(env!("OUT_DIR"), "/resource_dimensions.rs"));
pub mod core;
pub mod cuts;
pub mod enumeration;
/// Optional algorithms, numerical backends and specialized routing models.
pub mod extensions;
// Compatibility paths for callers predating the extensions namespace.
pub use extensions::{evrp, hgs, ipm};
pub mod graph;
pub mod labeling;
pub mod master;
pub mod pricing;
pub use graph::{Graph, ResourceKind, Window};
pub use labeling::{price, Controls, PricingResult, Route, Statistics};

pub type Result<T> = std::result::Result<T, core::error::OrError>;

pub(crate) fn invalid(message: impl Into<String>) -> core::error::OrError {
    core::error::OrError {
        kind: core::error::ErrorKind::InvalidArgument,
        message: message.into(),
    }
}
pub(crate) fn incomplete(message: impl Into<String>) -> core::error::OrError {
    core::error::OrError {
        kind: core::error::ErrorKind::Runtime,
        message: message.into(),
    }
}

pub mod inputs;
