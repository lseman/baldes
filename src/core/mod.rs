//! Shared errors, logging, and solver options.
pub mod error;
pub mod log;
pub mod options;

// Shared by pricing envelopes and master certificates.
pub(crate) mod rounding;
