//! Cut families and their shared registry and pricing state.
//!
//! Family implementations live in `capacity`, `knapsack`, `kpath` and
//! `rank_one`; common graph algorithms and native containers live in `shared`.
//! Historical module paths are re-exported for existing callers.
pub mod capacity;
pub mod knapsack;
pub mod kpath;
pub mod rank_one;
pub mod shared;

mod registry;

pub use registry::{
    coefficient, memory_policy, route_arcs, CandidateMode, CutClass, CutRegistry, FractionalColumn,
    MemoryKind, MemoryPolicy, MemoryType, RankOneCut, SeparationParameters,
};

// Compatibility with the original flat module layout.
pub use rank_one::{local_search, separation, subcuts, tuple};
pub use shared::{bits, mincut, pulse, rbtree, state, stdhash, stdsort};
