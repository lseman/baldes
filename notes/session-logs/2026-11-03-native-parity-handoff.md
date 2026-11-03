# Native parity handoff — 2026-11-03

The current handoff and next experiment are in `../../internal/handoffs/native-parity-20261103.md`; detailed captures are in `../../internal/audits/function-parity-20261102/README.md`. Both internal paths are Git-ignored and need separate preservation when transferring the work.

Arc-defined Ryan–Foster branching and ordered strong-branch candidate filtering are implemented and covered by native captures. The unresolved tree divergence occurs during the tenth root LP solve: a controlled replay attributes the captured difference to column order. The first pricing order difference arises from native rounded versus Rust raw exact costs. Rust's raw search preserves the exact-pricing certificate; the native ordering has not yet been reproduced without weakening that guarantee.

The Clp branching target passed 10/10 and the pricing porting target passed 70/70. Strong-branch filtering and captured tolerance tests passed. These checks do not establish whole-tree parity.

The session-log skill's `/log` command was not available in this environment, so this short log was written directly in its specified directory.
