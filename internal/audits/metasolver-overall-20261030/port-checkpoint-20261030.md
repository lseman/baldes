# MetaSolver paired enumeration port: checkpoint

This checkpoint records work in progress. It does not claim full parity with the
pinned reference binary (`103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6`).

- Integrated paired-front completion queries and paired label-pair fixing for
  supported resource and cut configurations. Completion retains labels when
  its bounds or source-to-machine mapping cannot justify a rejection.
- Added machine-front preparation, transition, concatenation and reconstruction
  paths, including original waiting coefficients and retained per-path machine
  states. Raw machine values remain distinct from source-model objectives.
- Integrated branch state, presence and knapsack handling into enumeration;
  added cut-row projection by original cut identity, path inspection, and
  interruption and stale-state handling.
- Added machine endpoint-domain hulls for completion and crossing joins, with
  conservative numerical comparisons and separate objective bounds.
- Added custom-resource callback state, indexed raw joins and path recording,
  plus captured paired-enumeration fixtures and regression cases.
- Gated Clp-dependent stochastic modules in default builds and exposed the
  source-control certificate code without requiring Clp.

The general source objective mapping for nonconstant or waiting machine fronts,
mixed covering-cut completion bounds, full candidate ordering and interruption
parity, and custom-resource runtime cardinalities remain open. An explicitly
supplied machine program can differ from the source recurrence; matching raw
machine values alone does not prove source objective equivalence.

Both default and Clp-enabled `cargo check -p baldes` succeeded during this work.
Earlier in the work, the Clp coupled-resource target passed 38 tests before the
latest integration changes. Those changes have not been regression-tested or
compared against the pinned binary, and formatting has been deferred. Existing
tests that assert engine disposal after enumeration interruption may need updates
to reflect the new retained-engine behavior.
