# Build and validation

[Documentation index](README.md) · [Project setup](../README.md#build)

Run commands from the repository root. Keep feature selection explicit when
checking behavior that depends on an LP or integer backend.

## Build matrix

| Features | Dependencies | Available components |
| --- | --- | --- |
| Default | Rust toolchain | Core pricing, cuts, enumeration, instance utilities, and HGS library |
| `clp` | C++14 compiler; initialized Clp/CoinUtils submodules | Clp master, routing root/tree, and `baldes` CLI |
| `highs` | Linkable HiGHS shared library with 32-bit `HighsInt` | HiGHS LP backend and generic integer solve |
| `clp,highs` | Both dependencies above | Routing tree plus optional enumerated HiGHS MIP |

```sh
git submodule update --init
cargo build --release --features clp,highs
```

HiGHS discovery checks `HIGHS_DIR/lib`, `vendor/highs/lib`, `pkg-config`, and
the system linker path. The shared library must also be available to the
runtime loader. See [build.rs](../build.rs) for native linking and Clp source
configuration.

## Full Rust checks

```sh
cargo test --workspace --all-features
cargo check --workspace --all-targets --all-features
cargo check --workspace --all-targets
```

The all-feature suite requires both backends. A build without `highs` rejects
requests for enumerated MIP; a passing default-feature build alone does not
exercise that path.

## Focused checks

| Area | Integration test targets |
| --- | --- |
| Branch candidates, pricing constraints, tree bounds | `branching`, `branch_price` |
| Forks, labels, enumeration, siblings, descendants | `node_lifecycle` |
| Capacity/Kohl and strong k-path/knapsack integration | `capacity_root`, `kpath_knapsack` |
| Supported option discovery, validation, atomic setters | `options` |
| Model transfer, integer outcomes, complete-node proofs | `highs_mip` |
| Argument handling and JSON output | `cli` |
| Input parsing and distance conventions | `cvrplib` |

```sh
cargo test --features clp --test branching --test branch_price --test node_lifecycle
cargo test --features clp --test capacity_root --test kpath_knapsack --test options
cargo test --features clp,highs --test highs_mip --test cli
```

Check backend boundaries as well as the combined build:

```sh
cargo test --features clp --test options --test cli
cargo test --no-default-features --features highs --test highs_mip --test options
```

Feature-gated tests run only when their backend is present. These commands
exercise different subsets, so compare coverage as well as pass counts.

## Python

```sh
python -m pip install '.[test]'
python -m pytest python/tests
```

The default extension build includes Clp. Rebuild/reinstall the extension after
changing Rust or PyO3 code; tests against an old installed wheel do not validate
the current sources. `pybaldes.has_clp` reports availability. Python exposes
root relaxations and low-level pricing, but no routing-tree entry point.

## What the checks establish

Small integer tests compare against exhaustive route or variable enumeration.
Lifecycle tests verify parent/sibling isolation and inherited branch/cut rows.
MIP tests distinguish feasible witnesses from proof outcomes and check that
limits do not certify an unfinished node.

Passing these checks establishes the cases exercised by the suite. Benchmark
timings depend on the instance, graph conventions, enabled cuts, backend, and
hardware. Preserve that configuration when reporting a performance change.
For documentation-only edits, verify local links, API examples, command flags,
and option tables without treating a full solver rerun as necessary.
