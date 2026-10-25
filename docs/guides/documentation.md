# Maintaining the documentation

[Documentation index](../README.md) · [Source layout](source-layout.md)

## Documentation organization

| Directory | Reader and purpose |
| --- | --- |
| `guides/` | Installation, first workflows, CLI, concepts, and contributor guidance |
| `pybaldes/` | Python-only handbook: installation through public API reference |
| `solver/` | Solver lifecycle, algorithms, guarantees, and integer search |
| `reference/` | Cross-interface options, resources, algorithms, and build behavior |
| `extensions/` | Specialized formulations and shared-master model contracts |
| `benchmarks/` | Measured results with reproduction conditions |

Keep introductory workflows runnable and short. Put field/default tables in the
relevant reference, mathematical contracts in solver/model guides, and Python
usage in `pybaldes/`. Cross-link shared guarantees rather than changing them
independently. `guides/python.md` remains an entry page for existing links.

## Build and validate

Documentation builds require Python 3.10+ and the pinned documentation packages,
not a solver compilation:

```bash
python -m pip install -r .github/scripts/docs-requirements.txt
python -m unittest discover -s .github/scripts -p 'test_compile_docs.py'
python .github/scripts/compile_docs.py --output /tmp/baldes-docs-preview
```

The compiler reads all Markdown under `docs/`, maps `README.md` to `index.html`,
checks local targets and heading fragments, rewrites source links to repository
URLs, copies referenced images, and produces a search index. It renders before
replacing an existing generated output. It refuses to replace non-generated
nonempty directories. Generated HTML is not the documentation source.

To preview locally, serve the parent directory of the generated docs output and
open the relevant `index.html`. The normal website build uses `landing/docs/`;
its deployment workflow is [.github/workflows/landing.yaml](../../.github/workflows/landing.yaml).
A website build includes the dedicated PyBaldes section at `docs/pybaldes/index.html`.

## Check claims against the interface

For Python, inspect [exports](../../python/pybaldes/__init__.py),
[routing wrappers](../../python/pybaldes/routing.py),
[network wrappers](../../python/pybaldes/networks.py), and native bindings under
[python/rust/src](../../python/rust/src). Public Python features come from
[python/rust/Cargo.toml](../../python/rust/Cargo.toml), not every root-crate feature.
For CLI claims, inspect the command's parser/help and execute `--help` from the
matching build. Use implementation and regression tests to verify behavior;
a docstring or type stub alone may omit a feature gate or describe old behavior.

When an API changes, update examples, defaults, return fields, and build gates
together. In particular, refresh [PyBaldes API signatures](../pybaldes/api.md)
after changing public wrappers or bindings. Validate all runnable examples using
the same source/native-extension build. State any unavailable optional backend
instead of claiming those examples ran.

## Review checklist

- Specify the interface and solve mode for each claim of optimality.
- Separate numerical bounds, independent certificates, incumbents, and phase-I objectives.
- State distance conventions, units, ID spaces, and per-arc vector ordering.
- Show feature requirements next to optional APIs.
- Avoid claiming complete enumeration from a truncated pool.
- Preserve existing public documentation paths or leave a linking entry page.
- Link every guide from a discoverable index and verify heading fragments.
- Record machine, limits, features, repetitions, and rounding for performance claims.

Do not rewrite benchmark measurements without rerunning their documented setup.
The root chart requires explicit `--rounding none`. Historical measurements and
internal audit artifacts do not establish current performance or full solver
correctness.
