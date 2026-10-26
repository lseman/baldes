# pyBaldes

pyBaldes is the Python package for BALDES routing root relaxations, RCSP pricing,
and shared-master network models with integer search.

`pybaldes.solve` accepts an instance/path, a scalar `PricingNetwork`, an array,
or a complete `NetworkProblem`. Scalar and one-element-array inputs normalize
to the same model. Data lives in `routing.py` and `networks.py`; `solve.py` owns
the public solve entry point.

Start with the [pyBaldes handbook](../docs/pybaldes/README.md).

- [Install and select build features](../docs/pybaldes/installation.md)
- [Run the quickstart](../docs/pybaldes/quickstart.md)
- [Model CVRP and VRPTW routing roots](../docs/pybaldes/routing.md)
- [Use the pricing solver](../docs/pybaldes/pricing.md)
- [Model shared-master networks](../docs/pybaldes/networks.md)
- [Look up the public API](../docs/pybaldes/api.md)

From the repository root, with the native submodules initialized:

```bash
python -m pip install .
```

The distribution and import name are both `pybaldes`. Source code is in
`pybaldes/`, PyO3 bindings are in `rust/`, and regression tests are in `tests/`.
The documentation distinguishes LP relaxations, integer witnesses, pricing
exactness, and independent certificates; check those contracts before treating
a reported objective as an optimal integer routing solution.

## Distribution releases

Version tags build and test portable wheels and attach them to a GitHub release.
See the [release guide](../docs/guides/releases.md) for supported platforms,
versioning, included backends and installation.
