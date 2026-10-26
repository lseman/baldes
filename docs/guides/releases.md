# Python wheels and releases

The `Python distributions` workflow builds `pybaldes` wheels for Linux x86-64
and ARM64 (manylinux2014), and macOS Intel and Apple Silicon. Wheels use the
CPython stable ABI, starting at Python 3.9. Each wheel is installed and tested
outside the checkout on Python 3.9 and 3.14 before it can be released.

The standard wheel includes bundled Clp, multi-time-window resources and the
default master branching policies. It uses portable CPU settings. HiGHS,
Gurobi and CUDA require separate source builds; Windows wheels are not currently
built because the bundled native backend configuration targets Unix systems.

## Create a release

1. Set the same version in the root `Cargo.toml` and
   `python/rust/Cargo.toml`, then refresh `Cargo.lock` with `cargo check -p pybaldes`.
   Python package metadata derives its version from the bindings crate.
2. Commit the release changes and push the commit.
3. Create and push a matching version tag, for example:

   ```sh
   python .github/scripts/check_release.py --tag v2.0.2
   git tag -a v2.0.2 -m 'BALDES 2.0.2'
   git push origin v2.0.2
   ```

A pushed `v*` tag triggers validation, builds all four wheels and a source
archive, publishes them to PyPI, and creates a GitHub release with generated notes and
`SHA256SUMS` after PyPI publication succeeds.
A failed build or installed-wheel test prevents release publication. The source
archive includes the vendored Clp/CoinUtils files, and wheel builds start from
that archive to check source completeness.

Tags must match the Cargo version exactly. Prerelease versions use Cargo syntax,
for example `2.1.0-rc.1` and tag `v2.1.0-rc.1`; maturin normalizes Python package
metadata. Prereleases are marked as such and do not replace the latest release.
Existing releases are not overwritten automatically.

Pull requests affecting packaging or solver sources also run the distribution
builds. A manual workflow run builds and tests artifacts without publishing.
The isolated PyPI job uses the repository secret `PYPI_API_TOKEN` with token
authentication. It receives tested artifacts and does not execute repository
build scripts. Existing PyPI files are skipped to allow retries after a partial
upload. Token authentication does not enable trusted-publishing attestations.
Only the GitHub release job has repository write permission; it uses the standard
GitHub token.

## Install or build

Install from PyPI in a virtual environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install pybaldes
```

For a local source build, initialize native dependencies first:

```sh
git submodule update --init --recursive
python -m pip install 'maturin[patchelf]>=1.9.4,<2'
maturin build --release --locked --sdist --out dist
```

Local Linux builds use the host's system ABI. Official Linux wheels use the
manylinux build container and its portability checks. See
[Python bindings](python.md) for editable builds and optional backends.
