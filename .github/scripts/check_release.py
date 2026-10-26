"""Validate a release tag against both Cargo package versions."""
import argparse
from pathlib import Path
import re
import tomllib


def check_release(root: Path, tag: str | None = None) -> str:
    versions = [
        tomllib.loads((root / name).read_text())["package"]["version"]
        for name in ("Cargo.toml", "python/rust/Cargo.toml")
    ]
    if versions[0] != versions[1]:
        raise ValueError(f"Rust/Python package versions differ: {versions}")
    version = versions[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?", version):
        raise ValueError(f"Unsupported release version: {version}")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must be v{version}")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if "version" not in project.get("dynamic", []):
        raise ValueError("Python version must be derived from its Cargo package")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    print(check_release(Path(__file__).resolve().parents[2], args.tag))
