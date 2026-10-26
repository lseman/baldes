#!/usr/bin/env python3
"""Verify, or download and verify, the pinned robust-PDPTW benchmark files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download missing files")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / "instances" / "RobustPDPTW"
    manifest = json.loads((project / "scripts/data/robust_pd_manifest.json").read_text(encoding="utf-8"))
    if args.download:
        root.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.exists() and args.download:
            url = f"https://raw.githubusercontent.com/abreualexp/robust-pdptw/{manifest['commit']}/Instances/{name}"
            with urlopen(url, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f"Downloaded checksum mismatch: {name}")
            path.write_bytes(data)
        if not path.exists():
            raise SystemExit(f"Missing {name}; run with --download")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Checksum mismatch: {name}")
    print(f"Verified {len(manifest['sha256'])} files at upstream commit {manifest['commit']}")


if __name__ == "__main__":
    main()
