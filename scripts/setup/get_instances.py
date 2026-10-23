#!/usr/bin/env python3
"""Download all CVRPLIB instance sets (7z format) and extract them to instances/."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://galgos.inf.puc-rio.br/cvrplib/en/instances"
INSTANCES_DIR = Path(__file__).resolve().parents[2] / "instances"


def fetch(url: str) -> str:
    """Fetch a URL and return its text content."""
    req = urllib.request.Request(url, headers={"User-Agent": "get_instances.py"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def extract_instance_sets(html: str) -> list[tuple[str, str]]:
    """Parse HTML and return list of (category/set_name, download_id) tuples."""
    pattern = re.compile(
        r'<a href="/cvrplib/en/instances/\d+" class="text-decoration-none" >(\w+)</a>'
        r'\s*Set <strong>(\w+)</strong>.*?'
        r'href="/cvrplib/en/download/instance-set/(\d+)"',
        re.DOTALL,
    )
    return pattern.findall(html)


def get_download_url(instance_id: str) -> str:
    """Return the full download URL for an instance set."""
    return f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/{instance_id}"


def get_remote_filename(download_url: str) -> str:
    """Get the remote filename from Content-Disposition header."""
    req = urllib.request.Request(
        download_url, headers={"User-Agent": "get_instances.py"}
    )
    with urllib.request.urlopen(req) as resp:
        # Follow redirects to get final Content-Disposition
        cd = resp.headers.get("Content-Disposition", "")
        match = re.search(r'filename=(.+)', cd)
        if match:
            return match.group(1).strip().strip('"')
    # Fallback
    return f"set-{download_url.split('/')[-1]}.7z"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "get_instances.py"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        received = 0
        chunk_size = 8192
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if total:
                    pct = received / total * 100
                    print(
                        f"\r  Downloading {dest.name} [{pct:.0f}%",
                        end="",
                        flush=True,
                    )
    print(f" done ({received:,} bytes)")


def extract_7z(archive: Path, target_dir: Path) -> None:
    """Extract a 7z archive using the 7z command-line tool."""
    if not shutil.which("7z"):
        print(
            f"\nERROR: '7z' command not found. "
            f"Install p7zip-full (Debian/Ubuntu) or p7zip (macOS).",
            file=sys.stderr,
        )
        sys.exit(1)
    subprocess.run(["7z", "x", str(archive), f"-o{target_dir}", "-y"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract all CVRPLIB instance sets."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without downloading.",
    )
    args = parser.parse_args()

    # Collect instance sets from both pages
    print("Fetching instance list ...")
    html1 = fetch(BASE_URL)
    html2 = fetch(f"{BASE_URL}/2")
    sets = extract_instance_sets(html1) + extract_instance_sets(html2)

    if not sets:
        print("ERROR: No instance sets found.", file=sys.stderr)
        sys.exit(1)

    # Ensure instances directory exists
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    for category, name, instance_id in sets:
        folder = Path(category) / name
        archive_name = get_remote_filename(get_download_url(instance_id))
        archive_path = INSTANCES_DIR / archive_name

        if args.dry_run:
            print(f"  {category}/{name} -> {archive_name} (not downloading)")
            continue

        download_file(get_download_url(instance_id), archive_path)
        print(f"  Extracting {category}/{name} ...", end=" ", flush=True)
        extract_7z(archive_path, INSTANCES_DIR)
        print("done")

    # Clean up archives
    print("\nCleaning up archives ...")
    for f in INSTANCES_DIR.glob("*.7z"):
        f.unlink()

    print(f"\nAll instance sets extracted to {INSTANCES_DIR}")


if __name__ == "__main__":
    main()
