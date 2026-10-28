#!/usr/bin/env python3
"""Download a named CVRPLib best-known solution using only the standard library."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://galgos.inf.puc-rio.br"
CATALOGS = {
    "cvrp": "/cvrplib/en/instances",
    "vrptw": "/cvrplib/en/instances/2",
}


class CatalogParser(HTMLParser):
    """Associate instance names with BKS links in the same table row."""

    def __init__(self):
        super().__init__()
        self.entries = []
        self.in_row = False
        self.in_name = False
        self.name_parts = []
        self.solution = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.name_parts = []
            self.solution = None
        if tag == "a" and self.in_row:
            href = attrs.get("href", "")
            self.in_name = bool(re.search(r"/download/instance/\d+$", href))
            if re.search(r"/download/bks/\d+$", href):
                self.solution = urljoin(BASE_URL, href)

    def handle_data(self, data):
        if self.in_name:
            self.name_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a":
            self.in_name = False
        if tag == "tr":
            name = "".join(self.name_parts).strip()
            if name:
                self.entries.append((name, self.solution))
            self.in_row = False
            self.in_name = False


def fetch(url, timeout):
    request = Request(url, headers={"User-Agent": "BALDES-get-bks/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def instance_name(value):
    name = Path(value).name
    if Path(name).suffix.lower() in {".vrp", ".txt", ".sol", ".dat"}:
        name = Path(name).stem
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Expected an instance name or path, such as C101.txt or X-n101-k25.vrp")
    return name


def find_solution(name, problem, timeout):
    for kind in CATALOGS if problem == "auto" else [problem]:
        parser = CatalogParser()
        parser.feed(fetch(urljoin(BASE_URL, CATALOGS[kind]), timeout).decode("utf-8"))
        matches = [(canonical, url) for canonical, url in parser.entries
                   if canonical.casefold() == name.casefold()]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous catalog entries for {name}")
        if matches:
            canonical, url = matches[0]
            if url is None:
                raise ValueError(f"{canonical} is listed but has no public BKS download")
            return canonical, url
    raise ValueError(f"Instance {name!r} was not found in the selected CVRPLib catalogs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="Instance name or local path (file need not exist)")
    parser.add_argument("--problem", choices=["auto", "cvrp", "vrptw"], default="auto")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: <instance>.sol)")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per request (default: 30)")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        name = instance_name(args.instance)
        canonical, url = find_solution(name, args.problem, args.timeout)
        output = args.output or Path(f"{canonical}.sol")
        if output.exists() and not args.force:
            raise ValueError(f"Output already exists: {output}; use --force to replace it")
        data = fetch(url, args.timeout)
        # An error or login page can be returned with HTTP 200.
        text = data.decode("utf-8-sig")
        if not re.search(r"^\s*Route\s+#?\d+\s*:", text, re.MULTILINE | re.IGNORECASE):
            raise ValueError(f"{url} did not return a route solution; no file written")
        with output.open("wb" if args.force else "xb") as stream:
            stream.write(data)
        print(f"Saved {canonical} BKS to {output}\nSource: {url}")
        return 0
    except (OSError, URLError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
