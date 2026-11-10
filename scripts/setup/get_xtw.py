#!/usr/bin/env python3
"""Download original XTW instances and published solutions from CVRPLib.

Existing files are verified and preserved. A manifest records source URLs and
SHA-256 hashes. Uses only the Python standard library.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin

from get_bks import BASE_URL, fetch

CATALOG = BASE_URL + '/cvrplib/en/vrptw/instances'

class Catalog(HTMLParser):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.row = None
        self.link = None
        self.parts = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'tr':
            self.row = {}
        if tag == 'a' and self.row is not None:
            self.link = attrs.get('href', '')
            self.parts = []
    def handle_data(self, data):
        if self.link is not None:
            self.parts.append(data)
    def handle_endtag(self, tag):
        if tag == 'a' and self.link is not None:
            name = ''.join(self.parts).strip()
            if re.search(r'/download/instance/\d+$', self.link) and re.fullmatch(r'XTW-n\d+', name):
                self.row.update(name=name, instance=urljoin(BASE_URL, self.link))
            if re.search(r'/download/bks/\d+$', self.link):
                self.row['solution'] = urljoin(BASE_URL, self.link)
            self.link = None
        if tag == 'tr' and self.row is not None:
            if 'name' in self.row:
                self.entries.append(self.row)
            self.row = None

def download(entry, output, timeout):
    result = dict(entry)
    for kind, suffix in [('instance', '.vrp'), ('solution', '.sol')]:
        if kind not in entry:
            raise ValueError(f"{entry['name']} has no published {kind}")
        path = output / (entry['name'] + suffix)
        data = fetch(entry[kind], timeout)
        text = data.decode('utf-8-sig')
        if kind == 'instance':
            if not re.search(r'^NAME\s*:\s*'+re.escape(entry['name'])+r'\s*$', text, re.M) or 'TIME_WINDOW_SECTION' not in text:
                raise ValueError(f"Invalid instance response for {entry['name']}")
        elif not re.search(r'^Route\s+#?\d+\s*:', text, re.M):
            raise ValueError(f"Invalid solution response for {entry['name']}")
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Existing file differs from published data: {path}; preserve or move it before retrying")
        else:
            with path.open('xb') as stream:
                stream.write(data)
        result[kind+'_sha256'] = hashlib.sha256(data).hexdigest()
        result[kind+'_file'] = path.name
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('instances/XTW'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error('workers and timeout must be positive')
    catalog = Catalog()
    catalog.feed(fetch(CATALOG, args.timeout).decode('utf-8'))
    entries = sorted(catalog.entries, key=lambda e: int(e['name'].split('-n')[1]))
    if len(entries) != 100 or len({e['name'] for e in entries}) != 100:
        raise ValueError(f'Expected 100 distinct XTW entries, found {len(entries)}')
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda e: download(e, args.output, args.timeout), entries))
    manifest = {'catalog': CATALOG, 'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'count': len(records), 'instances': records}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Verified {len(records)} original XTW instances and solutions in {args.output}")

if __name__ == '__main__':
    main()
