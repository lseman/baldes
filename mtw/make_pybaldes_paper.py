"""Sync benchmark records and regenerate the actual paper in mtw/satVRP.

Run from the BALDES repository root: python mtw/make_pybaldes_paper.py
The paper repository also supports: python scripts/make_pybaldes_table.py
"""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    paper = here / 'satVRP'
    data = paper / 'results/compiled-pricing'
    data.mkdir(parents=True, exist_ok=True)
    for backend, directory in [('python', 'python-labeling-matched-20261025'),
                               ('baldes', 'pybaldes-pricing-reuse-20261025')]:
        shutil.copy2(here / 'results' / directory / 'runs.jsonl', data / f'{backend}-runs.jsonl')
    subprocess.run([sys.executable, str(paper / 'scripts/make_pybaldes_table.py')], check=True)


if __name__ == '__main__':
    main()
