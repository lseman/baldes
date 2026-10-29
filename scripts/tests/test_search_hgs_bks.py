"""Integration checks for the batch runner using a controlled solver."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/benchmarks/search_hgs_bks.py"


class SearchRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "solver"
        self.binary.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv
assert '--get-bks' in args and '--intensify' in args
seed = int(args[args.index('--seed') + 1])
if seed == 9:
    sys.exit(7)
pathlib.Path(args[args.index('--solution-output') + 1]).write_text('Route #1: 1\\nCost 99\\n')
pathlib.Path(args[args.index('--json') + 1]).write_text(json.dumps({'cost': 99, 'seconds': 1, 'generations': 10}))
print('Intensify: initial=100.000000 best=99.000000')
""")
        self.binary.chmod(0o755)
        self.instance = self.root / "X-n1-k1.vrp"
        self.instance.write_text("example\n")
        self.output = self.root / "results"

    def run_search(self, *extra):
        return subprocess.run([sys.executable, str(RUNNER), str(self.instance), "--binary", str(self.binary),
                               "--output", str(self.output), *extra], capture_output=True, text=True)

    def test_separate_seed_outputs_and_comparison(self):
        process = self.run_search("--seeds", "1", "2", "--std-similarity")
        self.assertEqual(process.returncode, 0, process.stderr)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(summary["runs"]), 2)
        for seed, row in enumerate(summary["runs"], 1):
            self.assertTrue(row["candidate_improvement"])
            self.assertEqual(row["improvement"], 1)
            self.assertIn("--std-similarity", row["command"])
            self.assertTrue((self.output / f"X-n1-k1.seed-{seed}.sol").exists())
        self.assertNotEqual(summary["runs"][0]["solution"], summary["runs"][1]["solution"])
        self.assertNotEqual(self.run_search().returncode, 0)

    def test_failure_is_reported_and_later_runs_continue(self):
        process = self.run_search("--seeds", "9", "2")
        self.assertNotEqual(process.returncode, 0)
        rows = json.loads((self.output / "summary.json").read_text())["runs"]
        self.assertEqual([row["status"] for row in rows], ["error", "ok"])
        self.assertEqual(rows[0]["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
