"""Checks for the synthetic benchmark's scientific and reproducibility contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "generate_stochastic_instances.py"
spec = importlib.util.spec_from_file_location("generate_stochastic_instances", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class StochasticGenerationTests(unittest.TestCase):
    def test_grid_and_capacity_contracts(self):
        self.assertEqual(len(generator.GRID), 11)
        for n, m in generator.GRID:
            base = generator.base_instance(n, m, 1, 123)
            self.assertEqual(len(base["coordinates"]), n)
            self.assertEqual(len(base["demands"]), n - 1)
            self.assertTrue(all(0 <= c < 100 for xy in base["coordinates"] for c in xy))
            for demand in base["demands"]:
                self.assertIn(demand["support"], [list(range(1, 6)), list(range(6, 11)), list(range(11, 16))])
                self.assertEqual(demand["probabilities"], [0.1, 0.2, 0.4, 0.2, 0.1])
            for i in range(n):
                self.assertEqual(base["costs"][i][i], 0)
                for j in range(n):
                    self.assertEqual(base["costs"][i][j], base["costs"][j][i])
                    self.assertIsInstance(base["costs"][i][j], int)
            for fill in generator.FILL_RATES:
                instance = generator.with_fill_rate(base, fill)
                cap = instance["capacity"]
                self.assertGreaterEqual(cap, max(d["support"][-1] for d in base["demands"]))
                self.assertLessEqual(instance["actual_fill_rate"], float(fill) + 1e-14)
                self.assertLess(cap - instance["ideal_capacity"], 1.)
                self.assertEqual(instance["provenance"]["failure_penalty_applied"], 0.)
                self.assertFalse(instance["provenance"]["original_benchmark_instance"])

    def test_reproducibility_hashes_and_fill_rate_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = generator.generate(root / "first", 77, 2, [20])
            second = generator.generate(root / "second", 77, 2, [20])
            self.assertEqual(first, second)
            self.assertEqual(first["instance_count"], 8)
            for record in first["instances"]:
                data = (root / "first" / record["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(data, (root / "second" / record["file"]).read_bytes())
            paired = [json.loads((root / "first" / r["file"]).read_bytes()) for r in first["instances"][:4]]
            for instance in paired[1:]:
                for key in ["coordinates", "costs", "demands", "base_seed", "base_id"]:
                    self.assertEqual(instance[key], paired[0][key])
            # Adding a grid entry cannot perturb an existing instance.
            expanded = generator.generate(root / "expanded", 77, 2, [20, 30])
            for record in first["instances"]:
                self.assertEqual((root / "first" / record["file"]).read_bytes(),
                                 (root / "expanded" / record["file"]).read_bytes())
            self.assertEqual(expanded["instance_count"], 16)

    def test_conflicting_generation_cannot_overwrite_existing_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = generator.generate(root, 1, 1, [20])
            self.assertEqual(first, generator.generate(root, 1, 1, [20]))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(FileExistsError):
                generator.generate(root, 2, 1, [20])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_invalid_generation_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            for repetitions, sizes in [(0, [20]), (1, []), (1, [20, 20]), (1, [21])]:
                with self.assertRaises(ValueError):
                    generator.generate(Path(tmp), 1, repetitions, sizes)
            with self.assertRaises(ValueError):
                generator.with_fill_rate(generator.base_instance(20, 2, 1, 1), "0")


if __name__ == "__main__":
    unittest.main()
