import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("robust_benchmark", Path(__file__).resolve().parents[1] / "benchmarks/benchmark_robust.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class RobustBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.instance = dict(vehicles=2, capacity=4, rows=[
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 2, 0, 100, 0],
            [2, 2, 0, 2, 0, 100, 0],
        ])
        self.case = dict(gq=1, gt=1, aq="0.5", at="0.5")
        self.report = dict(status="Optimal", incumbent=6., bound=6., routes=[
            dict(nodes=[0, 1, 3], arcs=[0, 1], cost=2.),
            dict(nodes=[0, 2, 3], arcs=[2, 3], cost=4.),
        ])

    def test_paper_grid_has_1320_unique_cases(self):
        names = [p.stem for p in (benchmark.ROOT / "instances/Solomon").glob("*.txt") if p.stem.startswith(("C1", "R1", "RC1")) and p.stem not in benchmark.EXCLUDED]
        cases = benchmark.cases(names)
        self.assertEqual(len(cases), 1320)
        self.assertEqual(len({c['key'] for c in cases}), 1320)

    def test_robust_incumbent_replays_in_integer_units(self):
        self.assertEqual(benchmark.validate(self.report, self.case, self.instance), [])

    def test_nominal_feasible_route_can_fail_robust_capacity(self):
        self.report.update(incumbent=4., bound=4., routes=[dict(nodes=[0, 1, 2, 3], arcs=[0, 1, 2], cost=4.)])
        self.assertIn("robust capacity violation", benchmark.validate(self.report, self.case, self.instance))
        self.case['gq'] = 0
        self.assertEqual(benchmark.validate(self.report, self.case, self.instance), [])

    def test_nominal_feasible_route_can_fail_robust_time(self):
        self.instance['rows'][1][5] = 1
        self.assertIn("robust time-window violation", benchmark.validate(self.report, self.case, self.instance))
        self.case['gt'] = 0
        self.assertEqual(benchmark.validate(self.report, self.case, self.instance), [])

    def test_repeated_and_missing_customers_are_rejected(self):
        self.report['routes'][1] = copy.deepcopy(self.report['routes'][0])
        self.report['incumbent'] = self.report['bound'] = 4.
        self.assertIn("customer cover is not a partition", benchmark.validate(self.report, self.case, self.instance))

    def test_open_gap_does_not_count_as_optimal(self):
        self.report['bound'] = 5.9
        self.assertIn("Optimal with open gap", benchmark.validate(self.report, self.case, self.instance))

    def test_timeout_without_incumbent_is_unresolved_not_an_invalid_route(self):
        self.report.update(status="TimeLimit", incumbent=None, bound=0., routes=[])
        self.assertEqual(benchmark.validate(self.report, self.case, self.instance), [])
        self.report['status'] = 'Optimal'
        self.assertIn("Optimal without incumbent", benchmark.validate(self.report, self.case, self.instance))


if __name__ == '__main__':
    unittest.main()
