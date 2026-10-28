"""Offline regression checks for catalog lookup and safe solution downloads."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "get_bks", Path(__file__).resolve().parents[1] / "setup/get_bks.py"
)
get_bks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_bks)

CATALOG = b'''<table><tr><td><a title="Instance File"
href="/cvrplib/en/download/instance/380"> C101 </a></td>
<td><a href="/cvrplib/en/download/bks/380"><span>$827.3$</span></a></td></tr>
<tr><td><a href="/cvrplib/en/download/instance/381">C102</a></td></tr></table>'''
SOLUTION = b"Route #1: 1 2 3\nCost 827.3\n"


class BksTests(unittest.TestCase):
    def test_path_and_case_insensitive_lookup(self):
        with patch.object(get_bks, "fetch", side_effect=[b"<table></table>", CATALOG]):
            name = get_bks.instance_name("instances/c101.txt")
            self.assertEqual(get_bks.find_solution(name, "auto", 30),
                             ("C101", get_bks.BASE_URL + "/cvrplib/en/download/bks/380"))

    def test_missing_instance_and_unpublished_solution(self):
        for name, message in [("unknown", "not found"), ("C102", "no public BKS")]:
            with self.subTest(name=name), patch.object(get_bks, "fetch", return_value=CATALOG):
                with self.assertRaisesRegex(ValueError, message):
                    get_bks.find_solution(name, "vrptw", 30)

    def test_download_preserves_existing_file_and_rejects_html(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bks.sol"
            args = ["C101", "--problem", "vrptw", "-o", str(output)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(get_bks, "fetch", side_effect=[CATALOG, SOLUTION]):
                    self.assertEqual(get_bks.main(args), 0)
                self.assertEqual(output.read_bytes(), SOLUTION)
                with patch.object(get_bks, "fetch", return_value=CATALOG):
                    self.assertEqual(get_bks.main(args), 1)
                with patch.object(get_bks, "fetch", side_effect=[CATALOG, b"<html>Error</html>"]):
                    self.assertEqual(get_bks.main(args + ["--force"]), 1)
                self.assertEqual(output.read_bytes(), SOLUTION)
                with patch.object(get_bks, "fetch", side_effect=[CATALOG, SOLUTION]):
                    self.assertEqual(get_bks.main(args + ["--force"]), 0)


if __name__ == "__main__":
    unittest.main()
