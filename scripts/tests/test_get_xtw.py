"""Offline checks for XTW catalog association and preserving local downloads."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SETUP = Path(__file__).resolve().parents[1] / "setup"
sys.path.insert(0, str(SETUP))
try:
    spec = importlib.util.spec_from_file_location("get_xtw", SETUP / "get_xtw.py")
    get_xtw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(get_xtw)
finally:
    sys.path.pop(0)

class XtwDownloads(unittest.TestCase):
    def test_catalog_keeps_instance_and_solution_in_the_same_xtw_row(self):
        parser=get_xtw.Catalog()
        parser.feed('<table><tr><td><a href="/cvrplib/en/download/instance/767">XTW-n247</a></td>'
                    '<td><a href="/cvrplib/en/download/bks/767">12 / 16,401</a></td></tr>'
                    '<tr><td><a href="/cvrplib/en/download/instance/1">X-n101-k25</a></td>'
                    '<td><a href="/cvrplib/en/download/bks/1">100</a></td></tr></table>')
        self.assertEqual(parser.entries,[{'name':'XTW-n247','instance':get_xtw.BASE_URL+'/cvrplib/en/download/instance/767','solution':get_xtw.BASE_URL+'/cvrplib/en/download/bks/767'}])

    def test_existing_files_are_verified_and_never_replaced(self):
        entry={'name':'XTW-n247','instance':'instance-url','solution':'solution-url'}
        instance=b'NAME : XTW-n247\nTIME_WINDOW_SECTION\n1 0 10\n'
        solution=b'Route #1: 1\nCost 2\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(get_xtw,'fetch',side_effect=[instance,solution]):
                record=get_xtw.download(entry,root,30)
            self.assertEqual(len(record['instance_sha256']),64)
            with patch.object(get_xtw,'fetch',side_effect=[instance,solution]):
                get_xtw.download(entry,root,30)
            with patch.object(get_xtw,'fetch',return_value=instance+b'EOF\n'):
                with self.assertRaisesRegex(ValueError,'differs'):
                    get_xtw.download(entry,root,30)
            self.assertEqual((root/'XTW-n247.vrp').read_bytes(),instance)

    def test_error_pages_are_not_saved_as_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(get_xtw,'fetch',return_value=b'<html>Not found</html>'):
                with self.assertRaisesRegex(ValueError,'Invalid instance'):
                    get_xtw.download({'name':'XTW-n247','instance':'bad'},Path(directory),30)
            self.assertEqual(list(Path(directory).iterdir()),[])

if __name__=='__main__':unittest.main()
