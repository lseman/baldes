"""Run with python3 -m unittest discover -s scripts/tests -p 'test_gurobi_downloader.py'."""

import hashlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

loader = importlib.machinery.SourceFileLoader(
    "gurobi_downloader", str(Path(__file__).resolve().parents[1] / "setup" / "gurobi_downloader")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
downloader = importlib.util.module_from_spec(spec)
loader.exec_module(downloader)


class DownloaderTests(unittest.TestCase):
    def test_latest_release_is_numeric_and_excludes_prereleases(self):
        index = ''.join(f'<a href="/{v}/">release</a>' for v in
                        ["13.0.9", "13.0.10", "14.0.0beta", "14.0.0_gpu_alpha", "12.0.4"])
        self.assertEqual(downloader.latest_version(index), "13.0.10")
        with self.assertRaises(ValueError):
            downloader.latest_version('<a href="/14.0.0beta/">beta</a>')

    def test_archive_is_verified_and_cached(self):
        payload = b"downloaded archive"
        expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive.tar.gz"
            with patch.object(downloader, "open_url", return_value=io.BytesIO(payload)) as fetch:
                downloader.download("https://example.test/archive", archive, expected)
                downloader.download("https://example.test/archive", archive, expected)
                self.assertEqual(fetch.call_count, 1)
            with patch.object(downloader, "open_url", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    downloader.download("https://example.test/archive", archive, "0" * 32)
            self.assertEqual(archive.read_bytes(), payload)
            self.assertEqual(list(Path(directory).iterdir()), [archive])

    def test_checksum_must_name_the_requested_archive(self):
        checksum = "a" * 32
        self.assertEqual(downloader.checksum(checksum + "  package.tar.gz\n", "package.tar.gz"), checksum)
        with self.assertRaises(ValueError):
            downloader.checksum(checksum + "  other.tar.gz", "package.tar.gz")

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "requires safe tarfile extraction")
    def test_extract_publishes_complete_installation_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            archive = destination / "archive.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                for name in ["include/gurobi_c.h", "bin/gurobi_cl"]:
                    member = tarfile.TarInfo("gurobi1303/linux64/" + name)
                    output.addfile(member, io.BytesIO())
            installation = downloader.extract(archive, destination, "13.0.3", "linux64")
            self.assertTrue((installation / "include/gurobi_c.h").is_file())
            self.assertEqual(downloader.extract(archive, destination, "13.0.3", "linux64"), installation)

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "requires safe tarfile extraction")
    def test_unsafe_archive_never_publishes_an_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            archive = destination / "archive.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("gurobi1303/../../escaped")
                output.addfile(member, io.BytesIO())
            with self.assertRaises((tarfile.TarError, ValueError)):
                downloader.extract(archive, destination, "13.0.3", "linux64")
            self.assertEqual(list(destination.iterdir()), [archive])


if __name__ == "__main__":
    unittest.main()
