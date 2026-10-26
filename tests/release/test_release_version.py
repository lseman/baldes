import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_release", ROOT / ".github/scripts/check_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseVersionTests(unittest.TestCase):
    def test_current_version_and_matching_tag(self):
        version = release.check_release(ROOT)
        self.assertEqual(release.check_release(ROOT, f"v{version}"), version)

    def test_mismatched_tag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            release.check_release(ROOT, "v99.0.0")

    def test_mismatched_package_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python/rust").mkdir(parents=True)
            (root / "Cargo.toml").write_text('[package]\nversion = "1.0.0"\n')
            (root / "python/rust/Cargo.toml").write_text(
                '[package]\nversion = "1.0.1"\n'
            )
            with self.assertRaisesRegex(ValueError, "versions differ"):
                release.check_release(root)
