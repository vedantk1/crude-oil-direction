from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from oil_audit.report_output import preserve_report, write_audit_report


@contextmanager
def workspace():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            yield Path(".")
        finally:
            os.chdir(original)


def fixture(value=71):
    return json.dumps({"series": {"wti": {"minimum": {"date": "2019-01-02", "value": value}}}}).encode()


class ReportOutputTests(unittest.TestCase):
    def test_public_output_preserves_new_and_previous_raw_bytes_before_replacing(self):
        with workspace():
            target = Path("reports/price_audit.json")
            target.parent.mkdir()
            old, current = fixture(70), fixture()
            target.write_bytes(old)
            write_audit_report("price_audit.json", current, target)
            public = json.loads(target.read_bytes())
            self.assertIsNone(public["series"]["wti"]["minimum"]["value"])
            self.assertEqual(public["public_sanitization"]["original_sha256"], hashlib.sha256(current).hexdigest())
            for payload in (old, current):
                archived = Path("data/private-report-archive") / hashlib.sha256(payload).hexdigest() / "price_audit.json"
                self.assertEqual(archived.read_bytes(), payload)

    def test_private_output_retains_original_but_refuses_different_replacement(self):
        with workspace():
            target = Path("data/private/price.json")
            write_audit_report("price_audit.json", fixture(), target)
            self.assertEqual(target.read_bytes(), fixture())
            with self.assertRaisesRegex(ValueError, "existing originals"):
                write_audit_report("price_audit.json", fixture(99), target)
            self.assertEqual(target.read_bytes(), fixture())

    def test_invalid_policy_or_bytes_create_no_archive_or_output(self):
        with workspace():
            for name, payload in (("unknown.json", fixture()), ("price_audit.json", b'{"broken":')):
                with self.assertRaises(ValueError):
                    write_audit_report(name, payload, Path("reports/out.json"))
            self.assertFalse(Path("reports").exists())
            self.assertFalse(Path("data").exists())

    def test_output_paths_cannot_traverse_escape_or_place_raw_data_in_source_tree(self):
        with workspace():
            for path in (Path("../reports/out.json"), Path("/tmp/out.json"), Path("docs/out.json")):
                with self.assertRaises(ValueError):
                    write_audit_report("price_audit.json", fixture(), path)
            Path("reports").symlink_to(Path(tempfile.gettempdir()), target_is_directory=True)
            with self.assertRaises(ValueError):
                write_audit_report("price_audit.json", fixture(), Path("reports/out.json"))

    def test_private_archive_integrity_and_root_guard(self):
        with workspace():
            archived = preserve_report("price_audit.json", fixture())
            archived.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "refusing replacement"):
                preserve_report("price_audit.json", fixture())
            with self.assertRaisesRegex(ValueError, "ignored data"):
                preserve_report("price_audit.json", fixture(), Path("reports/archive"))

    def test_in_workspace_symlink_cannot_expose_private_originals(self):
        with workspace():
            Path("reports").mkdir()
            Path("data").symlink_to("reports", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                preserve_report("price_audit.json", fixture())
            self.assertEqual(list(Path("reports").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
