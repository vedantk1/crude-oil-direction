"""Release packaging uses disposable Git repositories and invented report data."""

from contextlib import chdir
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from oil_audit.public_reports import sanitize_report
from scripts.export_public_release import (ARCHIVE_NAME, ARCHIVE_PREFIX, MANIFEST_NAME,
                                          PUBLIC_REPORTS, _source_path,
                                          export_public_release)


def git(*arguments):
    return subprocess.run(["git", *arguments], check=True, capture_output=True).stdout


def commit():
    git("add", "--all")
    git("-c", "user.name=HISTORY_AUTHOR_SENTINEL", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "HISTORY_MESSAGE_SENTINEL")


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".test-public-release-", dir=".")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.cwd = chdir(self.workspace)
        self.cwd.__enter__()
        self.addCleanup(self.cwd.__exit__, None, None, None)
        git("init", "--quiet")
        Path("history-only.txt").write_bytes(b"PRIVATE_HISTORY_CONTENT_SENTINEL\n")
        commit()
        Path("history-only.txt").unlink()
        Path(".gitignore").write_text("data/\n.audit-deps/\n.venv/\n", encoding="utf-8")
        Path("README.md").write_bytes(b"Synthetic source export fixture\n")
        Path("example.py").write_bytes(b"print('invented example')\n")
        Path("reports").mkdir()
        for path, name in PUBLIC_REPORTS.items():
            Path(path).write_bytes(sanitize_report(name, b"{}\n"))
        commit()
        for directory in ("data/private-originals", ".audit-deps", ".venv"):
            root = Path(directory)
            root.mkdir(parents=True)
            (root / "ignored-backup.txt").write_bytes(b"IGNORED_BACKUP_SENTINEL\n")

    def test_export_contains_tracked_head_only_and_verifiable_source_manifest(self):
        original_history = git("log", "--format=%H%n%an%n%ae%n%B")
        original_head = git("rev-parse", "HEAD").decode().strip()
        original_status = git("status", "--porcelain=v1")
        result = export_public_release(Path("data/releases/first"))
        directory = Path(result["output_directory"])
        archive_payload = (directory / ARCHIVE_NAME).read_bytes()
        manifest_payload = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        expected = set(git("ls-files", "-z").decode().rstrip("\0").split("\0"))
        self.assertEqual(set(manifest["files"]), expected)
        self.assertEqual(manifest["source_commit"], original_head)
        self.assertEqual(result["source_commit"], original_head)
        self.assertEqual(result["file_count"], len(expected))
        self.assertEqual(result["archive_sha256"], hashlib.sha256(archive_payload).hexdigest())
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_payload).hexdigest())
        self.assertEqual(result["archive_bytes"], len(archive_payload))
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            self.assertEqual(set(archive.namelist()), {
                ARCHIVE_PREFIX + "/" + path for path in expected | {MANIFEST_NAME}})
            self.assertEqual(archive.read(ARCHIVE_PREFIX + "/" + MANIFEST_NAME), manifest_payload)
            for path, identity in manifest["files"].items():
                payload = archive.read(ARCHIVE_PREFIX + "/" + path)
                self.assertEqual(identity, {"bytes": len(payload),
                                            "sha256": hashlib.sha256(payload).hexdigest()})
                self.assertEqual(payload, git("show", "HEAD:" + path))
            for entry in archive.infolist():
                self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(entry.external_attr >> 16, 0o100644)
                self.assertEqual(entry.extra, b"")
                self.assertEqual(entry.comment, b"")
        for sentinel in (b"PRIVATE_HISTORY_CONTENT_SENTINEL", b"IGNORED_BACKUP_SENTINEL",
                         b"HISTORY_AUTHOR_SENTINEL", b"fixture@example.invalid", b"HISTORY_MESSAGE_SENTINEL"):
            self.assertNotIn(sentinel, archive_payload)
            self.assertNotIn(sentinel, manifest_payload)
        self.assertEqual(git("log", "--format=%H%n%an%n%ae%n%B"), original_history)
        self.assertEqual(git("status", "--porcelain=v1"), original_status)

    def test_repeat_export_is_byte_identical_and_existing_outputs_are_preserved(self):
        first = export_public_release(Path("data/releases/first"))
        second = export_public_release(Path("data/releases/second"))
        for name in (ARCHIVE_NAME, "manifest.json"):
            self.assertEqual((Path(first["output_directory"]) / name).read_bytes(),
                             (Path(second["output_directory"]) / name).read_bytes())
        preserved = {path: path.read_bytes() for path in Path(first["output_directory"]).iterdir()}
        with self.assertRaisesRegex(ValueError, "Existing release outputs are preserved"):
            export_public_release(Path(first["output_directory"]))
        self.assertEqual(preserved, {path: path.read_bytes() for path in preserved})

    def test_dirty_worktree_index_or_untracked_files_fail_before_output_creation(self):
        for kind in ("worktree", "index", "untracked"):
            with self.subTest(kind=kind):
                if kind == "untracked":
                    Path("untracked.txt").write_bytes(b"uncommitted fixture\n")
                else:
                    Path("README.md").write_bytes(b"uncommitted fixture\n")
                    if kind == "index":
                        git("add", "README.md")
                with self.assertRaisesRegex(ValueError, "clean working tree and index"):
                    export_public_release(Path("data/releases/rejected"))
                self.assertFalse(Path("data/releases").exists())
                if kind == "untracked":
                    Path("untracked.txt").unlink()
                else:
                    git("restore", "--staged", "--worktree", "README.md")

    def test_unsanitized_audit_is_rejected_without_changing_source_or_writing_output(self):
        original = b'{"series": {"synthetic": {"minimum": {"value": 123}}}}\n'
        path = Path("reports/price_audit.json")
        path.write_bytes(original)
        commit()
        with self.assertRaisesRegex(ValueError, "already carry valid public_sanitization"):
            export_public_release(Path("data/releases/rejected"))
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(Path("data/releases").exists())

    def test_wrong_report_marker_is_rejected_before_output_creation(self):
        path = Path("reports/price_audit.json")
        path.write_bytes(sanitize_report("vintage_audit.json", b"{}\n"))
        commit()
        with self.assertRaisesRegex(ValueError, "Unrecognized public sanitization metadata"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_missing_public_audit_is_rejected_before_output_creation(self):
        Path("reports/price_audit.json").unlink()
        commit()
        with self.assertRaisesRegex(ValueError, "all three sanitized price audit reports"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_tracked_private_data_is_rejected_even_if_force_added_to_ignored_directory(self):
        Path("data/raw-prices.txt").write_bytes(b"PRIVATE_TRACKED_DATA_SENTINEL\n")
        git("add", "--force", "data/raw-prices.txt")
        commit()
        with self.assertRaisesRegex(ValueError, "Tracked private or data paths"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_tracked_symlink_is_rejected_before_reading_its_target(self):
        Path("linked-original").symlink_to("data/private-originals/ignored-backup.txt")
        commit()
        with self.assertRaisesRegex(ValueError, "regular tracked files only"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_unsafe_tracked_filename_is_rejected_without_creating_an_archive(self):
        Path("unsafe\nname.txt").write_bytes(b"invented unsafe path fixture\n")
        commit()
        with self.assertRaisesRegex(ValueError, "Unsafe tracked filename"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_unsafe_source_filenames_and_reserved_manifest_fail_closed(self):
        for name in ("../outside", "/outside", "a//b", "a/./b", "a/../b", "a\\b", "a:b",
                     "a\nb", "a\x00b", "a\x7fb", "a\u202eb", "CON.txt", "docs/LPT1", "a.", "a ",
                     "a*b", "a?b", "a|b", "a<b", MANIFEST_NAME, MANIFEST_NAME.lower()):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _source_path(name)
        for name in (".git/config", "data/raw.csv", ".audit-deps/runtime", "private/original.json",
                     "backups/original.json", ".env", "docs/.env.local", "__pycache__/source.pyc"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Tracked private or data paths"):
                _source_path(name)

    def test_case_variant_file_directory_collision_fails_before_blob_reads(self):
        from scripts import export_public_release as module
        original_git = module._git

        def source(*arguments, **options):
            result = original_git(*arguments, **options)
            if arguments[0] == "ls-tree":
                result += b"100644 blob " + b"a" * 40 + b"\treadme.md/nested.txt\0"
            return result

        with patch.object(module, "_git", side_effect=source), \
             patch.object(module, "_blob_payloads", side_effect=AssertionError("Blob reads reached")):
            with self.assertRaisesRegex(ValueError, "file and directory names collide"):
                export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/releases").exists())

    def test_output_must_be_new_workspace_relative_release_directory_without_symlinks(self):
        for path in (Path("../outside"), Path("reports/export"), Path("data/releases"), Path.cwd() / "data/releases/x"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "workspace-relative directory"):
                export_public_release(path)
        Path("data/releases").symlink_to("private-originals", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not traverse symlinks"):
            export_public_release(Path("data/releases/rejected"))
        self.assertFalse(Path("data/private-originals/rejected").exists())

    def test_cli_runs_with_relative_output_and_does_not_export_git_metadata(self):
        script = Path(__file__).resolve().parents[1] / "scripts/export_public_release.py"
        completed = subprocess.run([sys.executable, str(script), "--output-root", "data/releases/cli"],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["output_directory"], "data/releases/cli")
        archive = Path(result["archive"]).read_bytes()
        self.assertNotIn(b"HISTORY_AUTHOR_SENTINEL", archive)
        self.assertFalse(Path("data/releases/cli/.git").exists())


if __name__ == "__main__":
    unittest.main()
