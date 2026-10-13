import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from oil_eval.runner import run


class RunnerTests(unittest.TestCase):
    def test_identical_run_reuses_artifacts_and_rejects_modified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"dataset_id": "synthetic", "manifest_sha256": "a" * 64}
            summary = {"holdout_evaluated": False, "prediction_count": 1, "scored_count": 0,
                       "unscored_count": 1, "metrics": {}}
            with patch("oil_eval.runner.load_development_data", return_value=(source, [], [], [])), \
                 patch("oil_eval.runner.replay_development", return_value=([{"forecast_date": "2017-01-05"}], summary)):
                first = run(root, root / "runs", root / "report.json")
                second = run(root, root / "runs", root / "report.json")
                self.assertEqual(first, second)
                destination = Path(first["evaluation_directory"])
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["numeric_threads"], 1)
                self.assertEqual(manifest["protocol"]["holdout_start"], "2023-01-01")
                (destination / "predictions.jsonl").write_text("modified\n")
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    run(root, root / "runs", root / "report.json")

    def test_cli_has_no_holdout_execution_option(self):
        result = subprocess.run([
            sys.executable, "-m", "oil_eval", "--dataset-dir", "data/unused", "--holdout",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --holdout", result.stderr)


if __name__ == "__main__":
    unittest.main()
