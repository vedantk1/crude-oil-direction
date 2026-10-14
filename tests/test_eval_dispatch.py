"""Dispatch failures must precede source reads, model work and output changes."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from oil_eval.protocol import (EXPERIMENTS, GENERIC_EXPERIMENTS, protocol_config,
                               require_generic_experiment)
from oil_eval.replay import replay_development
from oil_eval.runner import main, run


DELAY_EXPERIMENT = "inventory-availability-delay"


class UnreadableRows:
    def __iter__(self):
        raise AssertionError("A rejected experiment must not inspect rows")


class ExperimentDispatchTests(unittest.TestCase):
    def test_shared_guard_accepts_only_generic_experiments_without_changing_delay_protocol(self):
        self.assertEqual(GENERIC_EXPERIMENTS,
                         tuple(name for name in EXPERIMENTS if name != DELAY_EXPERIMENT))
        for experiment in GENERIC_EXPERIMENTS:
            with self.subTest(experiment=experiment):
                self.assertEqual(require_generic_experiment(experiment), experiment)
                self.assertIsInstance(protocol_config(experiment), dict)
        self.assertEqual(protocol_config(DELAY_EXPERIMENT)["protocol_id"],
                         "wti-inventory-availability-delay-development-v1")
        with self.assertRaisesRegex(ValueError, "python -m oil_eval.delay_runner"):
            require_generic_experiment(DELAY_EXPERIMENT)
        with self.assertRaisesRegex(ValueError, "Unknown development experiment"):
            require_generic_experiment("unknown")

    def test_run_rejects_delay_before_sources_references_runtime_or_output_selection(self):
        with tempfile.TemporaryDirectory(prefix=".test-dispatch-", dir=".") as directory:
            root = Path(directory).relative_to(Path.cwd())
            report = root / "wti-reference.json"
            report.write_bytes(b"immutable synthetic WTI reference\n")
            for destination in (None, report):
                with self.subTest(explicit_report=destination is not None), ExitStack() as stack:
                    stack.enter_context(patch("oil_eval.runner.WTI_REPORT", report))
                    names = ("protocol_config", "load_development_data", "load_reference",
                             "load_inventory_reference", "attach_inventory_features",
                             "load_inventory_comparison_reference", "load_inventory_regularization_reference",
                             "version", "_native_runtime_sha256", "replay_development")
                    mocks = [stack.enter_context(patch("oil_eval.runner." + name,
                              side_effect=AssertionError("Rejected experiment reached " + name)))
                             for name in names]
                    stack.enter_context(patch("oil_eval.runner.Path.glob",
                                              side_effect=AssertionError("Rejected experiment inspected sources")))
                    with self.assertRaisesRegex(ValueError, "python -m oil_eval.delay_runner"):
                        run(object(), root / "outputs", destination, experiment=DELAY_EXPERIMENT,
                            reference_report=object(), inventory_audit_report=object(),
                            inventory_reference_report=object(), regularization_reference_report=object())
                    for operation in mocks:
                        operation.assert_not_called()
                self.assertEqual(report.read_bytes(), b"immutable synthetic WTI reference\n")
                self.assertFalse((root / "outputs").exists())
                self.assertEqual(sorted(path.name for path in root.iterdir()), [report.name])

    def test_replay_rejects_delay_before_inspecting_malformed_inputs_or_calling_predictor(self):
        predictor = Mock(side_effect=AssertionError("Rejected experiment fitted a model"))
        with patch("oil_eval.replay.protocol_config") as config, \
             patch("oil_eval.replay.threadpool_limits") as threads, \
             patch("oil_eval.replay.training_rows") as training, \
             patch("oil_eval.replay.summarize_predictions") as summarize:
            with self.assertRaisesRegex(ValueError, "python -m oil_eval.delay_runner"):
                replay_development(UnreadableRows(), UnreadableRows(), UnreadableRows(),
                                   predictor=predictor, experiment=DELAY_EXPERIMENT,
                                   reference_rows=UnreadableRows(), inventory_reference_rows=UnreadableRows(),
                                   regularization_reference_rows=UnreadableRows())
            for operation in (config, threads, training, summarize, predictor):
                operation.assert_not_called()

    def test_generic_cli_help_omits_dedicated_experiment(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["oil_eval", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_result:
                main()
        self.assertEqual(exit_result.exception.code, 0)
        self.assertNotIn(DELAY_EXPERIMENT, output.getvalue())
        for experiment in GENERIC_EXPERIMENTS:
            self.assertIn(experiment, output.getvalue())

    def test_generic_cli_rejects_delay_with_dedicated_command_before_run(self):
        error = io.StringIO()
        arguments = ["oil_eval", "--dataset-dir", "data/missing",
                     "--experiment", DELAY_EXPERIMENT]
        with patch.object(sys, "argv", arguments), patch("oil_eval.runner.run") as execute, \
             redirect_stderr(error):
            with self.assertRaises(SystemExit) as exit_result:
                main()
        self.assertEqual(exit_result.exception.code, 2)
        self.assertIn("python -m oil_eval.delay_runner", error.getvalue())
        execute.assert_not_called()

    def test_module_cli_rejection_preserves_report_bytes_and_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory(prefix=".test-dispatch-cli-", dir=".") as directory:
            root = Path(directory).relative_to(Path.cwd())
            report = root / "reference.json"
            original = b"immutable synthetic public report\n"
            report.write_bytes(original)
            result = subprocess.run(
                [sys.executable, "-m", "oil_eval", "--dataset-dir", str(root / "missing"),
                 "--experiment", DELAY_EXPERIMENT, "--report", str(report),
                 "--output-root", str(root / "outputs")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("python -m oil_eval.delay_runner", result.stderr)
            self.assertEqual(report.read_bytes(), original)
            self.assertFalse((root / "outputs").exists())
            self.assertEqual(sorted(path.name for path in root.iterdir()), [report.name])


if __name__ == "__main__":
    unittest.main()
