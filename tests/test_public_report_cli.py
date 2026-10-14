import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit import __main__ as price_cli
from oil_audit import reconcile, vintages


class PublicReportCliTests(unittest.TestCase):
    def setUp(self):
        self.original = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.original)

    def check(self, name):
        report = json.loads((Path("reports") / name).read_bytes())
        self.assertEqual(report["public_sanitization"]["report"], name)
        copies = list(Path("data/private-report-archive").glob("*/" + name))
        self.assertEqual(len(copies), 1)
        return report

    def test_price_audit_default_output_cannot_reintroduce_quotes(self):
        fixture = {"series": {"wti": {"minimum": {"date": "2020-01-02", "value": 42.125}}}}
        with patch("sys.argv", ["oil_audit", "inspect", "--manifest", "data/synthetic.json"]), \
             patch.object(price_cli, "run_audit", return_value=fixture), patch("builtins.print"):
            price_cli.main()
        result = self.check("price_audit.json")
        self.assertIsNone(result["series"]["wti"]["minimum"]["value"])

    def test_vintage_audit_default_output_cannot_reintroduce_string_quotes(self):
        def audit(series):
            return {"first_vintage": "2011-04-06", "listed_vintage_count": 1,
                    "date_level_reconstruction_verified_against_samples": [{"latest_value": "SYNTHETIC_QUOTE"}]}
        with patch("sys.argv", ["vintages"]), patch.object(vintages, "ROOT", Path("data/synthetic")), \
             patch.object(vintages, "verify_manifest"), patch.object(vintages, "audit", side_effect=audit), \
             patch("builtins.print"):
            vintages.main()
        result = self.check("vintage_audit.json")
        self.assertNotIn(b"SYNTHETIC_QUOTE", Path("reports/vintage_audit.json").read_bytes())
        self.assertIsNone(result[vintages.SERIES[0]]["date_level_reconstruction_verified_against_samples"][0]["latest_value"])

    def test_reconciliation_default_output_cannot_reintroduce_mismatch_quotes(self):
        fixture = {"sampled_weekly_releases": [{"mismatches": [{"archived_price": 44.5, "current_price": 45.0}]}]}
        with patch("sys.argv", ["reconcile", "--manifest", "data/synthetic.json"]), \
             patch.object(reconcile, "reconcile", return_value=fixture), \
             patch.object(vintages, "ROOT", Path("data/synthetic")), patch("builtins.print"):
            reconcile.main()
        result = self.check("source_reconciliation.json")
        self.assertEqual(result["sampled_weekly_releases"][0]["mismatches"][0],
                         {"archived_price": None, "current_price": None})


if __name__ == "__main__":
    unittest.main()
