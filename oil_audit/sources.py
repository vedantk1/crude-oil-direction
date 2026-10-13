"""Fetch public source artifacts into a timestamped, checksummed snapshot."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCES = {
    "wti": "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls",
    "brent": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
    "spot_release": "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm",
    "daily_snapshot": "https://www.eia.gov/todayinenergy/prices.php",
}
ARCHIVE_ISSUES = ("2011-08-03", "2020-04-22", "2020-04-29", "2024-01-04")
RELEASE_SOURCES = {
    "table11-" + day: "https://www.eia.gov/petroleum/supply/weekly/archive/"
    + day[:4] + "/" + day.replace("-", "_") + "/csv/table11.csv"
    for day in ARCHIVE_ISSUES
}




def fetch_snapshot(output: Path, sources: dict | None = None) -> Path:
    """Download without credentials; preserve bytes and retrieval metadata."""
    started = datetime.now(timezone.utc)
    sources = SOURCES if sources is None else sources
    destination = output / started.strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source_id, url in sources.items():
        request = Request(url, headers={"User-Agent": "oil-data-audit/0.1"})
        with urlopen(request, timeout=60) as response:
            payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError(f"Unexpected source size: {source_id}")
            content_type = response.headers.get("Content-Type", "")
            last_modified = response.headers.get("Last-Modified")
            final_url = response.url
        suffix = Path(urlparse(url).path).suffix
        suffix = suffix if suffix in {".xls", ".csv"} else ".html"
        if suffix == ".xls" and not payload.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
            raise ValueError(f"Expected legacy XLS bytes for {source_id}")
        path = destination / (source_id + suffix)
        path.write_bytes(payload)
        records.append({
            "source_id": source_id,
            "requested_url": url,
            "resolved_url": final_url,
            "file": path.name,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "http_last_modified": last_modified,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        # Retain partial provenance if a later download fails.
        manifest = destination / "manifest.json"
        manifest.write_text(json.dumps({
            "started_at_utc": started.isoformat(),
            "complete": len(records) == len(sources),
            "sources": records,
        }, indent=2) + "\n")
    return manifest
