"""Download the LifeSnaps dataset from Zenodo into the git-ignored ``data/`` folder.

Source: Yfantidou et al. (2022), Zenodo record 7229547 (version 4, latest at the
time of writing), licence CC-BY-4.0. Only the CSV exports and the scored surveys
are extracted, plus the small SEMA3 BSON dump (prompt-level schedule/expiry
times are only there). The 9.7 GB Fitbit BSON dump is left inside the zip because the CSV
exports already contain the day-level Fitbit + SEMA3 data this project needs.

Usage::

    python scripts/download_data.py            # download + extract
    python scripts/download_data.py --no-extract

The archive is ~615 MB. Its MD5 checksum is verified against the value that
Zenodo reports for the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "7229547"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Members of the zip that this project uses (prefix match).
WANTED_PREFIXES = (
    "csv_rais_anonymized/",
    "scored_surveys/",
    "mongo_rais_anonymized/sema.bson",  # 11 MB: needed for prompt-level ESM compliance
)


def _record_metadata() -> dict:
    with urllib.request.urlopen(API_URL, timeout=60) as resp:
        return json.load(resp)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(dest_dir: Path = DATA_DIR) -> Path:
    meta = _record_metadata()
    licence = meta["metadata"].get("license", {}).get("id")
    if licence != "cc-by-4.0":
        sys.exit(f"Unexpected licence {licence!r}; stopping (expected cc-by-4.0).")
    (file_meta,) = [f for f in meta["files"] if f["key"].endswith(".zip")]
    expected_md5 = file_meta["checksum"].split(":", 1)[1]
    url = file_meta["links"]["self"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / file_meta["key"]
    if target.exists() and _md5(target) == expected_md5:
        print(f"{target} already present and verified.")
        return target
    print(f"Downloading {url} ({file_meta['size'] / 1e6:.0f} MB) -> {target}")
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if done % (50 << 20) < (1 << 20):
                print(f"  {done / 1e6:.0f} MB", flush=True)
    got = _md5(tmp)
    if got != expected_md5:
        sys.exit(f"MD5 mismatch: got {got}, expected {expected_md5}")
    tmp.rename(target)
    return target


def extract(archive: Path, dest_dir: Path = DATA_DIR) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            # find the wanted folder anywhere in the path (archive may have a top folder)
            for prefix in WANTED_PREFIXES:
                idx = name.find(prefix)
                if idx >= 0 and not info.is_dir():
                    rel = Path(name[idx:])
                    if ".." in rel.parts or rel.is_absolute():
                        raise ValueError(f"unsafe path in archive: {name}")
                    out = dest_dir / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, out.open("wb") as dst:
                        while block := src.read(1 << 20):
                            dst.write(block)
                    print(f"  extracted {rel}")
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    archive = download()
    if not args.no_extract:
        extract(archive)


if __name__ == "__main__":
    main()
