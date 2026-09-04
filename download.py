"""Fetch only the archive members this experiment needs.

The Zenodo record is a single 8.5 GB zip (18.3 GB unpacked). The experiment
touches 62 of the 321 volumes, so downloading the whole archive would waste
roughly 80% of the transfer. Zenodo serves the file over HTTP with byte ranges
(verified: `Accept-Ranges` is honoured, a range request returns 206), and a zip
stores its table of contents at the *end* of the file. So:

    1. range-fetch the last few MB  -> zip64 end-of-central-directory
    2. parse the central directory  -> name, size and offset of all 1256 members
    3. extract the metadata CSVs, which happen to live in that same tail
    4. work out which volumes are needed, and range-fetch only those members

    python download.py --index     # steps 1-3 only, a few MB
    python download.py             # everything the experiment needs, ~1.7 GB
    python download.py --full      # every member, ~8.5 GB

Members are stored deflated; we inflate them ourselves rather than shelling out
to a zip tool, because we are only ever holding one member in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import ARCHIVE_PREFIX, CACHE, DATA, ZIP_SIZE, ZIP_URL

INDEX_FILE = CACHE / "zip_index.json"
TAIL_BYTES = 4 * 1024 * 1024
UA = {"User-Agent": "montaging-us-transfer/1.0 (research; python-urllib)"}


# --------------------------------------------------------------------------- #
# HTTP


def fetch_range(start: int, end: int, retries: int = 4) -> bytes:
    """Inclusive byte range [start, end]."""
    headers = dict(UA)
    headers["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ZIP_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status not in (200, 206):
                    raise OSError(f"HTTP {r.status}")
                return r.read()
        except Exception as exc:  # network flakiness is expected on 1.7 GB
            last = exc
            time.sleep(2 * (attempt + 1))
    raise OSError(f"range {start}-{end} failed after {retries} tries: {last}")


# --------------------------------------------------------------------------- #
# zip parsing


def parse_central_directory(tail: bytes, tail_start: int) -> list[dict]:
    """Read the zip64 central directory out of the archive's tail."""
    j = tail.rfind(b"PK\x06\x06")
    if j < 0:
        raise ValueError("no zip64 end-of-central-directory in tail")
    total, = struct.unpack_from("<Q", tail, j + 32)
    cd_offset, = struct.unpack_from("<Q", tail, j + 48)
    p = cd_offset - tail_start
    if p < 0:
        raise ValueError("central directory is not inside the fetched tail")

    entries: list[dict] = []
    while tail[p:p + 4] == b"PK\x01\x02":
        (_v, _vn, _flag, meth, _mt, _md, crc, csz, usz,
         nlen, elen, clen, _dsk, _iat, _eat, lho) = struct.unpack_from(
            "<HHHHHHIIIHHHHHII", tail, p + 4)
        name = tail[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = tail[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (usz, csz, lho):
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, q)
                body, o = extra[q + 4:q + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xFFFFFFFF:
                        usz, = struct.unpack_from("<Q", body, o); o += 8
                    if csz == 0xFFFFFFFF:
                        csz, = struct.unpack_from("<Q", body, o); o += 8
                    if lho == 0xFFFFFFFF:
                        lho, = struct.unpack_from("<Q", body, o); o += 8
                q += 4 + hsz
        entries.append(dict(name=name, usz=usz, csz=csz, lho=lho, meth=meth, crc=crc))
        p += 46 + nlen + elen + clen

    if len(entries) != total:
        raise ValueError(f"parsed {len(entries)} entries, header says {total}")
    return entries


def member_bytes(entry: dict, blob: bytes | None = None, base: int = 0) -> bytes:
    """Inflate one member, either from an already-fetched blob or over HTTP."""
    if blob is None:
        # local header is 30 bytes plus two variable fields; grab a slack window
        raw = fetch_range(entry["lho"], entry["lho"] + entry["csz"] + 4096)
        off = 0
    else:
        raw, off = blob, entry["lho"] - base
    if raw[off:off + 4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {entry['name']}")
    nlen, elen = struct.unpack_from("<HH", raw, off + 26)
    start = off + 30 + nlen + elen
    data = raw[start:start + entry["csz"]]
    out = zlib.decompress(data, -15) if entry["meth"] == 8 else data
    if len(out) != entry["usz"]:
        raise ValueError(f"{entry['name']}: got {len(out)}, expected {entry['usz']}")
    if zlib.crc32(out) != entry["crc"]:
        raise ValueError(f"{entry['name']}: CRC mismatch")
    return out


# --------------------------------------------------------------------------- #
# steps


def build_index() -> list[dict]:
    """Download the tail, parse the directory, and write out the small files."""
    tail_start = ZIP_SIZE - TAIL_BYTES
    print(f"fetching last {TAIL_BYTES/1e6:.0f} MB of the archive for its index...")
    tail = fetch_range(tail_start, ZIP_SIZE - 1)
    entries = parse_central_directory(tail, tail_start)
    INDEX_FILE.write_text(json.dumps(entries))
    print(f"  {len(entries)} members, "
          f"{sum(e['usz'] for e in entries)/1e9:.2f} GB unpacked")

    n = 0
    for e in entries:
        if e["name"].endswith("/") or e["lho"] < tail_start:
            continue
        write_member(e, member_bytes(e, tail, tail_start))
        n += 1
    print(f"  extracted {n} small members that were already in the tail")
    return entries


def write_member(entry: dict, data: bytes) -> Path:
    rel = entry["name"]
    if rel.startswith(ARCHIVE_PREFIX):
        rel = rel[len(ARCHIVE_PREFIX):]
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def read_csv(name: str) -> list[dict]:
    path = DATA / "metadata" / name
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `python download.py --index` first")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def needed_members(entries: list[dict]) -> list[dict]:
    """Volumes and masks for the segmented volumes, plus transforms and metadata."""
    by_name = {e["name"]: e for e in entries}
    segs = read_csv("humerus_segmentations.csv")
    vols = read_csv("ultrasound_volumes.csv")
    trials = read_csv("trials.csv")

    wanted: set[str] = set()
    for r in segs:
        for key in ("mhd_path", "data_file_path", "stl_path"):
            if r[key]:
                wanted.add(ARCHIVE_PREFIX + r[key])
    seg_keys = {(r["trial_label"], int(r["volume_number"])) for r in segs}
    for v in vols:
        if (v["trial_label"], int(v["volume_number"])) in seg_keys:
            wanted.add(ARCHIVE_PREFIX + v["mhd_path"])
            wanted.add(ARCHIVE_PREFIX + v["data_file_path"])
    for t in trials:
        wanted.add(ARCHIVE_PREFIX + t["pose_estimation_calibrated_robot_transform"])
        wanted.add(ARCHIVE_PREFIX + t["hybrid_refined_registration_transform"])

    missing = sorted(n for n in wanted if n not in by_name)
    if missing:
        raise SystemExit(f"{len(missing)} needed members are not in the archive index:\n"
                         + "\n".join(missing[:5]))
    return [by_name[n] for n in sorted(wanted)]


def download(members: list[dict], workers: int = 8) -> None:
    """Fetch members concurrently.

    Each member is an independent range request, so the only shared state is the
    progress counter. Eight workers is enough to saturate the link without
    looking like a scraper to Zenodo.
    """
    todo = [e for e in members
            if not ((DATA / e["name"][len(ARCHIVE_PREFIX):]).exists()
                    and (DATA / e["name"][len(ARCHIVE_PREFIX):]).stat().st_size == e["usz"])]
    skipped = len(members) - len(todo)
    total = sum(e["csz"] for e in todo)
    state = {"done": 0, "n": 0}
    lock = threading.Lock()
    t0 = time.time()

    def one(e: dict) -> None:
        write_member(e, member_bytes(e))
        with lock:
            state["done"] += e["csz"]
            state["n"] += 1
            el = time.time() - t0
            rate = state["done"] / max(el, 1e-9)
            eta = (total - state["done"]) / max(rate, 1e-9)
            print(f"  [{state['n']:3d}/{len(todo)}] "
                  f"{state['done']/1e9:5.2f}/{total/1e9:.2f} GB "
                  f"{rate/1e6:5.1f} MB/s  eta {eta/60:4.1f} min  "
                  f"{Path(e['name']).name}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(one, todo):
                pass
    print(f"done: {total/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min "
          f"({skipped} members already present)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true",
                    help="only fetch the archive index and the small metadata files")
    ap.add_argument("--full", action="store_true",
                    help="download every member (~8.5 GB) instead of the subset")
    ap.add_argument("--workers", type=int, default=8, help="concurrent range requests")
    args = ap.parse_args()

    entries = (json.loads(INDEX_FILE.read_text())
               if INDEX_FILE.exists() and not args.index else build_index())
    if args.index:
        return

    members = ([e for e in entries if not e["name"].endswith("/")]
               if args.full else needed_members(entries))
    print(f"{len(members)} members to fetch, "
          f"{sum(e['csz'] for e in members)/1e9:.2f} GB compressed / "
          f"{sum(e['usz'] for e in members)/1e9:.2f} GB on disk")
    download(members, workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
