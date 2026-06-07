#!/usr/bin/env python3
"""
encode_focus_mp3s — Encode 3-min MP3 clips for ALL candidates on every
sample, so the public comparison.html has working audio on every cell.

For each (sample, candidate) pair:
  1. Find stages/<sample>/<cand>--<name>/output.wav
  2. ffmpeg → 3min (180s) MP3 128kbps mono 48kHz
  3. Write to assets/audio/<sample>_<cand_id>_3min.mp3

Skips files that already exist with non-zero size.

Scans tools/candidates/c*.py to discover all candidate IDs, so it
automatically picks up new candidates without needing to be edited.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import run_multi  # noqa: E402

DURATION_S = 180  # 3 min


def _all_candidate_ids() -> list[str]:
    """Scan tools/candidates/c*.py and return the c<NN> prefix for each."""
    out = []
    for p in sorted((REPO_ROOT / "tools" / "candidates").glob("c*.py")):
        stem = p.stem  # e.g. c10a_resemble_baseline
        idx = stem.find("_")
        if idx < 0:
            continue
        out.append(stem[:idx])
    return out


def main() -> int:
    n_encoded = 0
    n_skipped = 0
    n_missing = 0

    all_cands = _all_candidate_ids()
    print(f"[encode] processing {len(all_cands)} candidates × "
          f"{len(run_multi.SOURCES)} samples")

    for src, _start_s, _dur_s in run_multi.SOURCES:
        sample = run_multi.safe_name(src)
        for cand_id in all_cands:
            # Find the matching stage dir. cand_id is "c12" or "c10a".
            # The stage dir is "c<id>--<name>" so glob for it.
            matches = list((REPO_ROOT / "stages" / sample).glob(f"{cand_id}--*"))
            if not matches:
                n_missing += 1
                print(f"  skip: {sample}/{cand_id}: no stage dir")
                continue
            stage_dir = matches[0]
            wav = stage_dir / "output.wav"
            if not wav.exists():
                n_missing += 1
                print(f"  skip: {sample}/{cand_id}: missing {wav.name}")
                continue
            out_mp3 = REPO_ROOT / "assets" / "audio" / f"{sample}_{cand_id}_3min.mp3"
            if out_mp3.exists() and out_mp3.stat().st_size > 1024:
                n_skipped += 1
                continue
            out_mp3.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
                "-i", str(wav),
                "-ss", "60", "-t", str(DURATION_S),
                "-ac", "1", "-ar", "48000",
                "-codec:a", "libmp3lame", "-b:a", "128k",
                str(out_mp3),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAIL: {sample}/{cand_id}: {r.stderr}")
                continue
            n_encoded += 1
            size_mb = out_mp3.stat().st_size / 1024 / 1024
            print(f"  ok: {sample}/{cand_id} → {out_mp3.name} ({size_mb:.1f} MB)")

    print(f"\n[done] {n_encoded} encoded, {n_skipped} cached, {n_missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
