#!/usr/bin/env python3
"""
encode_focus_mp3s — Encode 3-min MP3 clips for the focus candidates on
each sample, so the public comparison.html has working audio on every cell.

For each (sample, focus_candidate) pair:
  1. Find stages/<sample>/<cand>--<name>/output.wav
  2. ffmpeg → 3min (180s) MP3 128kbps mono 48kHz
  3. Write to assets/audio/<sample>_<cand>_3min.mp3

Skips files that already exist with non-zero size.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import run_multi  # noqa: E402

FOCUS_CANDS = [
    "c12_deepfilternet",
    "c13_voicefixer",
    "c14_audiosr",
    "c15_mossformer2_48k",
    "c16_studio_pipeline",
    "c17_studio_voicefixer",
]
DURATION_S = 180  # 3 min


def main() -> int:
    n_encoded = 0
    n_skipped = 0
    n_missing = 0

    for src, start_s, dur_s in run_multi.SOURCES:
        sample = run_multi.safe_name(src)
        # Use the 5min-equivalent start so all samples start at a coherent
        # spot in the source (mid-recording, not the very start).
        # Paris is short and 0-3min is its only sensible range.
        for cand in FOCUS_CANDS:
            # Stage dir is c<NN>--<name>
            cand_id = cand.split("_", 1)[0]  # e.g. "c12"
            cand_name = cand.split("_", 1)[1]  # e.g. "deepfilternet"
            stage_dir = REPO_ROOT / "stages" / sample / f"{cand_id}--{cand_name}"
            wav = stage_dir / "output.wav"
            if not wav.exists():
                n_missing += 1
                print(f"  skip: {sample}/{cand}: missing {wav.name}")
                continue
            out_mp3 = REPO_ROOT / "assets" / "audio" / f"{sample}_{cand_id}_3min.mp3"
            if out_mp3.exists() and out_mp3.stat().st_size > 1024:
                n_skipped += 1
                continue
            out_mp3.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
                "-i", str(wav),
                "-ss", "60", "-t", str(DURATION_S),  # start 60s in, take 3min
                "-ac", "1", "-ar", "48000",
                "-codec:a", "libmp3lame", "-b:a", "128k",
                str(out_mp3),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAIL: {sample}/{cand}: {r.stderr}")
                continue
            n_encoded += 1
            size_mb = out_mp3.stat().st_size / 1024 / 1024
            print(f"  ok: {sample}/{cand} → {out_mp3.name} ({size_mb:.1f} MB)")

    print(f"\n[done] {n_encoded} encoded, {n_skipped} cached, {n_missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
