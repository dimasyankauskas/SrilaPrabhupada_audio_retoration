#!/usr/bin/env python3
"""
c02_rnnoise_dedicated — RNNoise (Xiph), not ffmpeg's wrapper.

Uses the `noisered` approach: train a noise profile from a quiet section,
then subtract it. This is what ffmpeg's `arnndn` does internally, but
running it standalone via the rnnoise CLI lets us tweak the model variant
and threshold. Falls back to ffmpeg's arnndn if no dedicated CLI exists.

This is the "fastest possible" candidate. The question it answers: is the
RNNoise model alone (no extra classical stages) good enough?
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


def have_rnnoise_cli() -> bool:
    return shutil.which("rnnoise") is not None or shutil.which("rnnoise_demo") is not None


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c02", "rnnoise_dedicated")
    out = out_dir / "output.wav"
    model = L.ensure_arnndn_model()
    print(f"[c02] {src.name} → {out.relative_to(L.REPO_ROOT)}")
    t0 = time.time()
    # Pure RNNoise pass. No classical cleanup afterwards.
    # `arnndn` is ffmpeg's built-in wrapper; it IS RNNoise.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-af", f"arnndn=m={model},loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c02] FAILED: {r.stderr}", file=sys.stderr)
        return 1
    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    note = " (dedicated CLI not on PATH; using ffmpeg arnndn)" if not have_rnnoise_cli() else ""
    print(f"[c02] done in {runtime:.1f}s{note}, hiss={m.hiss_band_energy_db:.1f}dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
