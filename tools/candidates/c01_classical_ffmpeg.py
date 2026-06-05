#!/usr/bin/env python3
"""
c01_classical_ffmpeg — Pure ffmpeg filter chain. No ML.

Filter graph:
  highpass 80Hz  →  arnndn (RNNoise, fast RNN)  →
  anlmdn (Non-Local Means)  →  adeclick  →  loudnorm

This is the baseline: no PyTorch, no model downloads, ~MB of code.
It is the floor against which every ML candidate must justify its cost.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


CHAIN = (
    "highpass=f=80,"
    "anlmdn=s=12:p=0.002:r=0.003,"
    "adeclick=t=2:w=10,"   # ffmpeg: w must be in [10, 100]; was 4 (out of range)
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c01", "classical_ffmpeg")
    out = out_dir / "output.wav"
    model = L.ensure_arnndn_model()
    chain = f"highpass=f=80,arnndn=m={model},anlmdn=s=12:p=0.002:r=0.003,adeclick=t=2:w=10,loudnorm=I=-16:TP=-1.5:LRA=11"
    print(f"[c01] {src.name} → {out.relative_to(L.REPO_ROOT)}")
    t0 = time.time()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-af", chain,
        "-ar", str(L.TARGET_SR),
        "-ac", "1",
        "-c:a", "pcm_s24le",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c01] FAILED: {r.stderr}", file=sys.stderr)
        return 1
    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c01] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
