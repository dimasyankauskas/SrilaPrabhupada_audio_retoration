#!/usr/bin/env python3
"""
00_inspect — Characterize the canonical source.

Run:  python3 tools/00_inspect.py

Outputs:
  stages/00--inspect/report.json   — full no-reference metrics
  stages/00--inspect/inspect.md    — human-readable summary
  stages/00--inspect/spectrogram.png  — visual reference for the agent

This stage NEVER modifies the source. It only reads.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_audio as L

# Matplotlib is optional. We import lazily so the rest of the harness
# works in headless / minimal environments.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


def spectrogram_png(data: np.ndarray, sr: int, dest: Path) -> None:
    if not HAVE_PLT:
        return
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.specgram(mono, NFFT=2048, Fs=sr, noverlap=1024, cmap="magma")
    ax.set_ylim(0, min(16_000, sr // 2))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"spectrogram — {dest.parent.name}")
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def main() -> int:
    L.require_cmd("ffmpeg")
    L.require_cmd("ffprobe")
    src = L._require_source()
    print(f"[00_inspect] reading {src.name}")
    t0 = time.time()
    metrics = L.measure(src)
    metrics.runtime_s = time.time() - t0
    out = L.stage_dir("00", "inspect")
    L.write_report(metrics, out / "report.json")

    # Spectrogram is optional. Skip if matplotlib missing.
    if HAVE_PLT:
        data, sr = L.load_wav(src)
        spectrogram_png(data, sr, out / "spectrogram.png")

    # Human-readable summary.
    md = []
    md.append(f"# Inspect — `{src.name}`\n")
    md.append(f"- duration: **{metrics.duration_s:.1f} s**\n")
    md.append(f"- sample rate: **{metrics.sample_rate} Hz**, channels: **{metrics.channels}**, bit depth: **{metrics.bit_depth}**\n")
    md.append(f"- peak: **{metrics.peak_dbfs:.1f} dBFS**, RMS: **{metrics.rms_dbfs:.1f} dBFS**\n")
    md.append(f"- integrated loudness: **{metrics.lufs} LUFS**, true peak: **{metrics.true_peak_dbtp} dBTP**, LRA: **{metrics.dynamic_range_lu} LU**\n")
    md.append(f"- speech band ({L.SPEECH_BAND[0]:.0f}–{L.SPEECH_BAND[1]:.0f} Hz) energy: **{metrics.speech_band_energy_db:.1f} dB**\n")
    md.append(f"- hiss band ({L.HISS_BAND[0]:.0f}–{L.HISS_BAND[1]:.0f} Hz) energy: **{metrics.hiss_band_energy_db:.1f} dB**\n")
    md.append(f"- low band ({L.LOW_BAND[0]:.0f}–{L.LOW_BAND[1]:.0f} Hz) energy: **{metrics.low_band_energy_db:.1f} dB**\n")
    md.append(f"- HF / BWE band ({L.HF_BAND[0]:.0f}–{L.HF_BAND[1]:.0f} Hz) energy: **{metrics.hf_extension_db:.1f} dB** "
              f"(1960s tape can't record above ~10 kHz; modern podcast ≈ -15 dB)\n")
    md.append(f"- spectral centroid: **{metrics.spectral_centroid_hz:.0f} Hz**\n")
    md.append(f"- zero-crossing rate: **{metrics.zero_crossing_rate:.4f}**\n")
    md.append("\n## Environment\n\n```json\n" + json.dumps(L.env_summary(), indent=2) + "\n```\n")
    if HAVE_PLT:
        md.append("\n## Spectrogram\n\n![](spectrogram.png)\n")
    (out / "inspect.md").write_text("".join(md))
    print(f"[00_inspect] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
