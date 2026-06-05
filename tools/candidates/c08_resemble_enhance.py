#!/usr/bin/env python3
"""
c08_resemble_enhance — Resemble AI's full enhance pipeline.

This is the "quality ceiling" candidate. On top of denoising, the
full enhance runs a conditional flow-matching (CFM) prior that
extends bandwidth and polishes perceptual quality. The cost is
~4x slower than denoise-only (~RTF 1.7 on M1).

At 1000+ file scale this means ~70 days of single-machine
processing — useful as a quality reference and for the most
prized recordings, but too slow to be the default.

Pipeline: enhance (nfe=16, solver=euler for ~2x speedup over
the default nfe=64 midpoint) → ffmpeg loudnorm → 24-bit/48kHz mono.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L

# resemble-enhance transitively imports librosa 0.10 which still
# references np.complex, removed in NumPy 1.20+. Must patch before
# any resemble import.
L.patch_numpy_complex()


def have_resemble() -> bool:
    try:
        import resemble_enhance  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c08", "resemble_enhance")
    pre = out_dir / "pre.wav"
    enh = out_dir / "enhanced.wav"
    out = out_dir / "output.wav"
    print(f"[c08] {src.name} → {out.relative_to(L.REPO_ROOT)}")
    if not have_resemble():
        print("[c08] SKIPPED: `resemble-enhance` not installed. Run `uv pip install 'resemble-enhance @ git+https://github.com/resemble-ai/resemble-enhance'` to enable.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    t_total = time.time()

    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(pre),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c08] decode failed: {r.stderr}", file=sys.stderr)
        return 1

    import torch
    import torchaudio
    from resemble_enhance.enhancer.inference import enhance as resemble_enhance_fn
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wav, sr = torchaudio.load(str(pre))
    wav = wav.mean(0)
    # nfe=16 + euler: ~2x faster than default (nfe=64, midpoint), 90% of the quality
    hwav, sr_out = resemble_enhance_fn(
        dwav=wav, sr=sr, device=device, run_dir=None,
        nfe=16, solver="euler",
    )
    torchaudio.save(str(enh), hwav[None].cpu(), sr_out)

    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(enh),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c08] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1

    runtime = time.time() - t_total
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c08] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        spectral_centroid_hz=0.0, zero_crossing_rate=0.0, runtime_s=0.0,
        extras={"status": "skipped: resemble-enhance not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
