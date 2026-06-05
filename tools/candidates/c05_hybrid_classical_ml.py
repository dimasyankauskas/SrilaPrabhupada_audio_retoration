#!/usr/bin/env python3
"""
c05_hybrid_classical_ml — Classical prep + ML denoise + loudnorm.

The hypothesis this tests: classical pre-processing (dehum, declick)
helps the ML model focus on the broadband hiss it was trained for,
instead of trying to also remove hum and clicks. Output quality
should be measurably better than either approach alone.

Pipeline:
  1. ffmpeg highpass + adeclick (remove rumble, transients) → pre.wav
  2. facebook/denoiser (broadband noise) → denoised.wav
  3. ffmpeg loudnorm (target -16 LUFS) → output.wav

This is what a professional audio engineer would attempt by hand:
clean the obvious defects, then run the AI.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


def have_denoiser() -> bool:
    try:
        import denoiser  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c05", "hybrid_classical_ml")
    pre = out_dir / "pre.wav"
    den = out_dir / "denoised.wav"
    out = out_dir / "output.wav"
    print(f"[c05] {src.name} → {out.relative_to(L.REPO_ROOT)}")
    if not have_denoiser():
        print("[c05] SKIPPED: `denoiser` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    import torch
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    t_total = time.time()

    # Step 1: classical prep at 48k/24/mono.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-af", "highpass=f=80,adeclick=t=2:w=10",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", str(pre),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c05] prep failed: {r.stderr}", file=sys.stderr)
        return 1

    # Step 2: ML denoise.
    wav, sr = L.load_wav(pre)
    if wav.ndim > 1:
        wav = wav.mean(axis=1, keepdims=True)
    wav_t = torch.from_numpy(wav.T).float()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = pretrained.dns64().to(device).eval()
    wav_t = convert_audio(wav_t, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        enhanced = model(wav_t.to(device))[0].cpu()
    enhanced_48k = convert_audio(enhanced, model.sample_rate, L.TARGET_SR, 1)
    L.save_wav(den, enhanced_48k.numpy().T, L.TARGET_SR, L.TARGET_SUBTYPE)

    # Step 3: final loudnorm.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(den),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c05] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1

    runtime = time.time() - t_total
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c05] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        spectral_centroid_hz=0.0, zero_crossing_rate=0.0, runtime_s=0.0,
        extras={"status": "skipped: denoiser package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
