#!/usr/bin/env python3
"""
c11_facebook_denoiser_loudnorm — c03 + a final loudnorm pass.

This is the "fixed" version of c03. c03 (facebook/denoiser) is the
strongest denoiser in the harness but its raw output sits around
-35 LUFS — way too quiet to be useful as-is. c11 adds the same
ffmpeg loudnorm pass that every other candidate uses, landing at
-16 LUFS (podcast standard).

Compare with:
  - c03 — same denoiser, no loudnorm. Pure ML reference. Sounds
    quiet and thin. Useful for understanding what the model alone
    does.
  - c05 — c03 + classical prep (highpass, declick) + loudnorm.
    The "engineer would do this" hybrid.
  - c11 — c03 + loudnorm only. Isolates the loudness fix.

The c11 row in compare.md should show the same hiss Δ and speech Δ
as c03, but with LUFS out around -16 to -17 instead of -35. If
your ears say c11 sounds "good" but c05 sounds "better", the
difference is the classical pre-processing, not the loudnorm.
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
    out_dir = L.stage_dir("c11", "facebook_denoiser_loudnorm")
    den = out_dir / "denoised.wav"
    out = out_dir / "output.wav"
    print(f"[c11] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_denoiser():
        print("[c11] SKIPPED: `denoiser` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    import torch
    import torchaudio
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    print(f"[c11] loading audio…", flush=True)
    wav, sr = L.load_wav(src)
    if wav.ndim > 1:
        wav = wav.mean(axis=1, keepdims=True)
    wav_t = torch.from_numpy(wav.T).float()  # [1, T]
    # Demucs 1-D conv config: MPS fails on long clips. CPU is reliable.
    device = "cpu"
    print(f"[c11] using device: {device}", flush=True)

    print(f"[c11] loading model…", flush=True)
    model = pretrained.dns64().to(device).eval()
    print(f"[c11]   {sum(p.numel() for p in model.parameters())/1e6:.1f}M params loaded", flush=True)

    print(f"[c11] running inference on {wav_t.shape[-1]/sr:.0f}s of audio…", flush=True)
    t0 = time.time()
    wav_t = convert_audio(wav_t, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        enhanced = model(wav_t.to(device))[0].cpu()
    # Upsample to 48 kHz (the model's internal rate is 16 kHz).
    enhanced_48k = convert_audio(enhanced, model.sample_rate, L.TARGET_SR, 1)
    out_2d = enhanced_48k.numpy().T  # [T, 1]
    print(f"[c11]   inference done in {time.time()-t0:.1f}s, writing denoised.wav…", flush=True)
    L.save_wav(den, out_2d, L.TARGET_SR, L.TARGET_SUBTYPE)

    # Step 3: loudnorm pass. Same target as c01/c05/c07/c08/c09.
    # This is the entire reason c11 exists vs c03.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(den),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c11] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1

    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c11] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
          f"speech={m.speech_band_energy_db:.1f}dB LUFS={m.lufs}")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        spectral_centroid_hz=0.0, zero_crossing_rate=0.0,
        runtime_s=0.0,
        extras={"status": "skipped: denoiser package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
