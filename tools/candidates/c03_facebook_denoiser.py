#!/usr/bin/env python3
"""
c03_facebook_denoiser — Meta's Demucs-based speech denoiser.

Model: facebook/denoiser (the speech enhancement model, NOT the music
separation one). 33 M params, MIT, 16 kHz mono internal, real-time on
laptop CPU. On M1 it auto-detects MPS and uses it if available,
otherwise falls back to CPU.

This is the candidate that matters most: it is the strongest open
ML model specifically trained for noisy speech, small enough to live
in 16 GB unified memory comfortably, and the only one with a paper +
DNS Challenge results behind it. No classical prep, no loudnorm —
the pure ML reference. Compare with c05 (hybrid) to see if prep helps.
"""
from __future__ import annotations

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
    out_dir = L.stage_dir("c03", "facebook_denoiser")
    out_wav = out_dir / "output.wav"
    print(f"[c03] {src.name} → {out_wav.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_denoiser():
        print("[c03] SKIPPED: `denoiser` package not installed. Run `uv pip install denoiser` to enable.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    import torch
    import torchaudio
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    print(f"[c03] loading audio…", flush=True)
    wav, sr = L.load_wav(src)
    if wav.ndim > 1:
        wav = wav.mean(axis=1, keepdims=True)
    wav_t = torch.from_numpy(wav.T).float()  # [1, T]
    # Demucs uses a 1-D conv config that torch's MPS backend does not
    # implement ("convolution_overrideable not implemented"). It works for
    # very short clips on MPS but fails on long ones. CPU is reliable;
    # unified memory means the speed penalty is moderate.
    device = "cpu"
    print(f"[c03] using device: {device} (Demucs not MPS-compatible)", flush=True)

    print(f"[c03] loading model…", flush=True)
    model = pretrained.dns64().to(device).eval()
    print(f"[c03]   model loaded, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    print(f"[c03] running inference on {wav_t.shape[-1]/sr:.0f}s of audio…", flush=True)
    t0 = time.time()
    wav_t = convert_audio(wav_t, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        enhanced = model(wav_t.to(device))[0].cpu()
    print(f"[c03]   inference done in {time.time()-t0:.1f}s, upsampling to 48k…", flush=True)
    enhanced_48k = convert_audio(enhanced, model.sample_rate, L.TARGET_SR, 1)
    runtime = time.time() - t0

    out_wav_2d = enhanced_48k.numpy().T  # [T, 1]
    print(f"[c03]   writing output.wav ({out_wav_2d.shape})…", flush=True)
    L.save_wav(out_wav, out_wav_2d, L.TARGET_SR, L.TARGET_SUBTYPE)
    print(f"[c03]   measuring metrics…", flush=True)
    m = L.measure(out_wav, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c03] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    """Placeholder so compare.md shows a clean ⏭ row instead of a -200 floor."""
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0,
        low_band_energy_db=-200.0, spectral_centroid_hz=0.0, zero_crossing_rate=0.0,
        runtime_s=0.0,
        extras={"status": "skipped: denoiser package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
