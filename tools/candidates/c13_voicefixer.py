#!/usr/bin/env python3
"""
c13_voicefixer — VoiceFixer, ICASSP 2022, 117M-param TFGAN vocoder + ResUNet.

Model: haoheliu/voicefixer (Liu et al. 2022), MIT, originally designed for
"severely degraded speech, such as real-world historical speech recordings."
The 3-stage pipeline is: ResUNet analyzer (VAD + degradation classification)
→ specialist subnets (denoiser, declipper, super-resolver) → TFGAN vocoder.

The interesting thing for our user is that VoiceFixer combines **denoising +
super-resolution** in one pass — c03 (Demucs) and c12 (DeepFilterNet) only
denoise. VoiceFixer re-synthesizes high-frequency content via the TFGAN
vocoder, which is the user's specific complaint about c03 ("doesn't sound
like podcast deep voice" — the high-frequency energy is missing).

Settings tuned for archive tape (per upstream source code in base.py):
  - mode=0 (default): no preprocessing, eval mode. The "default" mode is
    correct for archive audio. mode=1 pre-zeros HF before the model sees
    it (good for low-bitrate codecs, BAD for hiss — we want HF preserved).
  - cuda=False: CPU-only path. On M1, the voicefixer2 fork supports MPS
    by setting cuda=True, but upstream voicefixer==0.1.3 (what's on PyPI)
    ignores MPS — torch's CUDA path just runs slower on CPU.

Import shim: voicefixer==0.1.3 (the published package) imports librosa
0.9.2, which uses np.complex — removed in NumPy 1.20+. We have NumPy
1.26.4 (pinned in the venv). L.patch_numpy_complex() restores the alias
as `complex` so the import chain succeeds. Without this shim, the
candidate would never load (librosa can't be imported at all).

Sample rate: VoiceFixer's vocoder is hardcoded to 44.1 kHz. We have to
resample 48k→44.1k before inference, then 44.1k→48k after. The
resampling is done by ffmpeg (sox-resample quality), not VoiceFixer's
internal path, so it doesn't pollute the SR-conversion artifacts the
user already hears from c03.

Length: VoiceFixer's restore() handles arbitrary-length audio internally
(its 30s-windowed inference concatenates output), so a 10-min clip
runs in one pass.

Memory: model is 117M params + 135MB vocoder checkpoint loaded to
~/.cache/voicefixer/ on first import. Subsequent runs reuse the cache.

Compare with:
  - c03 — Demucs, aggressive hiss (-35 LUFS, sounds "thin")
  - c11 — c03 + loudnorm (fixes loudness, keeps thin voice)
  - c12 — DeepFilterNet, faster (RTF 0.04) but conservative on hiss
  - c07-c10f — Resemble denoise, natural voice but weak on hiss
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MUST run before importing voicefixer (librosa 0.9.2 inside the package
# uses np.complex, which is removed in numpy 1.20+). This is a no-op on
# numpy <1.20.
import lib_audio as L
L.patch_numpy_complex()


# VoiceFixer vocoder SR — hardcoded inside the model checkpoint
# (TFGAN was trained on 44.1 kHz LJSpeech data).
VF_SR = 44_100


def have_voicefixer() -> bool:
    try:
        from voicefixer import VoiceFixer  # noqa: F401
        return True
    except Exception:
        return False


def _ffmpeg_resample(src: Path, dst: Path, sr: int) -> None:
    """Resample src → dst at the given rate, preserving 24-bit PCM mono.

    Uses ffmpeg's soxr resampler (the project default) for the best
    quality. We use this rather than VoiceFixer's internal path so the
    SR conversion happens outside the model — keeps the artifact
    surface small.
    """
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s24le", str(dst),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {r.stderr}")


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c13", "voicefixer")
    den = out_dir / "denoised.wav"
    out = out_dir / "output.wav"
    print(f"[c13] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_voicefixer():
        print("[c13] SKIPPED: `voicefixer` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    # Step 1: resample 48k → 44.1k for VoiceFixer. The source is 48k mono
    # per the harness contract. We don't trust internal resamplers inside
    # voicefixer/restorer.py — the harness's ffmpeg path is what we use
    # for c01-c12 too, so the SR-conversion artifacts are consistent
    # across candidates.
    print(f"[c13] resampling {L.TARGET_SR}Hz → {VF_SR}Hz for VoiceFixer…", flush=True)
    vf_in = out_dir / "_vf_in_44k.wav"
    _ffmpeg_resample(src, vf_in, VF_SR)

    # Step 2: VoiceFixer inference. mode=0 = eval + no preprocessing
    # (correct for archive audio; mode=1 pre-zeros HF and would defeat
    # the hiss removal we want). cuda=False → CPU on M1. The voicefixer
    # package supports cuda=True on CUDA GPUs only — on M1 it falls
    # back to CPU regardless.
    from voicefixer import VoiceFixer
    print(f"[c13] loading VoiceFixer (117M params, TFGAN vocoder)…", flush=True)
    t_load = time.time()
    vf = VoiceFixer()
    print(f"[c13]   loaded in {time.time()-t_load:.1f}s", flush=True)

    print(f"[c13] running inference on {VF_SR}Hz audio…", flush=True)
    t0 = time.time()
    vf.restore(input=str(vf_in), output=str(den), cuda=False, mode=0)
    runtime = time.time() - t0
    print(f"[c13]   inference done in {runtime:.1f}s", flush=True)

    # Step 3: resample back to 48k mono. Then loudnorm to -16 LUFS for
    # the podcast target, same as c01/c05/c07/c08/c09/c11/c12.
    print(f"[c13] resampling {VF_SR}Hz → {L.TARGET_SR}Hz + loudnorm…", flush=True)
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(den),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c13] loudnorm/resample failed: {r.stderr}", file=sys.stderr)
        return 1

    # Cleanup intermediate 44.1k input — saves disk, not strictly needed.
    try:
        vf_in.unlink()
    except OSError:
        pass

    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c13] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
          f"speech={m.speech_band_energy_db:.1f}dB LUFS={m.lufs}")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        hf_extension_db=-200.0, spectral_centroid_hz=0.0, zero_crossing_rate=0.0,
        runtime_s=0.0,
        extras={"status": "skipped: voicefixer package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
