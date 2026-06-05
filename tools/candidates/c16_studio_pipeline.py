#!/usr/bin/env python3
"""
c16_studio_pipeline — DeepFilterNet3 → AudioSR → loudnorm.

This is the "combine the two best" pipeline. DeepFilterNet3 (c12) is
the cleanest conservative denoiser, and AudioSR (c14) is the only
candidate that does BWE. Running them in series gets us:
  - Less hiss than DFN alone (DFN is conservative; AudioSR's input
    side still sees hiss in the 5-12 kHz band, but the diffusion model
    tends to mask it)
  - More HF than DFN alone (the whole point — this is the
    bandwidth-extension step)
  - Less musical noise than VoiceFixer (c13), which is known to
    have artifacts from the TFGAN vocoder

Reuses c12's output if it exists. Same skip-gates as the underlying
candidates. Loudnorm at the end for the podcast target.

This is the candidate that should win on the "studio quality" metric
if the user wants natural-sounding speech with HF extension. c17 is
the alternative if the user prefers the one-shot simplicity of
VoiceFixer over the two-step pipeline.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# audiosr needs the np.complex patch before import.
import lib_audio as L
L.patch_numpy_complex()
L.patch_audiosr_librosa()

# DeepFilterNet3 (df package) needs the torchaudio 2.4+ shim.
# Reuse c12's installer so the import chain evaluates correctly.
import importlib
_c12 = importlib.import_module("c12_deepfilternet")
_c12._install_torchaudio_shim()


CHUNK_S = 10.0
CROSSFADE_S = 0.5


def have_audiosr() -> bool:
    try:
        import audiosr  # noqa: F401
        return True
    except Exception:
        return False


def have_deepfilternet() -> bool:
    try:
        import df  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c16", "studio_pipeline")
    out = out_dir / "output.wav"
    print(f"[c16] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)

    if not have_deepfilternet() or not have_audiosr():
        missing = []
        if not have_deepfilternet(): missing.append("deepfilternet")
        if not have_audiosr(): missing.append("audiosr")
        print(f"[c16] SKIPPED: missing packages: {', '.join(missing)}")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    # Step 1: locate or run c12.
    c12_out = L.stage_dir("c12", "deepfilternet") / "output.wav"
    if not c12_out.exists():
        print(f"[c16] c12 output missing; running c12 first…", flush=True)
        # Re-run via subprocess so the patch + c12 import is isolated.
        r = subprocess.run(
            [sys.executable, str(L.CANDIDATES_DIR / "c12_deepfilternet.py")],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not c12_out.exists():
            print(f"[c16] c12 failed: {r.stderr[-500:]}", file=sys.stderr)
            return 1

    # Step 2: BWE on c12's output. We feed c12's output to AudioSR —
    # the input is already 48 kHz mono (c12 does loudnorm to that),
    # so no resample. We use the same chunking as c14 to stay
    # memory-safe.
    import numpy as np
    import soundfile as sf
    import audiosr

    wav, sr = L.load_wav(c12_out)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != L.TARGET_SR:
        raise RuntimeError(f"[c16] c12 output is {sr}Hz, expected 48kHz")
    total_s = wav.shape[0] / sr
    print(f"[c16] BWE step: {total_s:.1f}s of DFN-denoised audio", flush=True)

    print(f"[c16] loading AudioSR speech model…", flush=True)
    t_load = time.time()
    model = audiosr.build_model(model_name="speech", device="cpu")
    print(f"[c16]   loaded in {time.time()-t_load:.1f}s", flush=True)

    chunk_samples = int(CHUNK_S * sr)
    n_chunks = (wav.shape[0] + chunk_samples - 1) // chunk_samples
    chunk_paths: list[Path] = []
    t0 = time.time()
    for i in range(n_chunks):
        s = i * chunk_samples
        e = min(s + chunk_samples, wav.shape[0])
        chunk = wav[s:e]
        chunk_p = out_dir / f"_in_{i:03d}.wav"
        sf.write(str(chunk_p), chunk.astype(np.float32), sr, subtype="PCM_24")
        wav_out = audiosr.super_resolution(
            model, str(chunk_p), ddim_steps=10, guidance_scale=3.5,
        )
        out_samples = wav_out.squeeze().cpu().numpy() if hasattr(wav_out, "cpu") else np.asarray(wav_out).squeeze()
        out_p = out_dir / f"_out_{i:03d}.wav"
        sf.write(str(out_p), out_samples.astype(np.float32), sr, subtype="PCM_24")
        chunk_paths.append(out_p)
        try:
            chunk_p.unlink()
        except OSError:
            pass
        print(f"[c16]   chunk {i+1}/{n_chunks} done "
              f"(elapsed {time.time()-t0:.1f}s)", flush=True)

    # Concat with crossfade.
    if len(chunk_paths) == 1:
        chunk_paths[0].rename(out)
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error"]
        for p in chunk_paths:
            cmd.extend(["-i", str(p)])
        prev = "0:a"
        flt = []
        for i in range(1, len(chunk_paths)):
            cur = f"{i}:a"
            out_label = f"out{i}"
            flt.append(f"[{prev}][{cur}]acrossfade=d={CROSSFADE_S}:c1=tri:c2=tri[{out_label}]")
            prev = out_label
        cmd.extend([
            "-filter_complex", ";".join(flt),
            "-map", f"[{prev}]",
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s24le", str(out),
        ])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[c16] crossfade failed: {r.stderr}", file=sys.stderr)
            return 1
        for p in chunk_paths:
            try:
                p.unlink()
            except OSError:
                pass

    # Final loudnorm.
    norm = out_dir / "_output_norm.wav"
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(out),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(norm),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c16] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1
    norm.replace(out)

    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c16] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
          f"HF={m.hf_extension_db:.1f}dB LUFS={m.lufs}")
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
        extras={"status": "skipped: c16 requires deepfilternet and audiosr"},
    )


if __name__ == "__main__":
    sys.exit(main())
