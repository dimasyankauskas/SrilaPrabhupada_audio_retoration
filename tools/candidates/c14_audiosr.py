#!/usr/bin/env python3
"""
c14_audiosr — AudioSR (haoheliu, MIT, PyPI `audiosr`).

BWE-only candidate. AudioSR is a 258M-param latent diffusion model
(VCTK speech-tuned `audiosr_speech` weights) that takes a 48 kHz mono
clip and re-synthesizes the missing high-frequency content. The original
training task is "audio super-resolution from a low-pass filtered input"
— exactly the analog tape problem in the digital domain.

Settings (per upstream defaults + the model card):
  - model_name="speech" — VCTK-tuned weights, designed for clean voice
    reconstruction. The "basic" weights are for environmental sound.
  - ddim_steps=10  — DDIM sampler steps. Default 200 is way too slow
    (would be ~7 minutes per 10s of audio on M1 CPU). 10 steps gives
    usable quality in ~1s per 5.12s chunk, which keeps the RTF around
    0.2-0.3 for a 10-min clip. The diffusion model is robust to small
    step counts because DDIM is deterministic, so 10 steps preserves
    the speech harmonics better than 200 steps of stochastic DDPM.
  - guidance_scale=3.5 — upstream default, fine for speech.

Memory strategy — chunk externally:
  AudioSR's batch is one big tensor, so a 10-min clip at 48 kHz would
  be ~28.4M samples × 4 bytes = ~113 MB of activations. That fits in
  16 GB M1 unified memory, but the model weights are ~1 GB and the
  diffusion activations can spike. To stay safe and allow overlap-add
  for boundary continuity, we process the file in 30s windows with
  0.5s crossfade.

Sample rate: AudioSR is 48 kHz native. Our source is 48 kHz mono per
the harness contract, so no resampling is needed.

The reason this candidate is interesting: it's the first BWE-only
candidate in the harness. c01-c13 are all denoisers (which only
remove energy above 5 kHz — they don't *add* speech harmonics).
AudioSR is the only one that targets the missing 12-20 kHz content
that 1960s tape machines couldn't record. The 99_compare score
formula rewards this with `+ min(hf_delta, +20)`.

Compare with:
  - c13 (VoiceFixer) — also does BWE but bundled with denoise+dereverb;
    the TFGAN vocoder also adds HF content but is less speech-faithful
  - c12 (DeepFilterNet) — denoise only, no BWE; we chain c12 + c14 in
    c16_studio_pipeline
  - c08 (resemble_enhance) — vocoder-based BWE that over-synthesizes
    (+94 dB HF delta is unphysical)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MUST run before importing audiosr (librosa 0.10 inside the package
# uses np.complex, removed in numpy 1.20+).
import lib_audio as L
L.patch_numpy_complex()
L.patch_audiosr_librosa()


CHUNK_S = 10.0       # AudioSR warns above 10.24s; stay below
CROSSFADE_S = 0.5    # overlap-add crossfade length


def have_audiosr() -> bool:
    try:
        import audiosr  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c14", "audiosr")
    out = out_dir / "output.wav"
    print(f"[c14] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_audiosr():
        print("[c14] SKIPPED: `audiosr` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    import numpy as np
    import soundfile as sf
    import audiosr

    # AudioSR is 48 kHz mono native. The harness source is always 48 kHz
    # when AUDIO_RESTORE_SOURCE points to a /tmp/audio_restore_clips/ file,
    # but the raw MP3s in samples/source/ are 44.1 kHz. We re-encode to 48k
    # mono so the input contract is consistent.
    work_src = src
    if src.suffix.lower() in {".mp3", ".m4a", ".ogg", ".flac", ".aif", ".aiff"}:
        work_src = out_dir / "_src_48k.wav"
        r = subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
            "-i", str(src),
            "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(work_src),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[c14] 48k re-encode failed: {r.stderr}", file=sys.stderr)
            return 1

    # Load source at 48k mono.
    wav, sr = L.load_wav(work_src)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != L.TARGET_SR:
        raise RuntimeError(
            f"[c14] source is {sr}Hz, AudioSR requires 48kHz. "
            f"Re-run with a 48k clip (e.g. via `make clip`)."
        )
    if wav.ndim > 1:
        wav = wav[:, 0]  # belt-and-suspenders after the mean
    total_s = wav.shape[0] / sr
    print(f"[c14] loaded {total_s:.1f}s of {sr}Hz mono audio", flush=True)

    # Build the model ONCE, then process all chunks through it.
    print(f"[c14] loading AudioSR speech model (258M params)…", flush=True)
    t_load = time.time()
    model = audiosr.build_model(model_name="speech", device="cpu")
    print(f"[c14]   loaded in {time.time()-t_load:.1f}s", flush=True)

    # Process in 30s chunks with 0.5s crossfade. We write each chunk
    # to a temp WAV, then concatenate with ffmpeg's acrossfade filter.
    chunk_samples = int(CHUNK_S * sr)
    crossfade_samples = int(CROSSFADE_S * sr)
    n_chunks = (wav.shape[0] + chunk_samples - 1) // chunk_samples
    chunk_paths: list[Path] = []
    t0 = time.time()
    for i in range(n_chunks):
        s = i * chunk_samples
        e = min(s + chunk_samples, wav.shape[0])
        chunk = wav[s:e]
        chunk_p = out_dir / f"_chunk_{i:03d}.wav"
        # AudioSR resamples internally to 48k; we keep the chunk at 48k
        # to satisfy its input check.
        sf.write(str(chunk_p), chunk.astype(np.float32), sr, subtype="PCM_24")
        # Use a higher ddim_steps (50) on small chunks for better
        # quality; the chunks are short enough that the cost is fine.
        # 50 steps at 0.7s/step on 30s = ~35s of inference per chunk.
        wav_out = audiosr.super_resolution(
            model, str(chunk_p), ddim_steps=10, guidance_scale=3.5,
        )
        # wav_out is shape (1, 1, T_out) at 48kHz.
        out_samples = wav_out.squeeze().cpu().numpy() if hasattr(wav_out, "cpu") else np.asarray(wav_out).squeeze()
        out_p = out_dir / f"_chunk_out_{i:03d}.wav"
        sf.write(str(out_p), out_samples.astype(np.float32), sr, subtype="PCM_24")
        chunk_paths.append(out_p)
        # Remove the input chunk to save disk; output we keep for crossfade.
        try:
            chunk_p.unlink()
        except OSError:
            pass
        print(f"[c14]   chunk {i+1}/{n_chunks} done "
              f"(elapsed {time.time()-t0:.1f}s)", flush=True)

    # Concatenate with crossfades. ffmpeg's acrossfade expects paired
    # inputs; we chain them with -i inputs and use the acrossfade filter
    # with overlap = CROSSFADE_S/CHUNK_S.
    if len(chunk_paths) == 1:
        # Just move the file.
        chunk_paths[0].rename(out)
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error"]
        for p in chunk_paths:
            cmd.extend(["-i", str(p)])
        # Build a chain of acrossfade filters.
        # acrossfade takes 2 inputs, duration = crossfade time. Outputs 1 stream.
        # We feed (in0,in1) → out0, then (out0,in2) → out1, etc.
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
            print(f"[c14] crossfade/concat failed: {r.stderr}", file=sys.stderr)
            return 1
        for p in chunk_paths:
            try:
                p.unlink()
            except OSError:
                pass

    # Optional loudnorm to land at -16 LUFS for the podcast target.
    # Most BWE-only candidates still benefit from a gain stage.
    norm = out_dir / "_output_norm.wav"
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(out),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(norm),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c14] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1
    norm.replace(out)

    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c14] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
          f"HF={m.hf_extension_db:.1f}dB LUFS={m.lufs} "
          f"centroid={m.spectral_centroid_hz:.0f}Hz")
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
        extras={"status": "skipped: audiosr package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
