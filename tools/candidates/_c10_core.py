#!/usr/bin/env python3
"""
_c10_core — shared logic for c07-tuning experiments.

The c07 candidate (Resemble denoise) uses all defaults. This module
exposes a knob table (PRESETS) and a `run()` function that performs the
same denoise pipeline as c07 but with overrides for `chunk_seconds`,
`overlap_seconds`, and `preemphasis`.

Real knobs available on the denoiser path (verified by inspection):
  - chunk_seconds  (default 30.0) — sliding-window size
  - overlap_seconds (default 1.0) — crossfade length between chunks
  - preemphasis    (default 0.97) — input high-end emphasis; mutated on
    `enhancer.denoiser.hp.preemphasis` before calling inference.

The CFM-solver knobs (cfm_solver_nfe, force_gaussian_prior) used by
the full Enhancer (c08) DO NOT apply here — the Denoiser is feed-forward,
no diffusion prior.

Used by c10a..c10f candidate wrappers in this directory. They look up
their preset by stage id (c10a → baseline, etc.) and call run().
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Make sure tools/lib_audio.py is importable when wrappers run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L

# resemble-enhance transitively imports librosa 0.10 which still
# references np.complex, removed in NumPy 1.20+. Must patch before
# any resemble import.
L.patch_numpy_complex()


# Preset name → knob overrides. baseline == c07's defaults exactly.
PRESETS: dict[str, dict] = {
    "baseline":   dict(chunk_seconds=30.0, overlap_seconds=1.0, preemphasis=0.97),
    "chunk_60":   dict(chunk_seconds=60.0, overlap_seconds=1.0, preemphasis=0.97),
    "chunk_15":   dict(chunk_seconds=15.0, overlap_seconds=1.0, preemphasis=0.97),
    "overlap_2":  dict(chunk_seconds=30.0, overlap_seconds=2.0, preemphasis=0.97),
    "preemph_85": dict(chunk_seconds=30.0, overlap_seconds=1.0, preemphasis=0.85),
    "preemph_70": dict(chunk_seconds=30.0, overlap_seconds=1.0, preemphasis=0.70),
}

# Maps wrapper file's stage id to its preset name. The 6 candidate files
# (c10a..c10f) parse their own filename and look up the preset here.
PRESET_BY_STAGE: dict[str, str] = {
    "c10a": "baseline",
    "c10b": "chunk_60",
    "c10c": "chunk_15",
    "c10d": "overlap_2",
    "c10e": "preemph_85",
    "c10f": "preemph_70",
}


def have_resemble() -> bool:
    try:
        import resemble_enhance  # noqa: F401
        return True
    except ImportError:
        return False


def _skipped(out_dir: Path, preset: str) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        spectral_centroid_hz=0.0, zero_crossing_rate=0.0, runtime_s=0.0,
        extras={
            "status": "skipped: resemble-enhance not installed",
            "preset": preset,
        },
    )


def run(stage_id: str, stage_name: str) -> int:
    """Run one c10x variant end-to-end. Returns process exit code.

    Mirrors c07's flow but uses resemble_enhance.denoiser.inference.inference
    directly so we can pass chunk_seconds / overlap_seconds, and mutates
    `enhancer.denoiser.hp.preemphasis` before inference.
    """
    L.require_cmd("ffmpeg")
    preset_name = PRESET_BY_STAGE.get(stage_id)
    if preset_name is None:
        print(f"[{stage_id}] unknown stage id; expected one of {list(PRESET_BY_STAGE)}",
              file=sys.stderr)
        return 2
    preset = PRESETS[preset_name]

    src = L._require_source()
    out_dir = L.stage_dir(stage_id, stage_name)
    pre = out_dir / "pre.wav"
    den = out_dir / "denoised.wav"
    out = out_dir / "output.wav"
    print(f"[{stage_id}] {src.name} → {out.relative_to(L.REPO_ROOT)}  preset={preset_name}  knobs={preset}")
    if not have_resemble():
        print(f"[{stage_id}] SKIPPED: resemble-enhance not installed.")
        L.write_report(_skipped(out_dir, preset_name), out_dir / "report.json")
        return 0

    t_total = time.time()

    # Step 1: decode to 48k mono WAV.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(pre),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{stage_id}] decode failed: {r.stderr}", file=sys.stderr)
        return 1

    # Step 2: load model, mutate preemphasis, run the denoiser's inference.
    import torch
    import torchaudio
    from resemble_enhance.enhancer.inference import load_enhancer
    from resemble_enhance.denoiser.inference import inference as d_inference

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enhancer = load_enhancer(None, device)
    # Mutate the input high-end emphasis knob. HParams is a frozen
    # dataclass, so we have to construct a new one with dataclasses.replace
    # and reassign the model's `hp` field. (Verified by attribute walk that
    # this is the only HParam knob on the Denoiser path that affects output.)
    import dataclasses
    enhancer.denoiser.hp = dataclasses.replace(
        enhancer.denoiser.hp, preemphasis=preset["preemphasis"],
    )

    wav, sr = torchaudio.load(str(pre))
    wav = wav.mean(0)  # mono [T]
    hwav, sr_out = d_inference(
        model=enhancer.denoiser,
        dwav=wav, sr=sr, device=device,
        chunk_seconds=preset["chunk_seconds"],
        overlap_seconds=preset["overlap_seconds"],
    )
    torchaudio.save(str(den), hwav[None].cpu(), sr_out)

    # Step 3: loudnorm → final 24-bit/48kHz mono.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(den),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{stage_id}] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1

    runtime = time.time() - t_total
    m = L.measure(out, runtime_s=runtime)
    # Stash the preset knobs in extras so the compare table can show them.
    m.extras["preset"] = preset_name
    m.extras["knobs"] = preset
    L.write_report(m, out_dir / "report.json")
    print(f"[{stage_id}] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
          f"speech={m.speech_band_energy_db:.1f}dB")
    return 0
