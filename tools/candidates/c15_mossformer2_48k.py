#!/usr/bin/env python3
"""
c15_mossformer2_48k — MossFormer2_SE_48K via ClearerVoice-Studio.

Model: alibabasglab/ClearerVoice-Studio, Apache-2.0, modelscope.
MossFormer2_SE_48K is a 48 kHz fullband speech denoiser+dereverber.
The user's specific complaint is "voice recorded in a temple room /
hotel room has audible echo + hiss" — no denoise-only candidate
addresses the echo. This is the only candidate in the harness that
does dereverb (Demucs/DFN/VoiceFixer all do denoise only).

Compared to other candidates:
  - c12 (DeepFilterNet) — denoise only, RTF 0.06, no dereverb
  - c13 (VoiceFixer) — denoise + BWE, weak dereverb
  - c03 (Demucs) — denoise only, aggressive on hiss, no dereverb
  - c01/c02/c04 — classical, no ML, no dereverb at all

MossFormer2 is the only model in this benchmark that explicitly
handles room reverberation. It's also the strongest 48 kHz fullband
denoiser per the ClearerVoice paper.

Settings: default. The model is 48 kHz mono native, no resampling
needed. The 'speech_enhancement' task includes both denoise and
dereverb.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


def have_clearvoice() -> bool:
    try:
        import clearvoice  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c15", "mossformer2_48k")
    out = out_dir / "output.wav"
    print(f"[c15] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_clearvoice():
        print("[c15] SKIPPED: `clearvoice` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    from clearvoice import ClearVoice

    # MossFormer2_SE_48K is 48 kHz mono native. The harness source is
    # always 48 kHz when AUDIO_RESTORE_SOURCE points to a clip from
    # /tmp/audio_restore_clips/, but the raw MP3s in samples/source/
    # are 44.1 kHz. We re-encode to 48k mono to keep the input contract
    # consistent for this model.
    work_src = src
    if src.suffix.lower() in {".mp3", ".m4a", ".ogg", ".flac", ".aif", ".aiff"}:
        # Likely a non-PCM source. Re-encode via ffmpeg to a 48k PCM
        # working file alongside the model output.
        work_src = out_dir / "_src_48k.wav"
        r = subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
            "-i", str(src),
            "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(work_src),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[c15] 48k re-encode failed: {r.stderr}", file=sys.stderr)
            return 1

    print(f"[c15] loading MossFormer2_SE_48K (48 kHz fullband, denoise+dereverb)…", flush=True)
    t_load = time.time()
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    print(f"[c15]   loaded in {time.time()-t_load:.1f}s", flush=True)

    # ClearVoice writes the output wav itself when given output_path +
    # online_write=True. The actual file lands at
    #   <output_path>/<model_name>/<input_basename>
    # so we pass the *parent* dir and move the file after.
    print(f"[c15] running inference on source…", flush=True)
    t0 = time.time()
    cv(input_path=str(work_src), online_write=True, output_path=str(out_dir))
    runtime = time.time() - t0
    print(f"[c15]   inference done in {runtime:.1f}s", flush=True)

    # ClearVoice nested the output under <out_dir>/MossFormer2_SE_48K/.
    # Move it to <out_dir>/output.wav.
    nested = out_dir / "MossFormer2_SE_48K" / work_src.name
    if not nested.exists():
        # Try alternative filenames
        for cand in out_dir.glob("MossFormer2_SE_48K/*"):
            if cand.is_file():
                nested = cand
                break
    if not nested.exists():
        print(f"[c15] FAIL: ClearVoice didn't write output (looked in {nested})", file=sys.stderr)
        return 1
    nested.rename(out)
    # Also clean up the empty MossFormer2_SE_48K dir if it's empty.
    try:
        (out_dir / "MossFormer2_SE_48K").rmdir()
    except OSError:
        pass

    # Loudnorm to podcast target.
    norm = out_dir / "_output_norm.wav"
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(out),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(norm),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c15] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1
    norm.replace(out)

    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c15] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
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
        extras={"status": "skipped: clearvoice package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
