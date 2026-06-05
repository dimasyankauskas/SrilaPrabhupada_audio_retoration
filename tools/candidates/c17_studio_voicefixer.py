#!/usr/bin/env python3
"""
c17_studio_voicefixer — VoiceFixer (one-shot denoise+dereverb+BWE) → loudnorm.

VoiceFixer is the only single-model candidate in the harness that does
denoise + dereverb + BWE in one pass. This is the "all-in-one" alternative
to c16 (which is a two-step DFN+AudioSR pipeline).

We rely on c13's output if it exists; otherwise we run c13. This makes
c17 effectively a "rename + measure" candidate, but it's worth having
in the ranking because:
  1. It guarantees the c13 output is loudnorm'd to -16 LUFS (c13 already
     does this internally; we re-do it for consistency with c16).
  2. It exposes the VoiceFixer output under a clear "studio" name so
     the user knows it's the BWE-enabled candidate.
  3. The metric comparison vs c16 is meaningful — both are end-to-end
     BWE pipelines, the user picks based on which sounds better.

This is the candidate that should win on metric if the user prefers
the one-shot simplicity of VoiceFixer over a multi-step pipeline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L

# voicefixer uses torchlibrosa (via librosa), which calls pad_center
# positionally — same upstream-librosa-0.11+ breakage audiosr hits.
# Re-apply the patch in case the parent process didn't.
try:
    L.patch_audiosr_librosa()  # actually patches librosa.util.pad_center
except Exception:
    pass
try:
    L.patch_numpy_complex()
except Exception:
    pass


def have_voicefixer() -> bool:
    try:
        from voicefixer import VoiceFixer  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c17", "studio_voicefixer")
    out = out_dir / "output.wav"
    print(f"[c17] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_voicefixer():
        print("[c17] SKIPPED: `voicefixer` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    # Step 1: locate or run c13. We import c13 in-process so the np.complex
    # patch and the torchlibrosa shim are reused without re-applying in a
    # subprocess (the patch state lives in the current interpreter).
    c13_out = L.stage_dir("c13", "voicefixer") / "output.wav"
    if not c13_out.exists():
        print(f"[c17] c13 output missing; running c13 in-process…", flush=True)
        import c13_voicefixer as c13
        rc = c13.main()
        if rc != 0 or not c13_out.exists():
            print(f"[c17] c13 failed (rc={rc})", file=sys.stderr)
            return 1

    # Step 2: re-loudnorm to -16 (c13 already does this, but a second
    # pass tightens the result and ensures the final report reflects
    # the loudnorm gain).
    norm = out_dir / "_output_norm.wav"
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(c13_out),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(norm),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c17] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1
    norm.replace(out)

    # Use c13's reported runtime so the score's RTF column is correct.
    c13_report = L.stage_dir("c13", "voicefixer") / "report.json"
    runtime = 0.0
    if c13_report.exists():
        import json
        runtime = json.loads(c13_report.read_text()).get("runtime_s", 0.0) or 0.0

    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c17] done (reused c13 runtime={runtime:.1f}s), "
          f"hiss={m.hiss_band_energy_db:.1f}dB HF={m.hf_extension_db:.1f}dB "
          f"LUFS={m.lufs}")
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
