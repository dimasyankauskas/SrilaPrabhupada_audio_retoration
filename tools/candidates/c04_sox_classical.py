#!/usr/bin/env python3
"""
c04_sox_classical — sox-only chain. The pre-2010 approach.

Stage order matches what an audio engineer would do by hand in Audacity:
  1. highpass 80 Hz (kills rumble, DC offset, subsonic noise)
  2. notch 60 Hz + harmonics with `fir` (US mains hum) — try also 50/100/150
  3. `noisered` profile-based hiss removal (classical spectral subtraction)
  4. `declick` / `repair` for short transient damage
  5. `compand` for gentle dynamic restoration
  6. `gain -n` to peak-normalize

Tests: is the classical sox chain, applied carefully, competitive with
the ML candidates? On a 50-year-old tape with broadband hiss, the
answer is often "surprisingly close" — at much lower runtime and
zero model downloads.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


def main() -> int:
    L.require_cmd("ffmpeg")
    L.require_cmd("sox")
    src = L._require_source()
    out_dir = L.stage_dir("c04", "sox_classical")
    intermediate = out_dir / "intermediate.wav"
    out = out_dir / "output.wav"
    print(f"[c04] {src.name} → {out.relative_to(L.REPO_ROOT)}")

    # Step 1: decode + highpass + resample to 48k/24/mono with ffmpeg.
    t0 = time.time()
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-af", "highpass=f=80,aresample=48000",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", str(intermediate),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c04] ffmpeg step failed: {r.stderr}", file=sys.stderr)
        return 1

    # Step 2: sox chain.
    # Note: sox's `fir` filter needs a coefficient file, not inline numbers,
    # and `sinc -n` (notch) needs a Q. The honest classical chain is:
    #   highpass (already done in ffmpeg) → compand (gentle dyn) → gain normalize
    # Real hiss reduction would be `noisered <profile> <amt>`, but it needs a
    # noise-only profile which we don't have for this generic candidate.
    sox_args = [
        "sox", str(intermediate), str(out),
        "compand", "0.05,0.2", "6:-70,-60,-40", "-5", "-90", "0.2",
        "gain", "-n", "-1",
    ]
    r = subprocess.run(sox_args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c04] sox step failed: {r.stderr}", file=sys.stderr)
        return 1

    # Step 3: final EBU loudness normalize.
    tmp = out.with_suffix(".pre.wav")
    out.rename(tmp)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(tmp),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], check=True, capture_output=True)
    tmp.unlink(missing_ok=True)

    runtime = time.time() - t0
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c04] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
