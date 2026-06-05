#!/usr/bin/env python3
"""
c09_audio_maister — audio-maister (VoiceFixer fork, "for talks and videos").

Why this candidate exists: the user pasted a third-party recommendation
suggesting audio-maister as the practical "lecture cleanup" tool. It is
a fork of VoiceFixer with a broader training set (general audio, not
just voice) — relevant for 70s lectures where the room ambience, audience,
and tape artifacts all matter. MIT, runs on M1 with the np.complex patch
that librosa<0.10 needs on numpy>=1.20.

Pipeline: audiomaister restore (mode=0, full restoration) → ffmpeg
loudnorm → 24-bit/48kHz mono.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L

# audio-maister (via librosa<0.10 + torchlibrosa) still references
# np.complex, removed in NumPy 1.20+. Must run before audiomaister is
# imported anywhere downstream.
L.patch_numpy_complex()


CHECKPOINT = Path.home() / ".cache" / "huggingface" / "hub" / "models--peterwilli--audio-maister" / "snapshots" / "main" / "audiomaister_v1.5.ckpt"


def have_audiomaister() -> bool:
    try:
        import audiomaister  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c09", "audio_maister")
    pre = out_dir / "pre.wav"  # 48k mono
    out = out_dir / "output.wav"
    print(f"[c09] {src.name} → {out.relative_to(L.REPO_ROOT)}")
    if not have_audiomaister():
        print("[c09] SKIPPED: `audiomaister` not installed. Run `uv pip install 'audiomaister @ git+https://github.com/peterwilli/audio-maister.git'` to enable.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    t_total = time.time()

    # Step 1: decode to 48k mono WAV.
    print(f"[c09] step 1/3: decode → 48k mono PCM…", flush=True)
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(pre),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c09] decode failed: {r.stderr}", file=sys.stderr)
        return 1
    print(f"[c09]   done in {time.time()-t_total:.1f}s", flush=True)

    # Step 2: audio-maister restoration.
    import torch
    from audiomaister import VoiceFixer
    from audiomaister.models.gs_audiomaister import AudioMaister

    if not CHECKPOINT.exists():
        print(f"[c09] checkpoint missing at {CHECKPOINT}, downloading…", file=sys.stderr)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "curl", "-sL", "--retry", "3", "--max-time", "600",
            "-o", str(CHECKPOINT),
            "https://huggingface.co/peterwilli/audio-maister/resolve/main/audiomaister_v1.5.ckpt?download=true",
        ], check=True)
    print(f"[c09] step 2/3: loading model from {CHECKPOINT}…", flush=True)
    t_load = time.time()
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    main_model = VoiceFixer(state["hparams"], 1, "vocals")
    main_model.load_state_dict(state["weights"])
    main_model.eval()
    inference_model = AudioMaister(main_model)
    print(f"[c09]   loaded in {time.time()-t_load:.1f}s, restoring…", flush=True)
    t_restore = time.time()
    # mode=0 → full restoration (denoise + enhance)
    inference_model.restore(input=str(pre), output=str(out), mode=0)
    print(f"[c09]   restored in {time.time()-t_restore:.1f}s", flush=True)

    # Step 3: loudnorm to -16 LUFS.
    print(f"[c09] step 3/3: loudnorm → -16 LUFS…", flush=True)
    tmp = out.with_suffix(".pre.wav")
    out.rename(tmp)
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(tmp),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0:
        print(f"[c09] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1
    print(f"[c09]   loudnorm done", flush=True)

    runtime = time.time() - t_total
    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c09] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB speech={m.speech_band_energy_db:.1f}dB")
    return 0


def _skipped(out_dir: Path) -> L.AudioMetrics:
    return L.AudioMetrics(
        path=str(out_dir / "MISSING"),
        duration_s=0.0, sample_rate=0, channels=0, bit_depth=0,
        peak_dbfs=-200.0, rms_dbfs=-200.0,
        lufs=None, true_peak_dbtp=None, dynamic_range_lu=None,
        hiss_band_energy_db=-200.0, speech_band_energy_db=-200.0, low_band_energy_db=-200.0,
        spectral_centroid_hz=0.0, zero_crossing_rate=0.0, runtime_s=0.0,
        extras={"status": "skipped: audiomaister not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
