#!/usr/bin/env python3
"""
c12_deepfilternet — DeepFilterNet3, 48 kHz-native speech denoiser.

Model: Rikorose/DeepFilterNet3 (Aug 2023), 2.1M params, MIT, 48 kHz mono.
Originally designed for real-time voice calls (VoIP, Zoom) but
community reports and the paper show it preserves more high-frequency
content than Demucs/c03 with fewer "musical noise" artifacts — the
exact complaint the user has about c03 sounding "thin".

Settings tuned for archive tape hiss (per community guidance):
  - post_filter=True: slightly over-attenuates noisy sections
  - atten_lim_db=24:    cap noise reduction at 24 dB (vs ~12 dB VoIP
                       default). 24 dB is the community-recommended
                       value for archive restoration; higher starts
                       clipping speech sibilants.

Backend: CPU. MPS is broken on M1/M2/M3 per upstream Issues
#118, #121, #135, #142 (NaN outputs). torch+torchaudio from PyPI
on M1 = CPU only. Estimated RTF 0.1 on M1 CPU for 48 kHz mono
(measured 0.1 on 1s test signal; should hold for longer clips).

Import shim: the published deepfilternet 0.5.6 imports
`torchaudio.backend.common.AudioMetaData`, which was removed in
torchaudio 2.4+ (we have 2.11). We don't actually use that class
(we load WAVs via L.load_wav and pass tensors directly), but the
import chain still tries to pull it in. The shim installs a stub
class before the first `from df...` import.

Length: the 10-minute clip fits in one pass (no chunking needed at
this size; the 50-min 32-bit-index bug is GPU-only).

Compare with:
  - c03 — Demucs, aggressive hiss (-35 LUFS, sounds "thin")
  - c11 — c03 + loudnorm (fixes loudness, keeps thin voice)
  - c07-c10f — Resemble denoise, natural voice but weak on hiss
"""
from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_audio as L


def _install_torchaudio_shim() -> None:
    """Stub the removed `torchaudio.backend.common.AudioMetaData` so that
    `from df.enhance import enhance, init_df` works on torchaudio 2.4+.

    We never actually call AudioMetaData in this script — we use
    L.load_wav + torch.from_numpy — but the import chain evaluates
    the type annotation at module load time, so a stub class is
    required for the import to succeed.
    """
    import torchaudio as ta
    if hasattr(ta, "backend") and hasattr(ta.backend, "common"):
        return  # already present (torchaudio < 2.4)
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # minimal stub — never instantiated
        def __init__(self, *a, **kw):
            pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    ta.backend = backend
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


# Install the shim at module load so `import df` works from
# have_deepfilternet() onwards. The shim is a no-op on older torchaudio.
_install_torchaudio_shim()


def have_deepfilternet() -> bool:
    try:
        import df  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    L.require_cmd("ffmpeg")
    src = L._require_source()
    out_dir = L.stage_dir("c12", "deepfilternet")
    den = out_dir / "denoised.wav"
    out = out_dir / "output.wav"
    print(f"[c12] {src.name} → {out.relative_to(L.REPO_ROOT)}", flush=True)
    if not have_deepfilternet():
        print("[c12] SKIPPED: `deepfilternet` package not installed.")
        L.write_report(_skipped(out_dir), out_dir / "report.json")
        return 0

    _install_torchaudio_shim()

    import torch
    import numpy as np
    from df.enhance import enhance, init_df

    print(f"[c12] loading audio…", flush=True)
    wav, sr = L.load_wav(src)
    if wav.ndim > 1:
        wav = wav.mean(axis=1, keepdims=True)
    if sr != L.TARGET_SR:
        # DeepFilterNet3 is 48 kHz native and will not resample
        # internally. The harness always produces 48k clips via
        # 00_inspect / make_clip, so this is a guard rail, not the
        # happy path. We refuse rather than silently resample with
        # an unknown algorithm — the clip is wrong, fix the upstream.
        raise RuntimeError(
            f"[c12] source is {sr}Hz, DeepFilterNet3 requires 48kHz mono. "
            f"Re-run with a 48k clip (e.g. via `make clip`)."
        )
    wav_t = torch.from_numpy(wav.T.astype(np.float32))  # [1, T], float32
    # CPU only — MPS produces NaN on M1/M2/M3 (upstream Issues #118, #121, #135).
    device = "cpu"
    print(f"[c12] using device: {device} (MPS broken for DFN3, per upstream issues)", flush=True)

    print(f"[c12] loading DeepFilterNet3 model…", flush=True)
    t_load = time.time()
    # post_filter=True — slightly over-attenuates noisy sections (the
    # README documents this as the "archive" knob).
    model, df_state, _ = init_df(
        post_filter=True, log_level="WARNING", log_file=None,
    )
    print(f"[c12]   {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
          f"loaded in {time.time()-t_load:.1f}s", flush=True)

    print(f"[c12] running inference on {wav_t.shape[-1]/sr:.0f}s of audio…", flush=True)
    t0 = time.time()
    with torch.no_grad():
        # atten_lim_db=24: cap noise reduction at 24 dB. The model default
        # (~12 dB) is tuned for VoIP and leaves too much hiss on a 1970s
        # tape. Community guidance: 24 dB is the sweet spot for archive
        # restoration; higher starts to clip speech sibilants.
        enhanced = enhance(model, df_state, wav_t, pad=True, atten_lim_db=24)
    runtime = time.time() - t0
    print(f"[c12]   inference done in {runtime:.1f}s, writing denoised.wav…", flush=True)
    out_2d = enhanced.numpy().T  # [T, 1]
    L.save_wav(den, out_2d, sr, L.TARGET_SUBTYPE)

    # Step 3: loudnorm pass. Same target as every other loudnorm'd
    # candidate (c01/c05/c07/c08/c09/c11). This is the same fix that
    # turned c03 into c11 — DeepFilterNet's output is also loudness-
    # uncorrected, so we land at -16 LUFS for the podcast target.
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(den),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(L.TARGET_SR), "-ac", "1", "-c:a", "pcm_s24le", str(out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[c12] loudnorm failed: {r.stderr}", file=sys.stderr)
        return 1

    m = L.measure(out, runtime_s=runtime)
    L.write_report(m, out_dir / "report.json")
    print(f"[c12] done in {runtime:.1f}s, hiss={m.hiss_band_energy_db:.1f}dB "
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
        extras={"status": "skipped: deepfilternet package not installed"},
    )


if __name__ == "__main__":
    sys.exit(main())
