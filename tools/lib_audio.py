"""
Shared helpers for the audio_restore evaluation harness.

Conventions:
- All scripts read from samples/source/ and write to stages/<id>--<name>/
- The original sample is NEVER modified. Only read.
- Outputs use 24-bit/48 kHz mono WAV unless the script says otherwise
- Every run writes a JSON sidecar with metrics
- Loudness is measured with ffmpeg's loudnorm (EBU R128) because pyloudnorm
  drifts on broadband tape hiss
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "samples" / "source"
STAGES_DIR = REPO_ROOT / "stages"
REPORTS_DIR = REPO_ROOT / "reports"
CANDIDATES_DIR = Path(__file__).resolve().parent / "candidates"
CACHE_DIR = REPO_ROOT / ".cache" / "models"

# Output format: CD-quality is 16/44.1; we target 24/48 because that's what
# most audio editors and archive-grade formats expect. Mono is preserved
# unless the script specifies stereo upmix.
TARGET_SR = 48_000
TARGET_SUBTYPE = "PCM_24"

# Frequency band boundaries (Hz). The "hiss" band is where 70s tape noise
# concentrates, so a good denoiser should reduce energy in this band while
# preserving energy in the speech band. The "hf" band is the missing
# 12-18 kHz content that 1960s tape machines couldn't record — a
# bandwidth-extension (BWE) candidate should *raise* this above the
# source, the way a modern podcast has energy out to 18-20 kHz.
HISS_BAND = (5_000.0, 12_000.0)
SPEECH_BAND = (300.0, 3_400.0)
LOW_BAND = (20.0, 200.0)
HF_BAND = (12_000.0, 18_000.0)


@dataclass
class AudioMetrics:
    """No-reference audio quality metrics. None of these need a clean target."""
    path: str
    duration_s: float
    sample_rate: int
    channels: int
    bit_depth: int
    peak_dbfs: float
    rms_dbfs: float
    lufs: float | None
    true_peak_dbtp: float | None
    dynamic_range_lu: float | None
    hiss_band_energy_db: float  # energy in [5k, 12k] Hz
    speech_band_energy_db: float  # energy in [300, 3.4k] Hz
    low_band_energy_db: float  # energy in [20, 200] Hz
    hf_extension_db: float  # energy in [12k, 18k] Hz — measure of bandwidth extension
    spectral_centroid_hz: float
    zero_crossing_rate: float
    runtime_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


# Multi-sample support: when AUDIO_RESTORE_SOURCE is set, _require_source
# returns that path verbatim instead of scanning samples/source/. When
# AUDIO_RESTORE_SAMPLE is set, stage_dir() nests stages under that sample
# name so concurrent samples don't clobber each other. Both env vars are
# read once at import. The default (env unset) preserves the original
# single-source behavior.
_CURRENT_SAMPLE: str | None = os.environ.get("AUDIO_RESTORE_SAMPLE") or None


def _current_sample() -> str | None:
    """Active sample name (or None when running single-source)."""
    return _CURRENT_SAMPLE


def _require_source() -> Path:
    """Return the source path. Honors AUDIO_RESTORE_SOURCE override; otherwise
    scans samples/source/ alphabetically (legacy behavior)."""
    override = os.environ.get("AUDIO_RESTORE_SOURCE")
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"AUDIO_RESTORE_SOURCE points to missing file: {p}")
        return p
    if not SOURCE_DIR.exists():
        sys.exit(f"missing source dir: {SOURCE_DIR}")
    files = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff"}
    )
    if not files:
        sys.exit(
            f"no audio in {SOURCE_DIR}\n"
            f"Place the canonical sample there (e.g. 760706AD-WASHINGTON DC.MP3)"
        )
    if len(files) > 1 and not _CURRENT_SAMPLE:
        # In multi-sample mode, multiple sources in samples/source/ is expected.
        # Only warn for legacy single-source runs.
        print(f"warning: multiple sources found, using {files[0].name}", file=sys.stderr)
    return files[0]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file via soundfile. Returns (samples [N,C], sr)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data, sr


def save_wav(
    path: Path,
    data: np.ndarray,
    sr: int,
    subtype: str = TARGET_SUBTYPE,
) -> None:
    """Write audio with the project's target format. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile requires contiguous C-order float32 for predictability.
    sf.write(str(path), np.ascontiguousarray(data), sr, subtype=subtype)


def db(x: float) -> float:
    """Linear amplitude to dBFS. Floors at -200 to avoid log(0)."""
    return 20.0 * np.log10(max(abs(x), 1e-10))


def band_energy_db(data: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """Energy (dB) in the band [fmin, fmax] Hz. Mono-mix first if multichannel."""
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    if mono.size < 64:
        return -200.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return -200.0
    e = float(np.mean(spec[mask] ** 2))
    return 10.0 * np.log10(max(e, 1e-20))


def spectral_centroid_hz(data: np.ndarray, sr: int) -> float:
    """Frequency-weighted mean of the magnitude spectrum."""
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    if mono.size < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    denom = float(spec.sum())
    if denom <= 0:
        return 0.0
    return float(np.sum(freqs * spec) / denom)


def zero_crossing_rate(data: np.ndarray) -> float:
    """Fraction of consecutive samples that cross zero. High = noisy/clicky."""
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    if mono.size < 2:
        return 0.0
    signs = np.sign(mono)
    # Treat exact zeros as positive to avoid phantom crossings.
    signs[signs == 0] = 1
    return float(np.mean(signs[:-1] != signs[1:]))


def ffprobe_json(path: Path) -> dict[str, Any]:
    """Run ffprobe and return the parsed stream metadata."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def lufs_via_ffmpeg(path: Path) -> dict[str, float | None]:
    """EBU R128 integrated loudness + true peak + loudness range, via ffmpeg.

    ffmpeg prints the loudnorm summary as a JSON block on stderr. The keys
    of interest are input_i, input_tp, input_lra. We regex-match them so we
    don't depend on ffmpeg's exact stderr layout.
    """
    out = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # Loudnorm prints even when the audio is very short / silent, but if
    # it errored out entirely, return Nones.
    if out.returncode != 0 and "input_i" not in out.stderr:
        return {"lufs": None, "true_peak_dbtp": None, "lra": None}
    import re
    def grab(key: str) -> float | None:
        m = re.search(rf'"{key}"\s*:\s*"(-?\d+(?:\.\d+)?)"', out.stderr)
        return float(m.group(1)) if m else None
    return {
        "lufs": grab("input_i"),
        "true_peak_dbtp": grab("input_tp"),
        "lra": grab("input_lra"),
    }


def measure(path: Path, runtime_s: float = 0.0) -> AudioMetrics:
    """Compute the full no-reference metric set for one audio file."""
    data, sr = load_wav(path)
    if data.ndim > 1:
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
    else:
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
    info = ffprobe_json(path)
    duration = float(info.get("format", {}).get("duration", 0.0))
    channels = data.shape[1] if data.ndim > 1 else 1
    # ffprobe reports bit_depth only for PCM; for MP3 it shows 0.
    bit_depth = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            try:
                bit_depth = int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0)
            except (TypeError, ValueError):
                bit_depth = 0
            break
    lufs_block = lufs_via_ffmpeg(path)
    return AudioMetrics(
        path=str(path),
        duration_s=duration,
        sample_rate=sr,
        channels=channels,
        bit_depth=bit_depth,
        peak_dbfs=db(peak),
        rms_dbfs=db(rms),
        lufs=lufs_block["lufs"],
        true_peak_dbtp=lufs_block["true_peak_dbtp"],
        dynamic_range_lu=lufs_block["lra"],
        hiss_band_energy_db=band_energy_db(data, sr, *HISS_BAND),
        speech_band_energy_db=band_energy_db(data, sr, *SPEECH_BAND),
        low_band_energy_db=band_energy_db(data, sr, *LOW_BAND),
        hf_extension_db=band_energy_db(data, sr, *HF_BAND),
        spectral_centroid_hz=spectral_centroid_hz(data, sr),
        zero_crossing_rate=zero_crossing_rate(data),
        runtime_s=runtime_s,
    )


def write_report(metrics: AudioMetrics, dest: Path) -> Path:
    """Write a JSON sidecar with the metrics. Returns the path written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics)
    payload["measured_at"] = datetime.now(timezone.utc).isoformat()
    with open(dest, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return dest


def stage_dir(stage_id: str, name: str, sample: str | None = None) -> Path:
    """Output dir for a stage. Layout:

        single-source (default): stages/<id>--<name>/
        multi-sample (sample=... or AUDIO_RESTORE_SAMPLE set):
            stages/<sample>/<id>--<name>/

    The `sample` arg wins over the env var; the env var wins over the
    default. Auto-creates.
    """
    leaf = f"{stage_id}--{name}"
    sample = sample or _CURRENT_SAMPLE
    d = (STAGES_DIR / sample / leaf) if sample else (STAGES_DIR / leaf)
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_cmd(cmd: str) -> None:
    """Exit if an external command isn't on PATH."""
    if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
        sys.exit(f"required command not found on PATH: {cmd}")


# Some pinned third-party ML packages (resemble_enhance via librosa 0.10,
# audio-maister via torchlibrosa) still reference np.complex, which was
# removed in NumPy 1.20+. Patching at import-time is the only way to keep
# the original packages working on a modern numpy without forking them.
def patch_numpy_complex() -> None:
    """Restore np.complex / np.float as the deprecated aliases. Idempotent."""
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]


def patch_audiosr_librosa() -> None:
    """audiosr 0.0.7 (Dec 2023) calls `pad_center(data, size)` positionally.
    librosa 0.11+ made `size` a keyword-only argument, so this crashes with
    `pad_center() takes 1 positional argument but 2 were given`. Affected
    call sites in this venv:

      - audiosr/utilities/audio/stft.py (direct import + positional call)
      - torchlibrosa/stft.py (HTSAT, used by audiosr's CLAP encoder)

    We monkey-patch librosa.util.pad_center itself to accept the legacy
    positional form. Idempotent.
    """
    try:
        import librosa
        import librosa.util
    except ImportError:
        return  # librosa not installed; nothing to patch
    if getattr(librosa.util.pad_center, "_patched_positional_compat", False):
        return
    _orig = librosa.util.pad_center

    def _compat(data, *args, **kwargs):
        # If a positional `size` was passed (legacy call), promote to keyword.
        if args and "size" not in kwargs:
            kwargs["size"] = args[0]
            args = args[1:]
        return _orig(data, *args, **kwargs)

    _compat._patched_positional_compat = True  # type: ignore[attr-defined]
    librosa.util.pad_center = _compat  # type: ignore[attr-defined]
    # Also bind the symbol wherever it was imported.
    for mod_name in (
        "audiosr.utilities.audio.stft",
        "torchlibrosa.stft",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "pad_center"):
            mod.pad_center = _compat  # type: ignore[attr-defined]


# ffmpeg's `arnndn` filter needs a model file path, not the literal name
# "rnnoise". The standard pre-trained model is the rnnoise-nu std.rnnn,
# published on HuggingFace by niobures. We fetch it once into CACHE_DIR
# and reuse it. ~303 KB.
_ARNNDN_MODEL_URL = (
    "https://huggingface.co/niobures/RNNoise/resolve/main/models/arnndn-models/std.rnnn"
)
_ARNNDN_MODEL_NAME = "std.rnnn"


def ensure_arnndn_model() -> Path:
    """Return the local path to a working ffmpeg arnndn model, downloading
    on first use. Idempotent. Exits if download fails."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / _ARNNDN_MODEL_NAME
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    import urllib.request
    print(f"[lib_audio] downloading arnndn model → {dest}", file=sys.stderr)
    try:
        with urllib.request.urlopen(_ARNNDN_MODEL_URL, timeout=30) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
    except Exception as e:
        sys.exit(f"failed to download {_ARNNDN_MODEL_URL}: {e}")
    if dest.stat().st_size < 100_000:
        sys.exit(f"downloaded model is suspiciously small: {dest.stat().st_size} bytes")
    return dest


def env_summary() -> dict[str, str]:
    """Snapshot of the toolchain so reports are reproducible."""
    cmds = ["ffmpeg", "ffprobe", "sox", "python3"]
    found = {}
    for c in cmds:
        r = subprocess.run(["which", c], capture_output=True, text=True)
        found[c] = r.stdout.strip() if r.returncode == 0 else "MISSING"
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "ffmpeg": found["ffmpeg"],
        "ffprobe": found["ffprobe"],
        "sox": found["sox"],
    }


if __name__ == "__main__":
    # Smoke test: load source, measure, print.
    src = _require_source()
    print(f"source: {src}")
    m = measure(src)
    print(json.dumps(asdict(m), indent=2))
