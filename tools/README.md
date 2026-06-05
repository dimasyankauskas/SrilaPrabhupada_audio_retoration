# tools/

Each script is one stage. They share helpers in `lib_audio.py`.

| script | purpose |
|--------|---------|
| `00_inspect.py` | read the source, measure it, write `stages/00--inspect/` |
| `candidates/c01_classical_ffmpeg.py` | ffmpeg arnndn+anlmdn+adeclick chain |
| `candidates/c02_rnnoise_dedicated.py` | pure RNNoise (fastest ML-ish baseline) |
| `candidates/c03_facebook_denoiser.py` | Meta's speech denoiser (ML, 135 MB) |
| `candidates/c04_sox_classical.py` | sox highpass+notch+noisered+compand |
| `candidates/c05_hybrid_classical_ml.py` | ffmpeg prep → denoiser → loudnorm |
| `99_compare.py` | run all candidates, score, rank → `reports/compare.md` |
| `lib_audio.py` | shared I/O, metrics, EBU loudness, band energies |

## Conventions

- All scripts read from `samples/source/`, write to `stages/<id>--<name>/`.
- Source is NEVER modified.
- Output format: 48 kHz / 24-bit / mono PCM.
- Every run writes a `report.json` sidecar with objective metrics.
- Exit code 0 = success. Non-zero = run failed; compare tool will mark the row ❌.
