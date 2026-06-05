# Candidates — what we tried and why

Each candidate is a self-contained script in `tools/candidates/`. They all
read from `samples/source/`, write to `stages/<id>--<name>/output.wav`,
and emit a `report.json` with objective metrics.

The harness `tools/99_compare.py` runs them all and produces
`reports/compare.md` with a ranked table. **The decision is made from
that table, not from this document.**

| ID | name | class | deps | M1 16GB? |
|----|------|-------|------|----------|
| c01 | classical_ffmpeg | ffmpeg filter chain (arnndn+anlmdn+adeclick) | ffmpeg only | yes |
| c02 | rnnoise_dedicated | pure RNNoise | ffmpeg (arnndn) | yes |
| c03 | facebook_denoiser | Demucs-based ML (135 MB) | ffmpeg + denoiser pkg | yes (CPU or MPS) |
| c04 | sox_classical | sox highpass/notch/noisered | ffmpeg + sox | yes |
| c05 | hybrid_classical_ml | ffmpeg prep + denoiser | ffmpeg + denoiser pkg | yes |

## How to add a new candidate

1. Create `tools/candidates/c06_<your_name>.py`.
2. Follow the existing template: import `lib_audio`, measure the source,
   write output to `stages/c06--<name>/output.wav`, call
   `L.measure()` and `L.write_report()`.
3. Run `make compare` — the new row appears in `reports/compare.md`.

The harness auto-discovers every `c*.py` file in `candidates/`.
