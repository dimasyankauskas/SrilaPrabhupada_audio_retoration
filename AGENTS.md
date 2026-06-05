# AGENTS.md — audio_restore

Project-local guidance for `/Users/atma/dev/OpenCode/projects/audio_restore`.

## Mission

Restore degraded 1970s tape audio (Hare Krishna lecture recordings,
primarily Srila Prabhupada) to high-quality modern audio, **for a
1000+ file corpus**, on a **non-profit budget**. The user is on an
M1 MacBook with 16 GB unified memory. **All processing must run
locally. No cloud APIs.**

The "best candidate" decision is a **cost-quality tradeoff at scale**:
compute time per 1000 files is the cost axis (dollars = $0, time =
real). The candidate that removes the most hiss with the least
speech damage, in the fewest M1-hours, wins.

## Core Rule: Compare Before Committing

**No pipeline is selected up front.** This is an evaluation phase.
Every plausible restoration approach is implemented as a candidate
script in `tools/candidates/`, every candidate runs on the same
source, and `reports/compare.md` is the decision artifact.

The agent must:

1. Read the source with `make inspect`.
2. Enumerate every candidate in `tools/candidates/`. **Read the
   directory, do not guess.**
3. Run them all with `make compare`.
4. Read `reports/compare.md` and recommend the top candidate by
   score (hiss removed, speech preserved, RTF) — *then* listen to
   confirm.
5. Wait for the user to pick. Implementation of a final pipeline
   happens *after* the pick.

Skipping the table and recommending "facebook/denoiser is probably
best" is a defect, not a shortcut.

## Canonical Sample

- File: `760706AD-WASHINGTON DC.MP3` — Srila Prabhupada lecture,
  Washington DC temple, 6 July 1976. 56 minutes, 96 kbps MP3,
  39 MB.
- Source: Google Drive —
  `https://drive.google.com/file/d/1r5ouCgv9wQ-DLtC57yU7Fx7OMxYL1ogy/view`
- The user downloads it; the agent does not.
- Place it in `samples/source/`. That directory is read-only to all
  scripts; never write there.
- Filename convention used by the user: `YYMMDD<initials>-<location>.<ext>`
  (Bhaktivedanta Archives style).
- For fast iteration a 60s WAV slice lives in
  `samples/fixtures/real-60s.wav` — copy it into `samples/source/`
  as a `.wav` to smoke-test the pipeline without the full 56-min
  file. The real MP3 sorts before `.wav` alphabetically and is
  always picked if present.

## Environment (verified on this machine, June 2026)

- macOS (darwin), M1-class Apple Silicon, 16 GB unified memory.
- On PATH: `ffmpeg` 8.0.1, `ffprobe`, `sox`, `python3.13` (Homebrew).
- Package manager: `uv` 0.10.9 (`/opt/homebrew/bin/uv`).
- `python3` defaults to Xcode 3.9. **Do not use it.** Use
  `/opt/homebrew/bin/python3.13` or the venv from `make setup`.
- The system `pip` is 21.2.4 (Xcode). Do not use it. Use
  `uv pip install --python .venv/bin/python <pkg>`.
- Installed in `.venv`: `denoiser`, `resemble-enhance` (from
  GitHub), `torch` 2.12 + `torchaudio` 2.11 + `torchcodec` 0.13
  (MPS supported). `numpy` was downgraded to 1.26.4 by
  `deepfilternet`; that install was abandoned because
  `torchaudio 2.11` removed `torchaudio.backend.common.AudioMetaData`
  which `deepfilternet 0.5.6` requires.

## Quick Reference

| task | command |
|------|---------|
| first-time setup | `make setup` |
| characterize source | `make inspect` |
| run all candidates | `make compare` |
| see current ranking | `make best` |
| clean runs (keep sample) | `make clean` |

`make compare` writes `reports/compare.md` (ranked table with hours
per 1000 files column) and `reports/compare.json` (machine-readable).
**The markdown table is the source of truth for "which candidate is
best."**

## Scoring (in `tools/99_compare.py`)

```
score = 2·hiss_delta
      - max(0, +speech_delta)   # removed speech is bad
      - 0.5·max(0, -LRA_delta)  # crushed dynamics is bad
      - max(0, RTF - 2)         # slower than 2x realtime is bad
```

Where:
- `hiss_delta` = dB removed from the 5–12 kHz band (positive = good)
- `speech_delta` = dB removed from 300 Hz–3.4 kHz (positive = bad)
- `LRA_delta` = change in EBU R128 loudness range (negative = bad)
- `RTF` = runtime ÷ audio duration

Plus a **hours per 1000 files** column for the scale decision:
`hours_per_1k = duration_min · RTF · 1000 / 60`. The current corpus
is ~56 min per file; 1000 files = 56,000 min.

## Score > 0 = removed more hiss than speech.
## Score > 30 = clearly the best on this metric.
## Hours/1k < 10h = viable on a single M1 over a long weekend.

## Candidate Lineup (current)

| ID  | approach                              | cost     | quality |
|-----|---------------------------------------|----------|---------|
| c01 | ffmpeg classical chain + loudnorm     | cheapest | decent  |
| c02 | ffmpeg arnndn (RNNoise) + loudnorm    | cheapest | weak    |
| c03 | facebook/denoiser (MPS) + loudnorm    | cheap    | strong  |
| c04 | sox compand + gain (no denoise)       | cheapest | bad     |
| c05 | ffmpeg prep + denoiser + loudnorm     | cheap    | mid     |
| c07 | resemble-enhance denoise + loudnorm   | medium   | mid     |
| c08 | resemble-enhance full enhance         | slowest  | varies  |

Notes on the ML models:
- **facebook/denoiser** — Demucs-based, 33 M params, 16 kHz, MIT.
  The model was trained on DNS Challenge data; the only candidate
  with a paper + standard benchmark. 134 MB download.
- **resemble-enhance** — 0.5 B-param latent diffusion pipeline
  (denoiser + conditional flow matching vocoder). MIT. 680 MB
  download. `denoise-only` is ~RTF 0.4 on M1; full `enhance` is
  ~RTF 2-3 on M1. Note: full `enhance` bandwidth-extends the
  output, which raises 5–12 kHz energy even when hiss is reduced —
  the band-energy metric can penalize it unfairly.
- **arnndn (RNNoise in ffmpeg)** — small RNN, 16 kHz, fast
  (RTF < 0.05). Model weights from niobures/RNNoise on HuggingFace
  (303 KB), auto-downloaded by `L.ensure_arnndn_model()`.

## Adding a Candidate

Write `tools/candidates/c<NN>_<name>.py` following the existing
template. The harness auto-discovers it. No other file needs to
change. After `make compare`, the new row appears in
`reports/compare.md`.

## Out of Scope

- **Cloud APIs** (Resemble cloud, Adobe Enhance, AudioSR web,
  ElevenLabs). Explicit user decision: local-only because non-profit
  budget.
- **Paid local models** (Resemble Pro, Adobe local). Same reason.
- Dolby B / dbx NR decode of the source tape. Without knowing the
  deck, the decode is guesswork; we apply gentle de-essing instead
  and leave the question for a later phase.
- Training or fine-tuning any model. Pre-trained weights only.
- Stem separation. The sample is single-voice lecture; Demucs /
  HTDemucs would be wasted compute.
- Mobile / web frontend. The project is a CLI harness.

## File Map

```
audio_restore/
├── AGENTS.md                       this file
├── opencode.json                   registers the .opencode/skills/ dir
├── .opencode/skills/
│   └── audio-restore/SKILL.md      the specialized skill
├── Makefile                        setup / inspect / compare / best / clean
├── requirements.txt                Python deps
├── samples/
│   ├── source/                     the source MP3 (read-only)
│   └── fixtures/                   dev fixtures (synthetic + 60s slice)
├── tools/
│   ├── lib_audio.py                shared I/O + metrics + arnndn model
│   ├── 00_inspect.py               characterize the source
│   ├── 99_compare.py               run all candidates, score, rank
│   └── candidates/                 one script per approach
│       ├── c01_classical_ffmpeg.py
│       ├── c02_rnnoise_dedicated.py
│       ├── c03_facebook_denoiser.py
│       ├── c04_sox_classical.py
│       ├── c05_hybrid_classical_ml.py
│       ├── c07_resemble_denoise.py
│       └── c08_resemble_enhance.py
├── stages/<id>--<name>/            per-stage outputs and report.json
├── reports/                        compare.md, compare.json
└── docs/
    └── REFERENCES.md               research citations
```

## Done Means

For each candidate run:
- `output.wav` exists at 48 kHz / 24-bit / mono PCM.
- `report.json` exists with all metric fields populated (LUFS, true
  peak, band energies, runtime).

For the project at this phase:
- `reports/compare.md` exists with a ranked table, including the
  `hours/1k files` column.
- The top candidate has been *listened to*, not just scored.
- The user has been told the recommendation with the cost-quality
  trade-off spelled out.

## When the User Picks a Winner

Update this file to add a "Production pipeline" section that names
the chosen approach. Keep `tools/candidates/` intact — they're the
evidence that the chosen approach is the best. Move the chosen
script's logic into `tools/NN_<verb>.py` (proper stage), not a
candidate. For the 1000-file scale-out, also write
`tools/batch_restore.py` (or similar) that takes a directory of
input files and produces a directory of restored outputs in
parallel (or sequentially) using the chosen approach.
