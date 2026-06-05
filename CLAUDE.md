# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **compare-first evaluation harness** for restoring degraded 1970s tape audio (Srila Prabhupada lectures) on a single M1 MacBook with 16 GB. The user has a 1000+ file corpus; the goal is to pick a restoration approach that wins on the cost-vs-quality tradeoff at scale. **All processing is local — no cloud APIs.**

The full project charter (mission, scoring formula, environment, candidate lineup, scoring heuristics) lives in `AGENTS.md`. Read it before doing anything non-trivial. The specialized skill at `.opencode/skills/audio-restore/SKILL.md` codifies the workflow as a sequence of steps.

## The core rule

**No pipeline is picked up front.** This is an evaluation phase. Every plausible restoration approach is a self-contained script in `tools/candidates/`. The harness runs them all, scores them objectively, and the ranking in `reports/compare.md` is the decision artifact. Do not recommend "X is probably best" without the table.

## Commands

| task | command |
|------|---------|
| first-time setup | `make setup` |
| characterize source (write baseline) | `make inspect` |
| run every candidate + rank | `make compare` |
| see current top-ranked candidate | `make best` |
| clear runs (keep `samples/source/`) | `make clean` |
| run a single candidate | `source .venv/bin/activate && python3 tools/candidates/c03_facebook_denoiser.py` |
| run a subset through compare | `source .venv/bin/activate && python3 tools/99_compare.py c01 c03` |

`make compare` writes `reports/compare.md` (ranked table — the source of truth) and `reports/compare.json` (machine-readable).

## Architecture

```
samples/source/        canonical input, READ-ONLY (sorts before .wav; real file wins)
  └── .gitkeep         placeholder; user drops source here
  └── fixtures/        synthetic_70s_tape.wav (smoke test) + real-60s.wav (fast iteration)

tools/
  lib_audio.py         shared I/O, AudioMetrics dataclass, ffmpeg/ffprobe wrappers,
                       band_energy_db, lufs_via_ffmpeg, arnndn model downloader,
                       np.complex patch for old libs
  00_inspect.py        reads source → stages/00--inspect/{report.json, inspect.md, spectrogram.png}
  99_compare.py        discovers c*.py in candidates/, runs each, scores, ranks
  candidates/c*.py     one self-contained restoration approach per file; auto-discovered

stages/<id>--<name>/   per-candidate output.wav + report.json (gitignored)
reports/               compare.md (decision artifact) + compare.json (gitignored)
docs/REFERENCES.md     research citations behind every choice
.opencode/skills/      specialized skill that orchestrates the workflow
```

**Convention that holds everywhere:** read from `samples/source/`, write `output.wav` + `report.json` to `stages/<id>--<name>/`, never modify the source. Output format is 48 kHz / 24-bit / mono PCM (`L.TARGET_SR`, `L.TARGET_SUBTYPE`). On success, exit 0; on failure, exit non-zero and the row in `compare.md` gets marked ❌.

## Scoring (in `tools/99_compare.py`)

```
score = 2·hiss_delta
      - max(0, +speech_delta)   # removed speech is bad
      - 0.5·max(0, -LRA_delta)  # crushed dynamics is bad
      - max(0, RTF - 2)         # slower than 2x realtime is bad
```

`hiss_delta` = dB removed from 5–12 kHz band (positive = good). `speech_delta` = dB removed from 300 Hz–3.4 kHz (positive = bad). `LRA_delta` = change in EBU R128 loudness range (negative = bad). `RTF` = runtime ÷ audio duration. The table also has a `hours/1k files` column = `duration_min · RTF · 1000 / 60` — the scale decision axis.

## Adding a candidate

Create `tools/candidates/c<NN>_<name>.py` following the existing template (see `c01_classical_ffmpeg.py` for the simplest example, or `c03_facebook_denoiser.py` for the ML pattern). The harness auto-discovers every `c*.py` — no other file needs to change. After `make compare`, the new row appears in `reports/compare.md`.

If the candidate depends on a package that may not be installed, gate it with a `have_<pkg>()` import check and call `L.write_report(_skipped(out_dir), ...)` so the compare table shows a clean `⏭` row instead of a -200 dB floor (see `c03_facebook_denoiser.py` for the pattern).

Some pinned third-party ML packages (`resemble_enhance`, `audiomaister` via `torchlibrosa`) still reference `np.complex`, which was removed in NumPy 1.20+. Call `L.patch_numpy_complex()` before importing them. The harness already downgraded NumPy to 1.26.4 in `.venv` for compatibility.

## Environment notes (verified on this M1)

- Use `uv` (`/opt/homebrew/bin/uv`), not the system `pip`. The system `python3` is Xcode 3.9 — do not use it. Use `/opt/homebrew/bin/python3.13` or the venv from `make setup`.
- `denoiser` runs on CPU, not MPS — Demucs uses a 1-D conv config that MPS doesn't implement for long clips. Unified memory keeps the speed penalty moderate.
- `ffmpeg` 8.0.1's `arnndn` filter needs a model file path, not a name. `L.ensure_arnndn_model()` fetches the standard `std.rnnn` (~303 KB from HuggingFace) into `.cache/models/` once and reuses it.

## Out of scope (per AGENTS.md)

Cloud APIs, paid local models, Dolby B / dbx NR decode, training/fine-tuning, stem separation, mobile/web frontend. The project is a CLI harness.

## When the user picks a winner

Move the chosen candidate's logic out of `candidates/` into `tools/NN_<verb>.py` as a proper stage. Keep the candidates around as evidence. For the 1000-file scale-out, add `tools/batch_restore.py` that takes a directory of inputs and produces restored outputs. AGENTS.md gets a "Production pipeline" section naming the chosen approach.
