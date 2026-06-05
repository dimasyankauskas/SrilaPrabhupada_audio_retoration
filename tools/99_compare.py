#!/usr/bin/env python3
"""
99_compare — Run every candidate and produce a ranked comparison.

Usage:
  python3 tools/99_compare.py                  # all candidates
  python3 tools/99_compare.py c01 c03 c05    # specific candidates

Output:
  reports/compare.md      human-readable ranking + decision matrix
  reports/compare.json    machine-readable scores (for the skill to load)
  reports/spectrograms/   side-by-side PNGs (if matplotlib installed)

Ranking heuristic (initial, refine as you learn):
  A good restoration:
    - reduces hiss-band energy by >= 3 dB
    - keeps speech-band energy loss < 3 dB
    - does not reduce dynamic range (LRA) by more than 2 LU
    - runs in < 2x realtime on M1 16GB
  Among candidates meeting all four, pick the one with the largest
  hiss reduction AND the smallest speech loss.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_audio as L

CANDIDATE_DIR = Path(__file__).resolve().parent / "candidates"


def discover_candidates(requested: list[str] | None = None) -> list[Path]:
    """Find candidate scripts. Either the requested set or every c*.py."""
    all_cands = sorted(CANDIDATE_DIR.glob("c*.py"))
    if not requested:
        return all_cands
    out = []
    for r in requested:
        matches = [c for c in all_cands if c.stem == r or c.stem.startswith(r + "_")]
        if not matches:
            print(f"[99_compare] no candidate matches {r!r}", file=sys.stderr)
            continue
        out.extend(matches)
    return out


def run_candidate(script: Path) -> dict[str, Any]:
    """Run one candidate and collect its report + exit code."""
    import subprocess
    print(f"[99_compare] running {script.name}")
    r = subprocess.run(
        ["python3", str(script)],
        capture_output=True, text=True, cwd=L.REPO_ROOT,
    )
    stage_id = script.stem.split("_", 1)[0]  # "c01"
    stage_name = script.stem.split("_", 1)[1] if "_" in script.stem else script.stem
    report_path = L.stage_dir(stage_id, stage_name) / "report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
    return {
        "candidate": script.stem,
        "exit_code": r.returncode,
        "stdout_tail": r.stdout[-500:],
        "stderr_tail": r.stderr[-500:],
        "report": report,
    }


def baseline() -> dict[str, Any] | None:
    """Load inspect report for the source. None if inspect hasn't run."""
    p = L.stage_dir("00", "inspect") / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def score(report: dict[str, Any], base: dict[str, Any] | None) -> dict[str, float]:
    """Compute deltas vs source, plus runtime + loudness penalties.

    Deltas are positive when the candidate *removed* energy. So:
      hiss_delta > 0  → hiss was reduced (good)
      speech_delta > 0 → speech was removed (bad)
      lra_delta < 0   → dynamics were crushed (bad)

    The lufs_drift term penalizes output that lands far from the
    -16 LUFS podcast target — the most common failure mode is the
    ML model outputting at -30 to -40 LUFS, which the metric
    formula would otherwise reward (silence scores well on hiss Δ).
    Output within ±3 LU of -16 is free; beyond that, 0.3 pts per LU
    of drift.

    Bandwidth-extension (hf_extension_db_delta) rewards candidates
    that add 12-18 kHz content — the missing "studio" content that
    1960s tape physically couldn't record. A 30 dB jump (tape →
    modern podcast) earns +30 to the score. Without this term, BWE
    candidates like VoiceFixer look like failures because the TFGAN
    vocoder also adds 5-12 kHz hiss-band energy.

    Centroid pull gently biases toward a 5 kHz "studio" centroid,
    penalizing outputs that are too dull (under 2.5 kHz) or
    artifact-bright (over 7.5 kHz). Weight 0.1 — it's a tiebreaker,
    not a dominant signal.

    Skipped candidates (no real output) get a sentinel score and skipped=True.
    """
    if not base or not report:
        return {"hiss_delta_db": 0.0, "speech_delta_db": 0.0, "lra_delta_lu": 0.0,
                "lufs_drift_lu": 0.0, "rtf": 0.0, "hf_delta_db": 0.0,
                "centroid_delta_hz": 0.0, "score": -1e9, "skipped": True}
    extras = (report.get("extras") or {})
    if str(extras.get("status", "")).startswith("skipped"):
        return {"hiss_delta_db": 0.0, "speech_delta_db": 0.0, "lra_delta_lu": 0.0,
                "lufs_drift_lu": 0.0, "rtf": 0.0, "hf_delta_db": 0.0,
                "centroid_delta_hz": 0.0, "score": -1e9, "skipped": True}
    # Hiss delta — measured as the hiss-to-speech ratio. This is the
    # user-perceived "noise floor" relative to the speech content, and
    # it's gain-invariant (so we don't penalize candidates just because
    # they were loudnorm'd to -16 LUFS while the baseline is at -22).
    # Positive hiss_delta = output has more hiss relative to speech
    # than the source did. We reward NEGATIVE hiss_delta (hiss reduced
    # relative to speech).
    base_hiss = base.get("hiss_band_energy_db", 0.0) or 0.0
    base_speech = base.get("speech_band_energy_db", 0.0) or 0.0
    out_hiss = report.get("hiss_band_energy_db", 0.0) or 0.0
    out_speech = report.get("speech_band_energy_db", 0.0) or 0.0
    # hiss/speech in dB = hiss_db - speech_db. Smaller (more negative)
    # is better — the hiss is below the speech. We want the candidate
    # to have a SMALLER value than the source.
    base_ratio = base_hiss - base_speech
    out_ratio = out_hiss - out_speech
    hiss_delta = base_ratio - out_ratio  # positive = hiss reduced relative to speech
    speech_delta = (base.get("speech_band_energy_db", 0.0) or 0.0) - (report.get("speech_band_energy_db", 0.0) or 0.0)
    lra_delta = (report.get("dynamic_range_lu", 0.0) or 0.0) - (base.get("dynamic_range_lu", 0.0) or 0.0)
    rt_s = report.get("runtime_s", 0.0) or 0.0
    duration = base.get("duration_s", 1.0) or 1.0
    rtf = rt_s / duration
    # Loudness drift: how far the output is from -16 LUFS, minus a
    # 3-LU tolerance band. None means loudnorm didn't report (e.g.
    # silent input) — don't penalize, since we don't know.
    target_lufs = -16.0
    tolerance_lu = 3.0
    lufs_out = report.get("lufs")
    if lufs_out is None:
        lufs_drift = 0.0
    else:
        lufs_drift = max(0.0, abs(lufs_out - target_lufs) - tolerance_lu)
    # Bandwidth-extension delta: how much 12-18 kHz energy the output has
    # relative to the source. Positive = output has more HF (BWE did work).
    # The reward is CAPPED at +20 dB — beyond that, the model is
    # hallucinating HF content (c08 Resemble enhance, c13 VoiceFixer
    # TFGAN vocoder both produce +50 to +90 dB HF deltas, which is
    # unphysical for a 0.86-peak signal and sounds metallic).
    # A clean BWE target is to bring HF from -42 dB (tape) to -15 dB
    # (modern podcast), a +27 dB delta — capping at +20 keeps the
    # reward meaningful without rewarding over-synthesis.
    base_hf = base.get("hf_extension_db", -200.0) or -200.0
    out_hf = report.get("hf_extension_db", -200.0) or -200.0
    hf_delta = out_hf - base_hf
    # Also penalize absolute HF that's too high (artifacts).
    # -15 dB is the "studio" target; above 0 dB is overcooked.
    hf_target_db = -15.0
    hf_overshoot_penalty = max(0.0, (out_hf - hf_target_db) - 10.0) * 0.2
    # Spectral centroid shift: a positive shift means brighter output.
    # Pull toward 5 kHz target with mild weight (0.1).
    base_centroid = base.get("spectral_centroid_hz", 0.0) or 0.0
    out_centroid = report.get("spectral_centroid_hz", 0.0) or 0.0
    centroid_delta = out_centroid - base_centroid
    centroid_penalty = 0.1 * abs(out_centroid - 5000.0) / 1000.0
    # Quality score:
    #   - hiss reduction (weight 2.0) — the dominant signal
    #   - HF extension (weight 1.0, capped at +20 dB) — rewards BWE
    #   - HF overshoot penalty (weight 0.2) — penalizes over-synthesis
    #   - speech loss (weight 1.0) — bad
    #   - LRA crush (weight 0.5) — bad
    #   - centroid pull (weight 0.1) — gentle bias toward studio
    #   - runtime over 2x (weight 1.0) — bad
    #   - loudness drift (weight 0.3) — bad
    hf_reward = min(hf_delta, 20.0)
    score = (
        2.0 * hiss_delta
        + 1.0 * hf_reward
        - hf_overshoot_penalty
        - 1.0 * max(0.0, speech_delta)   # removed speech is bad
        - 0.5 * max(0.0, -lra_delta)     # dynamic range crush is bad
        - centroid_penalty               # gentle pull to 5 kHz centroid
        - 1.0 * max(0.0, rtf - 2.0)      # over 2x realtime is bad
        - 0.3 * lufs_drift               # output too quiet OR too loud is bad
    )
    return {
        "hiss_delta_db": round(hiss_delta, 2),
        "speech_delta_db": round(speech_delta, 2),
        "lra_delta_lu": round(lra_delta, 2),
        "lufs_drift_lu": round(lufs_drift, 2),
        "rtf": round(rtf, 3),
        "hf_delta_db": round(hf_delta, 2),
        "hf_reward_db": round(hf_reward, 2),
        "hf_overshoot_db": round(max(0.0, (out_hf - hf_target_db) - 10.0), 2),
        "centroid_delta_hz": round(centroid_delta, 1),
        "score": round(score, 3),
        "skipped": False,
    }


def main(argv: list[str]) -> int:
    # When a sample is active (multi-sample mode), write reports under
    # reports/multi/<sample>/ so concurrent samples don't clobber each
    # other. The default writes to reports/compare.{md,json}.
    sample = L._current_sample()
    if sample:
        out_dir = L.REPORTS_DIR / "multi" / sample
    else:
        out_dir = L.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    base = baseline()
    if base is None:
        print("[99_compare] no baseline — run `python3 tools/00_inspect.py` first")
        return 1
    requested = argv[1:] if len(argv) > 1 else None
    cands = discover_candidates(requested)
    if not cands:
        print("[99_compare] no candidates found")
        return 1

    results = []
    for script in cands:
        r = run_candidate(script)
        s = score(r["report"], base)
        r["score"] = s
        results.append(r)

    # Rank by score desc, but keep skipped candidates at the bottom regardless.
    results.sort(key=lambda r: (r["score"].get("skipped", False), -r["score"]["score"]))

    # JSON dump.
    (out_dir / "compare.json").write_text(json.dumps({
        "baseline": base,
        "results": results,
    }, indent=2, default=str))

    # Markdown table.
    md = ["# Compare — candidate ranking\n"]
    md.append(f"Baseline: `{Path(base['path']).name}`  |  duration {base['duration_s']:.1f}s\n\n")
    md.append("| candidate | hiss Δ (dB) | HF Δ (dB) | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | hours/1k files | status |\n")
    md.append("|-----------|-------------|-----------|---------------|------------|----------|-----|-------|----------------|--------|\n")
    # Hours to process 1000 files of this duration. RTF is already duration-multiple,
    # so per-file wall-clock = duration_s * RTF; 1000 files / 3600 s = hours.
    base_dur_min = base["duration_s"] / 60.0
    for r in results:
        s = r["score"]
        if s.get("skipped"):
            status_extra = r.get("report", {}).get("extras", {}).get("status", "skipped")
            md.append(f"| {r['candidate']} | — | — | — | — | — | — | — | — | — | ⏭ {status_extra} |\n")
            continue
        status = "✅" if r["exit_code"] == 0 else "❌"
        hours_per_1k = base_dur_min * s["rtf"] * 1000 / 60.0
        # LUFS out — show the measured value, or — if loudnorm didn't report
        rpt = r.get("report", {})
        lufs = rpt.get("lufs")
        lufs_str = f"{lufs:+.1f}" if lufs is not None else "—"
        md.append(f"| {r['candidate']} | {s['hiss_delta_db']:+.1f} | {s.get('hf_delta_db', 0.0):+.1f} | {s['speech_delta_db']:+.1f} | "
                  f"{s['lra_delta_lu']:+.1f} | {lufs_str} | {s['rtf']:.2f} | "
                  f"{s['score']:+.2f} | {hours_per_1k:.0f} h | {status} |\n")
    md.append("\n## Reading the table\n\n")
    md.append("- **hiss Δ** — how much the hiss-band vs speech-band ratio dropped in the candidate. (hiss_db − speech_db) is the gain-invariant hiss floor. Positive = hiss reduced relative to speech.\n")
    md.append("- **HF Δ** — change in 12–18 kHz energy. Positive = bandwidth extension worked (studio-quality HF added). The single biggest gap between 1970s tape and modern recordings.\n")
    md.append("- **speech Δ** — how much energy was removed from 300–3.4 kHz. Positive = removed speech (bad).\n")
    md.append("- **LRA Δ** — change in loudness range. Negative = dynamics got crushed (bad).\n")
    md.append("- **LUFS out** — measured output loudness. The podcast target is -16. Anything more than 3 LU away from -16 is penalized in the score (see formula).\n")
    md.append("- **RTF** — runtime ÷ audio duration. <1.0 is faster than realtime.\n")
    md.append("- **score** — composite: `2·hiss_ratio_Δ + min(HF_Δ, +20) − max(0,+speech_Δ) − 0.5·max(0,−LRA_Δ) − 0.1·|centroid−5000|/1000 − max(0,RTF−2) − 0.3·max(0,|LUFS+16|−3)`. Higher is better.\n")
    md.append("- hiss_ratio_Δ = (hiss_db − speech_db) in source minus (hiss_db − speech_db) in candidate. Positive = hiss reduced relative to speech (the gain-invariant measure).\n")
    md.append("\n## Recommendation\n\n")
    finished = [r for r in results if not r["score"].get("skipped") and r["exit_code"] == 0]
    if finished:
        winner = finished[0]
        md.append(f"**Best candidate so far: `{winner['candidate']}`** with score {winner['score']['score']:+.2f}.\n\n")
    else:
        md.append("No candidate produced a usable report yet. Check that `tools/00_inspect.py` and at least one candidate ran successfully.\n\n")
    (out_dir / "compare.md").write_text("".join(md))
    print(f"[99_compare] wrote {out_dir / 'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
