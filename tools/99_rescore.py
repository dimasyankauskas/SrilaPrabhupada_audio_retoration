#!/usr/bin/env python3
"""99_rescore — re-run the scoring formula on existing report.json files.

Useful when the scoring formula changes but the candidates haven't. Reads
report.json + baseline, calls the same score() function 99_compare uses,
writes a new compare.{md,json} without re-running any audio processing.

Usage:
  python3 tools/99_rescore.py            # rescore current sample (or single)
  python3 tools/99_rescore.py > out.md   # capture output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_audio as L
import importlib.util

# Load 99_compare's score() function so we always use the same logic.
spec = importlib.util.spec_from_file_location("compare_module", L.REPO_ROOT / "tools" / "99_compare.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> int:
    sample = L._current_sample()
    out_dir = L.REPORTS_DIR / "multi" / sample if sample else L.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_p = L.stage_dir("00", "inspect") / "report.json"
    if not base_p.exists():
        print(f"[99_rescore] no baseline: {base_p}", file=sys.stderr)
        return 1
    base = json.loads(base_p.read_text())

    results = []
    sample_dir = L.STAGES_DIR / sample if sample else L.STAGES_DIR
    for stage_dir in sorted(sample_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        report_p = stage_dir / "report.json"
        if not report_p.exists():
            continue
        report = json.loads(report_p.read_text())
        score = mod.score(report, base)
        results.append({
            "candidate": stage_dir.name.split("--", 1)[0],
            "stage": stage_dir.name,
            "score": score,
            "report": report,
        })

    # Sort: skipped at bottom, then by score desc.
    results.sort(key=lambda r: (r["score"].get("skipped", False), -r["score"]["score"]))

    (out_dir / "compare.json").write_text(json.dumps({
        "baseline": base,
        "results": results,
    }, indent=2, default=str))

    # Markdown table.
    md = [f"# Compare — candidate ranking (rescore only)\n"]
    md.append(f"Baseline: `{Path(base['path']).name}`  |  duration {base['duration_s']:.1f}s\n\n")
    md.append("| candidate | hiss Δ (dB) | HF Δ (dB) | HF reward | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | status |\n")
    md.append("|-----------|-------------|-----------|-----------|---------------|------------|----------|-----|-------|--------|\n")
    for r in results:
        s = r["score"]
        if s.get("skipped"):
            status = r.get("report", {}).get("extras", {}).get("status", "skipped")
            md.append(f"| {r['candidate']} | — | — | — | — | — | — | — | — | ⏭ {status} |\n")
            continue
        rep = r["report"]
        lufs = rep.get("lufs")
        lufs_s = f"{lufs:+.1f}" if lufs is not None else "—"
        md.append(f"| {r['candidate']} | {s['hiss_delta_db']:+.1f} | {s.get('hf_delta_db', 0.0):+.1f} | "
                  f"{s.get('hf_reward_db', 0.0):+.1f} | {s['speech_delta_db']:+.1f} | "
                  f"{s['lra_delta_lu']:+.1f} | {lufs_s} | {s['rtf']:.2f} | {s['score']:+.2f} | ✅ |\n")
    md.append("\n## Scoring formula\n\n")
    md.append("`score = 2·hiss_Δ + min(HF_Δ, +20) − max(0, +speech_Δ) − 0.5·max(0, −LRA_Δ) "
              "− 0.1·|centroid−5000|/1000 − max(0, RTF−2) − 0.3·max(0, |LUFS+16|−3)`\n\n")
    md.append("HF Δ is capped at +20 dB to prevent over-synthesis reward; "
              "absolute HF > -5 dB is penalized as artifact.\n")
    if results:
        winner = results[0]
        md.append(f"\n**Best: `{winner['candidate']}` with score {winner['score']['score']:+.2f}**\n")

    (out_dir / "compare.md").write_text("".join(md))
    print(f"[99_rescore] wrote {out_dir / 'compare.md'} ({len(results)} candidates)", file=sys.stderr)
    print("".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
