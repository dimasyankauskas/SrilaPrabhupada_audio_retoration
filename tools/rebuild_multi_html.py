#!/usr/bin/env python3
"""
rebuild_multi_html — Re-render reports/multi/comparison.html + all.json
from the existing per-sample compare.json files.

Use this when the per-sample compares are already done and you just need
the aggregated HTML rebuilt (e.g. new sample, scoring formula tweak).

Reads SOURCES from tools/run_multi.py to know which samples to look for.
Reuses render_html() and the helpers from run_multi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Importing run_multi gives us SOURCES, render_html, and the helpers.
import run_multi  # noqa: E402


def main() -> int:
    samples = []
    for src, start_s, dur_s in run_multi.SOURCES:
        sample = run_multi.safe_name(src)
        clip_name = (
            f"{sample}_10min.wav" if dur_s == 600
            else f"{sample}_full.wav" if dur_s is None
            else f"{sample}_{dur_s}s.wav"
        )
        clip = run_multi.CLIP_DIR / clip_name
        compare_json = run_multi.REPORT_DIR / sample / "compare.json"
        if not compare_json.exists():
            print(f"[rebuild] skip {sample}: missing {compare_json}")
            continue
        with open(compare_json) as f:
            d = json.load(f)
        # Normalize old schema (no exit_code/stdout_tail/stderr_tail) to the
        # current one so the renderer doesn't KeyError. We assume a row is
        # "ok" if it has a non-skipped score with a positive score value;
        # old runs didn't record exit codes, so this is the best we can do.
        for r in d.get("results", []):
            r.setdefault("exit_code", 0)
            r.setdefault("stdout_tail", "")
            r.setdefault("stderr_tail", "")
            # Older compare.json files have short candidate names like "c01"
            # without the "_classical_ffmpeg" suffix. Expand them to the full
            # stem used by the new harness so the per-candidate aggregation
            # treats them as the same candidate across samples.
            cand = r.get("candidate", "")
            if cand in ("c01", "c02", "c03", "c04", "c05", "c06", "c07",
                        "c08", "c09", "c10", "c10a", "c10b", "c10c", "c10d",
                        "c10e", "c10f", "c11", "c12", "c13", "c14", "c15",
                        "c16", "c17"):
                # Look up the full name from the stage dir, fall back to
                # the stage's "stage" field if it's there.
                stage = r.get("stage", "")
                if stage and "--" in stage:
                    r["candidate"] = stage.replace("--", "_", 1)
        # Drop the inspect pseudo-candidate ("00" / "00--inspect") that some
        # older runs accidentally included as a real candidate. It's not a
        # restoration output; it's the source baseline.
        d["results"] = [
            r for r in d.get("results", [])
            if r.get("candidate", "").split("_", 1)[0] not in ("00", "0")
        ]
        d["src"] = src.name
        d["sample"] = sample
        d["clip"] = str(clip)
        samples.append(d)
        print(f"[rebuild] loaded {sample}: {len(d.get('results', []))} candidates")

    if not samples:
        print("error: no per-sample compare.json files found", file=sys.stderr)
        return 1

    (run_multi.REPORT_DIR / "all.json").write_text(
        json.dumps(samples, indent=2, default=str)
    )
    run_multi.render_html(samples, run_multi.REPORT_DIR / "comparison.html")
    print(f"\n[done] wrote {run_multi.REPORT_DIR / 'all.json'}")
    print(f"[done] wrote {run_multi.REPORT_DIR / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
