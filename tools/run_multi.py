#!/usr/bin/env python3
"""
run_multi — Run the candidate harness across multiple sources, side by side.

For each source in SOURCES:
  1. Make a 10-minute clip (0:00–10:00) at 48 kHz / 24-bit / mono PCM.
     Clips land in /tmp/audio_restore_clips/ so samples/source/ is not touched.
  2. Run tools/00_inspect.py and tools/99_compare.py as subprocesses with
     AUDIO_RESTORE_SOURCE and AUDIO_RESTORE_SAMPLE set. Each candidate writes
     its output to stages/<sample>/c<N>--<name>/.
  3. Copy reports/multi/<sample>/compare.{json,md} aside (they live there
     during the run; we keep them).

After all sources:
  4. Aggregate into reports/multi/all.json and render reports/multi/comparison.html.

HTML layout (per-sample primary, per-candidate secondary):
  - Header with run date + list of sources.
  - One card per sample: baseline metrics + ranking table.
  - One row per candidate across samples (consistency view).
  - Footer: mean ± std-dev, recommended winner.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import lib_audio as L  # noqa: E402

# Sources to compare. Edit this list to add/remove. Order is preserved in
# the HTML output. Each path is relative to REPO_ROOT unless absolute.
# Each entry: (path, clip_start_seconds, clip_duration_seconds).
# The clip is taken from [start, start+duration]. Set duration to None
# to use the source's full length (Paris 1972 is only 4:53, no room for 5-15).
SOURCES: list[tuple[Path, int, int | None]] = [
    (Path("samples/source/760706AD-WASHINGTON DC.wav"),  0, 600),  # 1976 DC, 0-10min (whole file)
    (Path("samples/source/670322SB-SAN_FRANCISCO.mp3"),  0, 600),  # 1967 SF, 0-10min
    (Path("samples/source/720321SB.BOM.mp3"),          300, 600),  # 1972 BOM, 5-15min slice
    (Path("samples/source/680323MW-SAN_FRANCISCO.mp3"), 300, 600),  # 1968 SF morning walk, 5-15min slice
    (Path("samples/source/720720IV.PAR.mp3"),            0, None),  # 1972 Paris IV, full 4:53
]

CLIP_DIR = Path("/tmp/audio_restore_clips")
REPORT_DIR = REPO_ROOT / "reports" / "multi"


# ---------- helpers ----------

def safe_name(p: Path) -> str:
    """Filesystem-safe stem. '760706AD-WASHINGTON DC' → '760706AD_WASHINGTON_DC'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", p.stem).strip("_")


def make_clip(src: Path, dest: Path, start_s: int, duration_s: int | None) -> None:
    """Decode/clip [start_s, start_s + duration_s] of src → dest at 48k/24/mono PCM.

    Re-runs are skipped if the dest already exists with non-zero size.
    If duration_s is None, the source is decoded to its full length.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[clip] cached: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration_label = f"{duration_s}s" if duration_s else "full length"
    print(f"[clip] ffmpeg → {dest.name}  (start={start_s}s, dur={duration_label})")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-ss", str(start_s),
        "-i", str(src),
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le",
    ]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += [str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg clip failed for {src}: {r.stderr}")


def run_harness(source_clip: Path, sample: str) -> dict:
    """Run inspect + compare on the clip. Returns parsed compare.json."""
    env = os.environ.copy()
    env["AUDIO_RESTORE_SOURCE"] = str(source_clip)
    env["AUDIO_RESTORE_SAMPLE"] = sample
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "tools"))

    for script in ("00_inspect.py", "99_compare.py"):
        cmd = [sys.executable, str(REPO_ROOT / "tools" / script)]
        print(f"[{sample}] running {script}…", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"[{sample}] FAILED {script} (exit {r.returncode}, {dt:.1f}s):", file=sys.stderr)
            print(r.stderr[-1000:], file=sys.stderr)
            raise RuntimeError(f"{script} failed for sample {sample}")
        # Show the last line of stdout for progress.
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"[{sample}] {script} done in {dt:.0f}s — {tail}")

    # compare.json was written to reports/multi/<sample>/compare.json.
    compare_path = REPORT_DIR / sample / "compare.json"
    if not compare_path.exists():
        raise RuntimeError(f"compare.json missing at {compare_path}")
    payload = json.loads(compare_path.read_text())
    payload["_sample"] = sample
    payload["_clip"] = str(source_clip)
    return payload


# ---------- HTML rendering ----------

def fmt(x, spec=".2f", na="—") -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN
        return na
    if isinstance(x, str):
        return escape(x)
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return escape(str(x))


def sign(x, spec=".1f", na="—") -> str:
    if x is None:
        return na
    return f"{x:+{spec}}"


def _rating_widget_html(sample: str, cand: str) -> str:
    """5-star clickable widget. The JS reads/writes localStorage and
    paints the .filled class on each <button> based on the stored value.

    `data-sample` and `data-cand` carry the storage key. The visible state
    (which stars are filled) is reflected via the .filled class on each
    <button>, painted by JS at load and after each click.
    """
    buttons = "".join(
        f"<button type='button' data-n='{n}' aria-label='Rate {n} of 5' "
        f"title='{n}/5'>★</button>"
        for n in range(1, 6)
    )
    return (
        f"<div class='rating' data-sample='{escape(sample, quote=True)}' "
        f"data-cand='{escape(cand, quote=True)}' data-rating='0'>"
        f"{buttons}</div>"
    )


def candidate_table_html(results: list[dict], sample: str | None = None) -> str:
    """Render a ranking table for one sample's results. `sample` is the
    sample key (e.g. '760706AD_WASHINGTON_DC') used as the rating-storage
    key. Skipped rows get no star widget.
    """
    rows = []
    for r in results:
        s = r["score"]
        if s.get("skipped"):
            reason = r.get("report", {}).get("extras", {}).get("status", "skipped")
            rows.append(
                f"<tr class='skipped'><td class='cand'>{escape(r['candidate'])}</td>"
                f"<td colspan='8' class='skip-note'>⏭ {escape(reason)}</td></tr>"
            )
            continue
        status = "✅" if r["exit_code"] == 0 else "❌"
        # The rating cell embeds the sample key. If `sample` is None (legacy
        # callsite), ratings are still clickable but won't survive a reload
        # — keep the widget rendered so the UI is consistent.
        sk = sample or "_unknown"
        rows.append(
            f"<tr>"
            f"<td class='cand'>{escape(r['candidate'])}</td>"
            f"<td class='num'>{sign(s['hiss_delta_db'])}</td>"
            f"<td class='num'>{sign(s['speech_delta_db'])}</td>"
            f"<td class='num'>{sign(s['lra_delta_lu'])}</td>"
            f"<td class='num'>{fmt(s['rtf'], '.2f')}</td>"
            f"<td class='num score'>{sign(s['score'], '.2f')}</td>"
            f"<td class='num'>{fmt(r.get('_hours_per_1k'), '.0f')} h</td>"
            f"<td class='status'>{status}</td>"
            f"<td class='rating-cell'>{_rating_widget_html(sk, r['candidate'])}</td>"
            f"</tr>"
        )
    return (
        "<table class='rank'>"
        "<thead><tr>"
        "<th>candidate</th><th>hiss Δ (dB)</th><th>speech Δ (dB)</th>"
        "<th>LRA Δ (LU)</th><th>RTF</th><th>score</th><th>hours/1k</th><th>status</th>"
        "<th>your rating</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def stage_leaf(cand_stem: str) -> str:
    """Convert candidate script stem ('c03_facebook_denoiser') to the stage
    dir name used on disk ('c03--facebook_denoiser'). The harness builds
    stage dirs as `<id>--<name>` where `<id>` is the part before the first
    underscore and `<name>` is everything after."""
    if "_" in cand_stem:
        idx = cand_stem.index("_")
        return f"{cand_stem[:idx]}--{cand_stem[idx + 1:]}"
    return cand_stem


# Focus candidates that have per-sample 3-min MP3s in assets/audio/, so
# their comparison.html audio cells work on the public GitHub Pages URL.
# Other candidates fall back to the local stages/<sample>/<cand>/output.wav
# path (which works when viewing the HTML locally but 404s on Pages
# because stages/ is gitignored).
FOCUS_CAND_IDS = ("c12", "c13", "c14", "c15", "c16", "c17")


def public_audio_path(sample: str, cand_stem: str) -> str | None:
    """Return the relative path (from reports/multi/comparison.html) of a
    3-min MP3 for `cand_stem` on `sample`, if it has been pre-encoded into
    assets/audio/. Returns None if not available (use stages/ fallback).
    """
    if "_" not in cand_stem:
        return None
    cand_id = cand_stem.split("_", 1)[0]
    if cand_id not in FOCUS_CAND_IDS:
        return None
    mp3 = REPO_ROOT / "assets" / "audio" / f"{sample}_{cand_id}_3min.mp3"
    if not mp3.exists():
        return None
    # assets/audio/<sample>_<cand>_3min.mp3 relative to reports/multi/comparison.html
    return f"../../assets/audio/{sample}_{cand_id}_3min.mp3"


def audio_cell_html(label: str, rel_path: str, tag: str) -> str:
    """One cell of an audio-comparison table.

    Renders a small ▶ button that, on click, expands an inline <audio controls>
    element pointing at rel_path (relative to the HTML, i.e. '../../stages/...').
    `tag` is a unique DOM id anchor for the JS toggle. `label` is shown next
    to the button so the user knows which file they're about to play.
    """
    return (
        f"<td class='audio-cell'>"
        f"<div class='audio-label'>{escape(label)}</div>"
        f"<button class='play-btn' data-target='{escape(tag, quote=True)}' "
        f"aria-label='Play {escape(label)}'>▶ play</button>"
        f"<audio id='{escape(tag)}' class='hidden' controls preload='none' "
        f"src='{escape(rel_path, quote=True)}'></audio>"
        f"</td>"
    )


def per_sample_audio_table_html(s: dict) -> str:
    """One row, one cell per audio file (source + 8 candidates), all for this sample.

    Layout: columns = [source clip, c01, c02, c03, c05, c07, c08, c09].
    Audio paths are relative to reports/multi/comparison.html, so they go up
    two levels to reach stages/<sample>/<cand>/output.wav.
    """
    sample = s["sample"]
    sample_label = s["src"]
    # Build a list of (label, relative path, score) tuples in stable order.
    items: list[tuple[str, str, dict | None]] = []
    # Source clip first.
    items.append((
        f"source ({sample_label})",
        f"../../stages/{sample}/00--inspect/report.json",  # placeholder, swapped below
        None,
    ))
    # 8 candidates, in score-sorted order so the highest-scored is leftmost.
    sorted_results = sorted(
        [r for r in s["results"] if not r["score"].get("skipped")],
        key=lambda r: -r["score"]["score"],
    )
    for r in sorted_results:
        cand = r["candidate"]
        stage_dir = stage_leaf(cand)  # e.g. c03--facebook_denoiser
        # Prefer the public-friendly MP3 path when available; fall back to
        # the local stages/ path for non-focus candidates.
        public = public_audio_path(sample, cand)
        rel = public if public else f"../../stages/{sample}/{stage_dir}/output.wav"
        sc = r["score"]
        items.append((
            f"{cand} (score {sc['score']:+.1f})",
            rel,
            sc,
        ))

    # Replace the source placeholder with the actual clip path. The clip is
    # at /tmp/audio_restore_clips/<sample>_10min.wav (outside the repo) for
    # local viewing. On the public GitHub Pages URL, file:// is blocked, so
    # fall back to the assets/audio/<sample>_3min.mp3 if it exists.
    src_clip_path = s.get("clip", "")
    if src_clip_path and Path(src_clip_path).exists():
        items[0] = (items[0][0], f"file://{src_clip_path}", None)
    else:
        public_src = REPO_ROOT / "assets" / "audio" / f"{sample}_3min.mp3"
        if public_src.exists():
            items[0] = (items[0][0], f"../../assets/audio/{sample}_3min.mp3", None)

    # Build header (compact column labels)
    header_cells = "".join(
        f"<th>{escape(label.split(' (')[0])}</th>" for label, _, _ in items
    )
    audio_cells = "".join(
        audio_cell_html(label, path, f"aud-{sample}-{i}")
        for i, (label, path, _) in enumerate(items)
    )

    return (
        f"<table class='audio-grid' data-sample='{escape(sample)}'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody><tr>{audio_cells}</tr></tbody>"
        "</table>"
    )


def per_candidate_audio_table_html(samples: list[dict]) -> str:
    """Rows = candidates, columns = samples, for cross-sample listening.

    Useful to hear how a single candidate behaves on different tapes. The
    source clip goes in the first row for A/B reference.
    """
    # First row: source clips across samples.
    header_cells = "<th></th>" + "".join(
        f"<th>{escape(s['sample'])}</th>" for s in samples
    )

    def row(label: str, get_path, cand_class: str = "") -> str:
        cells = "".join(get_path(s) for s in samples)
        return f"<tr class='{cand_class}'><td class='cand audio-label'>{escape(label)}</td>{cells}</tr>"

    rows: list[str] = []
    # Source row first.
    def src_path(s: dict) -> str:
        clip = s.get("clip", "")
        if clip and Path(clip).exists():
            path = f"file://{clip}"
        else:
            public = REPO_ROOT / "assets" / "audio" / f"{s['sample']}_3min.mp3"
            path = f"../../assets/audio/{s['sample']}_3min.mp3" if public.exists() else ""
        return audio_cell_html(s["src"], path, f"aud-src-{s['sample']}")
    rows.append(row("source", src_path, "src-row"))

    # One row per candidate, in score-sorted order using the mean from the
    # first sample's ranking (closest to the user's mental model).
    all_cands = sorted(
        {r["candidate"] for s in samples for r in s["results"]},
    )
    # Use the DC sample's ordering as the canonical row order, falling
    # back to the first sample that has the candidate.
    for cand in all_cands:
        for s in samples:
            r = next((r for r in s["results"] if r["candidate"] == cand), None)
            if r and not r["score"].get("skipped"):
                first = s
                first_r = r
                break
        else:
            continue
        sc = first_r["score"]

        def cell_for(s: dict, c: str = cand) -> str:
            # find this candidate's result in this sample
            r = next((r for r in s["results"] if r["candidate"] == c), None)
            if not r or r["score"].get("skipped"):
                return "<td class='audio-cell muted'>—</td>"
            public = public_audio_path(s["sample"], c)
            path = public if public else f"../../stages/{s['sample']}/{stage_leaf(c)}/output.wav"
            label = f"{c} on {s['sample']} (score {r['score']['score']:+.1f})"
            return audio_cell_html(label, path, f"aud-x-{c}-{s['sample']}")

        rows.append(
            f"<tr><td class='cand audio-label'>{escape(cand)} "
            f"<span class='muted'>(DC: {sc['score']:+.1f})</span></td>"
            + "".join(cell_for(s) for s in samples)
            + "</tr>"
        )

    return (
        f"<table class='audio-grid cross-sample'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def baseline_card_html(base: dict) -> str:
    if not base:
        return "<p class='muted'>no baseline</p>"
    return (
        "<table class='baseline'>"
        f"<tr><th>duration</th><td>{fmt(base['duration_s'], '.1f')} s</td>"
        f"<th>peak</th><td>{fmt(base['peak_dbfs'], '.1f')} dBFS</td></tr>"
        f"<tr><th>LUFS</th><td>{fmt(base.get('lufs'), '.2f')}</td>"
        f"<th>true peak</th><td>{fmt(base.get('true_peak_dbtp'), '.2f')} dBTP</td></tr>"
        f"<tr><th>LRA</th><td>{fmt(base.get('dynamic_range_lu'), '.1f')} LU</td>"
        f"<th>centroid</th><td>{fmt(base.get('spectral_centroid_hz'), '.0f')} Hz</td></tr>"
        f"<tr><th>hiss band</th><td>{fmt(base['hiss_band_energy_db'], '.1f')} dB</td>"
        f"<th>speech band</th><td>{fmt(base['speech_band_energy_db'], '.1f')} dB</td></tr>"
        "</table>"
    )


def per_sample_html(samples: list[dict]) -> str:
    """One card per sample, each with baseline + ranking."""
    cards = []
    for s in samples:
        # Compute hours/1k for each result row, attach for the renderer.
        base_dur_min = s["baseline"]["duration_s"] / 60.0
        results = []
        for r in s["results"]:
            r2 = dict(r)
            sc = r2.get("score", {})
            if not sc.get("skipped") and sc.get("rtf"):
                r2["_hours_per_1k"] = base_dur_min * sc["rtf"] * 1000 / 60.0
            else:
                r2["_hours_per_1k"] = None
            results.append(r2)
        # Sort: skipped at bottom, then by score desc.
        def _rank_key(r):
            sc = r.get("score", {}) or {}
            if sc.get("skipped"):
                return (1, 0.0)
            return (0, -float(sc.get("score", 0.0) or 0.0))
        results.sort(key=_rank_key)

        winners = [r for r in results if not r.get("score", {}).get("skipped") and r.get("exit_code") == 0][:3]
        winner_html = ""
        if winners:
            top = winners[0]
            sc = top["score"]
            winner_html = (
                "<div class='winner'>"
                f"<strong>Top:</strong> <code>{escape(top['candidate'])}</code> "
                f"score {sign(sc['score'], '.2f')} "
                f"(hiss {sign(sc['hiss_delta_db'])}, speech {sign(sc['speech_delta_db'])} dB, "
                f"RTF {fmt(sc['rtf'], '.2f')})"
                "</div>"
            )

        cards.append(
            "<section class='card'>"
            f"<h2>{escape(s['src'])}</h2>"
            f"<p class='meta'>sample: <code>{escape(s['sample'])}</code> · "
            f"clip: <code>{escape(Path(s['clip']).name)}</code></p>"
            "<h3>Baseline</h3>"
            f"{baseline_card_html(s['baseline'])}"
            "<h3>Candidate ranking</h3>"
            f"{candidate_table_html(results, sample=s['sample'])}"
            f"{winner_html}"
            "<h3>Listen &mdash; side by side</h3>"
            "<p class='meta'>Click any ▶ to expand the player. Higher-scored "
            "candidates are to the left of the source for quick A/B.</p>"
            f"{per_sample_audio_table_html(s)}"
            "</section>"
        )
    return "\n".join(cards)


def _candidate_score_table(samples: list[dict]) -> dict:
    """Build the candidate → per-sample score map shared by the consistency
    table and the unified ranking table. Returns:

      {
        'rows': [ { 'cand', 'scores': [...], 'mean', 'std', 'wins' }, ... ],
        'winning_scores': [float, ...],  # max score per sample
        'samples_in_order': [sample_dict, ...],  # for header rendering
      }

    Sorted by mean score desc; candidates with no valid scores are dropped.
    """
    # For each sample, find the winning score (max across candidates).
    winning_scores = []
    for s in samples:
        valid = [
            r.get("score", {}).get("score")
            for r in s["results"]
            if not r.get("score", {}).get("skipped")
        ]
        winning_scores.append(max(valid) if valid else None)

    cand_scores: dict[str, list[float | None]] = {}
    for s in samples:
        for r in s["results"]:
            sc = r.get("score", {})
            val = None if sc.get("skipped") else sc.get("score")
            cand_scores.setdefault(r["candidate"], []).append(val)

    rows = []
    for cand, scores in cand_scores.items():
        valid = [x for x in scores if x is not None]
        if not valid:
            continue
        mean = statistics.mean(valid)
        std = statistics.stdev(valid) if len(valid) > 1 else 0.0
        # "wins (top-1)" = number of samples where this candidate's score
        # matches the per-sample winner. Real ranking, not self-max.
        wins = sum(
            1 for s, win in zip(scores, winning_scores)
            if s is not None and win is not None and s == win
        )
        rows.append({
            "cand": cand, "scores": scores,
            "mean": mean, "std": std, "wins": wins,
            "n_valid": len(valid),
        })
    rows.sort(key=lambda r: -r["mean"])
    return {"rows": rows, "winning_scores": winning_scores,
            "samples_in_order": samples}


def consistency_table_html(samples: list[dict]) -> str:
    """One row per candidate, columns are scores across samples, plus mean ± std."""
    data = _candidate_score_table(samples)
    cells = lambda scores: "".join(
        f"<td class='num'>{sign(s, '.2f') if s is not None else '—'}</td>"
        for s in scores
    )
    rows_html = "".join(
        f"<tr><td class='cand'>{escape(r['cand'])}</td>{cells(r['scores'])}"
        f"<td class='num score'>{sign(r['mean'], '.2f')}</td>"
        f"<td class='num'>{fmt(r['std'], '.2f')}</td>"
        f"<td class='num'>{r['wins']}/{r['n_valid']}</td></tr>"
        for r in data["rows"]
    )
    header = "".join(f"<th>{escape(s['sample'])}</th>" for s in data["samples_in_order"])
    return (
        "<table class='consistency'>"
        f"<thead><tr><th>candidate</th>{header}"
        "<th>mean score</th><th>std-dev</th><th>wins (top-1)</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def unified_ranking_table_html(samples: list[dict]) -> str:
    """Single ranking table that replaces the 3 per-sample ranking tables.

    One row per candidate. Columns: candidate, mean, std, per-sample score
    (one column per sample, in the order they appear in SOURCES), wins, your
    rating (aggregate across samples, populated by JS at load time).
    """
    data = _candidate_score_table(samples)
    score_cells = lambda scores: "".join(
        f"<td class='num'>{sign(s, '.2f') if s is not None else '—'}</td>"
        for s in scores
    )
    rows_html = "".join(
        f"<tr><td class='cand'>{escape(r['cand'])}</td>"
        f"{score_cells(r['scores'])}"
        f"<td class='num score'>{sign(r['mean'], '.2f')}</td>"
        f"<td class='num'>{fmt(r['std'], '.2f')}</td>"
        f"<td class='num'>{r['wins']}/{r['n_valid']}</td>"
        f"<td class='num rating-aggregate' data-cand='{escape(r['cand'], quote=True)}'>—</td>"
        f"</tr>"
        for r in data["rows"]
    )
    header_scores = "".join(
        f"<th>{escape(s['sample'])}<br><span class='muted' style='font-weight:400'>{escape(s['src'])}</span></th>"
        for s in data["samples_in_order"]
    )
    return (
        "<table class='unified'>"
        f"<thead><tr><th>candidate</th>{header_scores}"
        "<th>mean</th><th>std-dev</th><th>wins</th><th>your rating ★</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def tldr_card_html(samples: list[dict]) -> str:
    """The TL;DR card at the top of the page.

    Three side-by-side blocks:
      1. Top by mean score — the overall recommended winner.
      2. Top per sample — the winner for each of the 3 tapes.
      3. Your verdict (top-3 by user rating) — JS-populated.

    Reads from the same `_candidate_score_table()` data the unified table
    uses, so the numbers can't drift.
    """
    data = _candidate_score_table(samples)
    rows = data["rows"]

    # Block 1: top by mean score.
    if rows:
        top = rows[0]
        per_sample = "".join(
            f"<span class='tldr-chip'>{escape(s['sample'])}: "
            f"<strong>{sign(score, '.2f') if score is not None else '—'}</strong></span>"
            for s, score in zip(data["samples_in_order"], top["scores"])
        )
        block_mean = (
            "<div class='block block-mean'>"
            "<h3>Top by mean score</h3>"
            f"<div class='big'><code>{escape(top['cand'])}</code></div>"
            f"<div class='big-num'>{sign(top['mean'], '.2f')}</div>"
            f"<div class='muted'>std-dev {fmt(top['std'], '.2f')} · "
            f"wins {top['wins']}/{top['n_valid']}</div>"
            f"<div class='chips'>{per_sample}</div>"
            "</div>"
        )
    else:
        block_mean = "<div class='block block-mean'><h3>Top by mean score</h3><p class='muted'>no completed candidates</p></div>"

    # Block 2: top per sample. For each sample, find the candidate with the
    # highest score; the row label is the sample short name.
    per_sample_blocks = []
    for s in data["samples_in_order"]:
        best = None
        for r in s["results"]:
            sc = r.get("score", {})
            if sc.get("skipped"):
                continue
            v = sc.get("score")
            if v is None:
                continue
            if best is None or v > best[1]:
                best = (r["candidate"], v)
        if best is None:
            per_sample_blocks.append(
                f"<li><code>{escape(s['sample'])}</code> — <span class='muted'>no winner</span></li>"
            )
        else:
            per_sample_blocks.append(
                f"<li><code>{escape(s['sample'])}</code> — "
                f"<strong><code>{escape(best[0])}</code></strong> "
                f"<span class='big-num'>{sign(best[1], '.2f')}</span></li>"
            )
    block_per_sample = (
        "<div class='block block-per-sample'>"
        "<h3>Top per sample</h3>"
        f"<ul class='tldr-list'>{''.join(per_sample_blocks)}</ul>"
        "</div>"
    )

    # Block 3: your verdict (top-3 by user rating). JS populates at load
    # time. Empty placeholder so the layout is stable.
    block_verdict = (
        "<div class='block block-verdict'>"
        "<h3>Your verdict (★ ratings)</h3>"
        "<span id='user-verdict' class='muted'>loading…</span>"
        "<p class='muted small'>Click ★ in any per-sample ranking below.</p>"
        "</div>"
    )

    return f"<section class='tldr'>{block_mean}{block_per_sample}{block_verdict}</section>"


def per_sample_html_collapsed(samples: list[dict]) -> str:
    """Render the per-sample cards inside <details> blocks, collapsed by
    default. Reuses `per_sample_html()`'s card content but wraps each in a
    <details> with the sample name as the summary.
    """
    # Reuse the existing per_sample_html() to get the full card markup, then
    # split on each card's <section class='card'>...</section>. Cheap and
    # avoids duplicating the per-sample render logic.
    raw_cards = per_sample_html(samples)
    # per_sample_html() returns "".join(cards); each card is one <section
    # class='card'>. We split on the opening of each card and pair it with
    # the sample title parsed from the card's <h2>.
    parts = re.findall(
        r"<section class='card'>(.*?)</section>", raw_cards, flags=re.DOTALL
    )
    titles = re.findall(r"<h2>(.*?)</h2>", raw_cards, flags=re.DOTALL)
    out = []
    for title, body in zip(titles, parts):
        out.append(
            "<details class='per-sample-card'>"
            f"<summary><span class='ps-name'>{escape(title)}</span>"
            "<span class='muted small'>click to expand baseline + A/B</span>"
            "</summary>"
            f"<section class='card'>{body}</section>"
            "</details>"
        )
    return "".join(out)


CSS = """
:root { --fg: #1a1a1a; --muted: #6b6b6b; --bg: #fff; --card: #f7f7f8;
        --border: #e2e2e5; --accent: #0b6bcb; --warn: #b45f06; --ok: #1f7a3a; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
       color: var(--fg); background: var(--bg); max-width: 1200px;
       margin: 2rem auto; padding: 0 1.5rem; line-height: 1.45; }
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.3rem; margin: 1.5rem 0 0.25rem; }
h3 { font-size: 1.0rem; margin: 1.0rem 0 0.5rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: 0.04em; }
.meta { color: var(--muted); font-size: 0.9rem; margin: 0 0 0.5rem; }
code { font-family: "SF Mono", "Menlo", monospace; background: #f0f0f3;
       padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
section.card { background: var(--card); border: 1px solid var(--border);
               border-radius: 8px; padding: 1.25rem 1.5rem; margin: 1.5rem 0; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border);
         font-size: 0.92rem; }
th { background: #ececef; font-weight: 600; color: var(--muted); }
td.num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: "SF Mono", "Menlo", monospace; }
th:not(:first-child), td.num { text-align: right; }
td.cand { font-family: "SF Mono", "Menlo", monospace; font-size: 0.88rem; }
td.score { font-weight: 700; }
tr.skipped { color: var(--muted); }
tr.skipped td.skip-note { font-style: italic; }
table.baseline th, table.baseline td { font-size: 0.88rem; }
table.baseline th { background: transparent; color: var(--muted); font-weight: 500; }
.winner { background: #e8f3ff; border-left: 3px solid var(--accent);
          padding: 0.6rem 0.9rem; margin: 0.5rem 0 0; border-radius: 0 4px 4px 0; }
.summary { margin-top: 2.5rem; padding: 1rem 1.5rem;
           background: #fef9e7; border: 1px solid #f4d970; border-radius: 8px; }
.muted { color: var(--muted); }
table.audio-grid { width: 100%; table-layout: fixed; }
table.audio-grid th { font-size: 0.78rem; font-weight: 500; }
table.audio-grid td { padding: 6px; vertical-align: top; }
td.audio-cell { text-align: center; }
.audio-label { font-size: 0.78rem; color: var(--muted);
               font-family: "SF Mono", "Menlo", monospace;
               margin-bottom: 4px; word-wrap: break-word; }
button.play-btn { font: inherit; font-size: 0.85rem; padding: 4px 12px;
                  border: 1px solid var(--border); border-radius: 4px;
                  background: #fff; cursor: pointer; color: var(--accent); }
button.play-btn:hover { background: #e8f3ff; border-color: var(--accent); }
audio { display: block; width: 100%; height: 32px; margin-top: 6px; }
audio.hidden { display: none; }
table.audio-grid.cross-sample td.audio-cell { padding: 4px; }
table.audio-grid tr.src-row { background: #fffbe6; }
footer { color: var(--muted); font-size: 0.85rem; margin: 2.5rem 0 1rem;
         border-top: 1px solid var(--border); padding-top: 1rem; }

/* 1-5 star rating widget. Five <button>s in a flex row. The .filled class
   is painted by JS based on the stored value; the .hover-N class is
   painted by the mouseenter handler to preview an in-progress rating. */
.rating { display: inline-flex; gap: 1px; user-select: none; }
.rating button { background: transparent; border: 0; padding: 1px 3px;
                 font-size: 1.1rem; line-height: 1; cursor: pointer;
                 color: #d4d4d8; transition: color 80ms ease; }
.rating button.filled { color: #f5a623; }
.rating button:focus { outline: 2px solid var(--accent); outline-offset: 1px;
                       border-radius: 2px; }
/* Hover preview: when JS adds .hover-N to the parent, stars 1..N go gold. */
.rating.hover-1 button:nth-child(-n+1),
.rating.hover-2 button:nth-child(-n+2),
.rating.hover-3 button:nth-child(-n+3),
.rating.hover-4 button:nth-child(-n+4),
.rating.hover-5 button:nth-child(-n+5) { color: #f5a623; }
td.rating-cell { padding: 4px 6px; text-align: center; white-space: nowrap; }
.user-verdict { margin-top: 0.75rem; font-size: 0.95rem; }
.user-verdict #user-verdict { color: var(--fg); }

/* Sticky nav. Stays visible while scrolling; bg matches the page so it
   doesn't visually float over content. Pills are small + muted. */
nav.sticky { position: sticky; top: 0; z-index: 10; background: var(--bg);
             border-bottom: 1px solid var(--border);
             padding: 8px 0; margin: 0 -1.5rem 1.5rem;
             display: flex; flex-wrap: wrap; gap: 6px;
             padding-left: 1.5rem; padding-right: 1.5rem; }
nav.sticky a { font-size: 0.85rem; padding: 4px 10px; border-radius: 999px;
               border: 1px solid var(--border); color: var(--fg);
               text-decoration: none; background: var(--card); }
nav.sticky a:hover { background: #e8f3ff; border-color: var(--accent); color: var(--accent); }

/* TL;DR card: 3 side-by-side blocks that answer "who is winning". */
section.tldr { display: grid; grid-template-columns: 1.2fr 1fr 1fr;
              gap: 1rem; margin: 1.25rem 0 1.5rem; }
section.tldr .block { background: var(--card); border: 1px solid var(--border);
                      border-radius: 8px; padding: 1rem 1.25rem;
                      border-left: 3px solid var(--accent); }
section.tldr .block-mean { border-left-color: var(--accent); }
section.tldr .block-per-sample { border-left-color: var(--ok); }
section.tldr .block-verdict { border-left-color: #b45f06; }
section.tldr h3 { margin: 0 0 0.5rem; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.04em;
                 font-size: 0.75rem; }
section.tldr .big { font-size: 1.0rem; margin-bottom: 0.25rem; }
section.tldr .big-num { font-size: 1.5rem; font-weight: 700;
                       font-variant-numeric: tabular-nums;
                       font-family: "SF Mono", "Menlo", monospace; }
section.tldr .chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 0.5rem; }
section.tldr .tldr-chip { font-size: 0.78rem; padding: 2px 6px;
                          border: 1px solid var(--border); border-radius: 4px;
                          background: #fff; font-family: "SF Mono", "Menlo", monospace; }
section.tldr .tldr-list { list-style: none; padding: 0; margin: 0; }
section.tldr .tldr-list li { padding: 4px 0; border-bottom: 1px dashed var(--border);
                             display: flex; align-items: center; gap: 8px; }
section.tldr .tldr-list li:last-child { border-bottom: 0; }
section.tldr .small { font-size: 0.75rem; margin-top: 0.4rem; }

/* Unified ranking table: same column types as the consistency table it
   replaces, plus a .rating-aggregate cell. */
table.unified th, table.unified td { font-size: 0.85rem; }
table.unified td.rating-aggregate { font-family: "SF Mono", "Menlo", monospace;
                                    color: var(--muted); }

/* Per-sample <details> cards: closed by default; click to expand. The
   summary acts as a clickable header that mimics the .card look. */
details.per-sample-card { margin: 0.75rem 0; }
details.per-sample-card > summary { cursor: pointer; list-style: none;
        background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 0.6rem 1rem; display: flex;
        align-items: baseline; gap: 1rem; }
details.per-sample-card > summary::-webkit-details-marker { display: none; }
details.per-sample-card > summary::before { content: '▸'; color: var(--muted);
        font-size: 0.9rem; transition: transform 120ms ease; display: inline-block; }
details.per-sample-card[open] > summary::before { transform: rotate(90deg); }
details.per-sample-card > summary .ps-name { font-weight: 600; }
details.per-sample-card[open] > summary { border-bottom-left-radius: 0;
        border-bottom-right-radius: 0; border-bottom-color: transparent; }
details.per-sample-card[open] > section.card { border-top-left-radius: 0;
        border-top-right-radius: 0; margin-top: 0; }

/* Responsive: stack the TL;DR on narrow screens. */
@media (max-width: 800px) {
  section.tldr { grid-template-columns: 1fr; }
  nav.sticky { margin: 0 -1rem 1.5rem; padding-left: 1rem; padding-right: 1rem; }
}
"""


def render_html(samples: list[dict], dest: Path) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # JS lives in a separate non-f-string so its literal `{` and `}` don't
    # collide with f-string substitutions in `body` below.
    js = """
// Click any ▶ to expand the inline <audio controls>. Click again to collapse.
document.addEventListener('click', function (e) {
  var btn = e.target.closest('.play-btn');
  if (!btn) return;
  var id = btn.getAttribute('data-target');
  var audio = document.getElementById(id);
  if (!audio) return;
  audio.classList.toggle('hidden');
  btn.textContent = audio.classList.contains('hidden') ? '▶ play' : '⏸ hide';
  if (!audio.classList.contains('hidden')) {
    audio.play().catch(function(){});
  } else {
    audio.pause();
  }
});

// ---------- 1-5 star ratings ----------
// Ratings live in localStorage under a single key as a JSON map of
// "<sample>::<candidate>" → integer 1..5. Click same star to unset.
var RATINGS_KEY = 'audio_restore_ratings_v1';
function loadRatings() {
  try {
    var raw = localStorage.getItem(RATINGS_KEY);
    if (!raw) return {};
    var parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch (err) {
    return {};
  }
}
function saveRatings(map) {
  try { localStorage.setItem(RATINGS_KEY, JSON.stringify(map)); }
  catch (err) { /* quota or disabled — ignore */ }
}
function paintRating(ratingEl, n) {
  ratingEl.setAttribute('data-rating', n);
  var stars = ratingEl.querySelectorAll('button');
  for (var i = 0; i < stars.length; i++) {
    var v = parseInt(stars[i].getAttribute('data-n'), 10);
    if (v <= n) stars[i].classList.add('filled');
    else stars[i].classList.remove('filled');
  }
}

// Click: set or unset a rating.
document.addEventListener('click', function (e) {
  var star = e.target.closest('.rating button');
  if (!star) return;
  var rating = star.parentElement;
  var sample = rating.getAttribute('data-sample');
  var cand = rating.getAttribute('data-cand');
  if (!sample || !cand) return;
  var n = parseInt(star.getAttribute('data-n'), 10);
  var map = loadRatings();
  var key = sample + '::' + cand;
  if (map[key] === n) { delete map[key]; }
  else { map[key] = n; }
  saveRatings(map);
  paintRating(rating, map[key] || 0);
  refreshRatingViews();
});

// Hover preview: highlight stars 1..N when hovering star N.
document.addEventListener('mouseover', function (e) {
  var star = e.target.closest('.rating button');
  if (!star) return;
  var rating = star.parentElement;
  var n = parseInt(star.getAttribute('data-n'), 10);
  // Remove all hover-N classes first.
  for (var k = 1; k <= 5; k++) rating.classList.remove('hover-' + k);
  rating.classList.add('hover-' + n);
});
document.addEventListener('mouseout', function (e) {
  var star = e.target.closest('.rating button');
  if (!star) return;
  // Only clear if the mouse actually left the .rating container.
  var rating = star.parentElement;
  if (rating.contains(e.relatedTarget)) return;
  for (var k = 1; k <= 5; k++) rating.classList.remove('hover-' + k);
});

// Recompute mean rating per candidate and render the user-verdict block AND
// every td.rating-aggregate cell. Both views read from the same aggregation
// so they cannot drift.
function refreshRatingViews() {
  var map = loadRatings();
  // Aggregate by candidate across all samples.
  var byCand = {};
  for (var key in map) {
    if (!Object.prototype.hasOwnProperty.call(map, key)) continue;
    var parts = key.split('::');
    if (parts.length !== 2) continue;
    var cand = parts[1];
    if (!byCand[cand]) byCand[cand] = [];
    byCand[cand].push(map[key]);
  }
  var entries = [];
  for (var c in byCand) {
    if (!Object.prototype.hasOwnProperty.call(byCand, c)) continue;
    var rs = byCand[c];
    var mean = rs.reduce(function (a, b) { return a + b; }, 0) / rs.length;
    entries.push({ cand: c, mean: mean, n: rs.length });
  }
  // User-verdict block in the TL;DR card + summary footer.
  var verdictEl = document.getElementById('user-verdict');
  if (verdictEl) {
    if (entries.length === 0) {
      verdictEl.innerHTML = "<em>no ratings yet — click ★ below to rate</em>";
    } else {
      entries.sort(function (a, b) {
        if (b.mean !== a.mean) return b.mean - a.mean;
        return b.n - a.n;
      });
      var top = entries.slice(0, 3);
      verdictEl.innerHTML = top.map(function (e) {
        return "<code>" + escapeHtml(e.cand) + "</code> " +
               "<strong>" + e.mean.toFixed(2) + "</strong>/5" +
               " <span class='muted'>(" + e.n + " sample" +
               (e.n === 1 ? "" : "s") + ")</span>";
      }).join(" &nbsp;·&nbsp; ");
    }
  }
  // Unified ranking table's per-row aggregate cells. Each cell carries
  // data-cand; we look up that candidate in byCand and write the mean.
  var cells = document.querySelectorAll('td.rating-aggregate');
  for (var i = 0; i < cells.length; i++) {
    var cell = cells[i];
    var name = cell.getAttribute('data-cand');
    var entry = byCand[name];
    if (entry && entry.length) {
      var m = entry.reduce(function (a, b) { return a + b; }, 0) / entry.length;
      cell.textContent = m.toFixed(2) + ' ★ (' + entry.length + ')';
      cell.classList.remove('muted');
    } else {
      cell.textContent = '—';
      cell.classList.add('muted');
    }
  }
}

// Minimal HTML escaper for the verdict list (defense in depth — the
// candidate names come from localStorage which the user controls locally,
// but the file:// origin is shared so we don't trust the storage either).
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Paint every .rating element from storage, then refresh both views.
function initRatings() {
  var map = loadRatings();
  var widgets = document.querySelectorAll('.rating');
  for (var i = 0; i < widgets.length; i++) {
    var w = widgets[i];
    var key = w.getAttribute('data-sample') + '::' + w.getAttribute('data-cand');
    paintRating(w, map[key] || 0);
  }
  refreshRatingViews();
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initRatings);
} else {
  initRatings();
}
"""
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>audio_restore — multi-sample compare</title>
<style>{CSS}</style>
</head>
<body>
<h1>audio_restore — multi-sample compare</h1>
<p class="meta">Generated {escape(now)} · {len(samples)} source(s):
{', '.join(f'<code>{escape(s["src"])}</code>' for s in samples)}</p>

<nav class="sticky">
  <a href="#winners">🏆 TL;DR</a>
  <a href="#rankings">📊 Ranking</a>
  <a href="#audio">🎧 Listen</a>
  <a href="#per-sample">📁 Per-sample</a>
  <a href="#ratings">⭐ Ratings</a>
</nav>

<section id="winners">
{tldr_card_html(samples)}
</section>

<h2 id="rankings">Ranking across all samples</h2>
<p class="meta">One row per candidate. Score columns = per-sample score.
Mean / std-dev / wins are computed from the same data. Click ★ in a
per-sample card below to add a rating — it will appear in the rightmost
"your rating ★" column.</p>
{unified_ranking_table_html(samples)}

<h2 id="audio">Listen &mdash; same candidate across samples</h2>
<p class="meta">One row per candidate, columns = samples. Click ▶ to
expand the player. Source row at top is the unprocessed 10-minute clip
for A/B reference. Use this for the "ear test" &mdash; the same
candidate on different tapes.</p>
{per_candidate_audio_table_html(samples)}

<h2 id="per-sample">Per-sample detail</h2>
<p class="meta">Click a sample to expand its baseline metrics and
side-by-side A/B audio (all candidates for that one sample). Closed
by default to keep this section scannable.</p>
{per_sample_html_collapsed(samples)}

<div class="summary" id="ratings">
{summary_html(samples)}
</div>

<footer>
<p>Source files are in <code>samples/source/</code> (read-only).
10-minute clips live in <code>/tmp/audio_restore_clips/</code>.
Per-candidate outputs are under <code>stages/&lt;sample&gt;/</code>.</p>
<p>Scoring formula: <code>score = 2·hiss − max(0,+speech) − 0.5·max(0,−LRA) − max(0,RTF−2) − 0.3·max(0,|LUFS+16|−3)</code></p>
</footer>

<script>
{js}
</script>
</body>
</html>
"""
    dest.write_text(body)
    print(f"[html] wrote {dest}")


def summary_html(samples: list[dict]) -> str:
    """Recommend the candidate with the highest mean score across samples."""
    cand_scores: dict[str, list[float | None]] = {}
    for s in samples:
        for r in s["results"]:
            sc = r.get("score", {})
            cand_scores.setdefault(r["candidate"], []).append(
                None if sc.get("skipped") else sc.get("score")
            )
    summary = []
    for cand, scores in cand_scores.items():
        valid = [x for x in scores if x is not None]
        if valid:
            summary.append((
                cand,
                statistics.mean(valid),
                statistics.stdev(valid) if len(valid) > 1 else 0.0,
                len(valid),
            ))
    summary.sort(key=lambda x: -x[1])
    if not summary:
        return "<p class='muted'>No completed candidates.</p>"
    top = summary[0]
    runs = " / ".join(
        f"{len([x for x in s['results'] if not x.get('score', {}).get('skipped')])} of {len(s['results'])}"
        for s in samples
    )
    return (
        f"<strong>Recommended by mean score across {len(samples)} sample(s):</strong> "
        f"<code>{escape(top[0])}</code> "
        f"with mean <strong>{top[1]:+.2f}</strong> "
        f"(std-dev {top[2]:.2f}, completed on {top[3]} of {len(samples)} samples). "
        f"<br><span class='muted'>Per-sample completed counts: {escape(runs)}.</span>"
        "<p class='muted' style='font-size: 0.8rem; margin-top: 0.5rem;'>"
        "Your ★ ratings appear at the top of the page in the TL;DR card. "
        "Ratings are saved in this browser only (localStorage). "
        "Re-running <code>make multi</code> preserves them. "
        "Click any ★ in a per-sample card to rate; click the same star "
        "again to clear."
        "</p>"
    )


# ---------- main ----------

def main() -> int:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify sources exist before doing any work.
    for src, _, _ in SOURCES:
        if not src.exists():
            print(f"error: source not found: {src}", file=sys.stderr)
            return 1

    samples = []
    for src, start_s, dur_s in SOURCES:
        sample = safe_name(src)
        # Filename reflects duration: "10min" if 600s, otherwise "<seconds>s".
        clip_name = f"{sample}_10min.wav" if dur_s == 600 else (
            f"{sample}_full.wav" if dur_s is None else f"{sample}_{dur_s}s.wav"
        )
        clip = CLIP_DIR / clip_name
        print(f"\n=== {src.name} → {clip.name} ===")
        make_clip(src, clip, start_s, dur_s)
        result = run_harness(clip, sample)
        result["src"] = src.name
        result["sample"] = sample
        result["clip"] = str(clip)
        samples.append(result)

    (REPORT_DIR / "all.json").write_text(json.dumps(samples, indent=2, default=str))
    render_html(samples, REPORT_DIR / "comparison.html")
    print(f"\n[done] open {REPORT_DIR / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
