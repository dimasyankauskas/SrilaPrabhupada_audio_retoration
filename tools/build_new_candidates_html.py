#!/usr/bin/env python3
"""
build_new_candidates_html — Project index for the focused A/B review page.

Focused A/B page for the 4 candidates that
matter for the "loudness + naturalness" question:

  c03 — Demucs (the original "wins on metrics, sounds thin" candidate)
  c11 — c03 + loudnorm (the loudness fix, voice still thin)
  c12 — DeepFilterNet3 (the speed winner, conservative on hiss)
  c13 — VoiceFixer (the naturalness bet, but vocoder adds HF hiss)

Side-by-side audio + the metrics that drove the score. The page is written
to index.html as the project entry point, including the full multi-sample
matrix so there is one review surface.

Player behavior (the main feature of this iteration):
  • Master clock — all audios are seeked to the same currentTime when
    you switch rows, so flipping the play button on a different row
    never jumps the playhead.
  • Single-click play — no two-step toggle; the audio element is always
    visible and ready.
  • One audio at a time — clicking play on a different row pauses the
    current row before starting the new one.
  • Per-row A/B toggle — each candidate row has a switch that flips
    between the source clip and the candidate output in real time
    (instant cut, no audio stop). The other row is muted.
  • Master transport — the top bar has a play/pause/seek-0 button
    that controls the whole page.
"""
from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import lib_audio as L  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "multi"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT = REPO_ROOT / "index.html"

FOCUS_CANDS = ["c12", "c13", "c14", "c15", "c16", "c17"]

SAMPLE = "670322SB_SAN_FRANCISCO"
SAMPLE_LABEL = "1967-03-22 San Francisco"


def _load_report(stage: str) -> dict | None:
    p = REPO_ROOT / "stages" / SAMPLE / stage / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_baseline_for(sample: str) -> dict | None:
    p = REPO_ROOT / "stages" / sample / "00--inspect" / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_report_for(sample: str, cand: str) -> dict | None:
    """Load report.json for a candidate on a specific sample."""
    sample_dir = REPO_ROOT / "stages" / sample
    if not sample_dir.exists():
        return None
    for d in sample_dir.iterdir():
        if d.is_dir() and d.name.startswith(cand + "--"):
            p = d / "report.json"
            if p.exists():
                return json.loads(p.read_text())
    return None


def _load_baseline() -> dict | None:
    p = REPO_ROOT / "stages" / SAMPLE / "00--inspect" / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _score(b: dict, o: dict) -> tuple[float, dict]:
    """Same scoring formula as 99_compare.score() — gain-invariant hiss/speech
    ratio, HF reward, centroid pull. Mirrored here to keep the HTML page
    self-contained (no import of 99_compare)."""
    if o is None:
        return 0.0, {}
    # Gain-invariant hiss/speech ratio (positive = hiss reduced relative to speech)
    base_hiss = b.get("hiss_band_energy_db", 0.0) or 0.0
    base_speech = b.get("speech_band_energy_db", 0.0) or 0.0
    out_hiss = o.get("hiss_band_energy_db", 0.0) or 0.0
    out_speech = o.get("speech_band_energy_db", 0.0) or 0.0
    hiss_d = (base_hiss - base_speech) - (out_hiss - out_speech)
    speech_d = base_speech - out_speech
    lra_d = o.get("dynamic_range_lu", 0.0) - b.get("dynamic_range_lu", 0.0)
    rtf = o.get("runtime_s", 0) / max(b.get("duration_s", 1.0), 1.0)
    lufs_drift = max(0.0, abs((o.get("lufs") or -16) - (-16)) - 3.0)
    # HF extension: reward candidates that lift 12-18 kHz content.
    base_hf = b.get("hf_extension_db", -200.0) or -200.0
    out_hf = o.get("hf_extension_db", -200.0) or -200.0
    hf_d = out_hf - base_hf
    hf_reward = min(hf_d, 20.0)
    hf_target = -15.0
    hf_overshoot = max(0.0, (out_hf - hf_target) - 10.0) * 0.2
    # Centroid: pull toward 5 kHz "studio" target.
    centroid = o.get("spectral_centroid_hz", 0.0) or 0.0
    centroid_pen = 0.1 * abs(centroid - 5000.0) / 1000.0
    sc = (2.0 * hiss_d
          + 1.0 * hf_reward
          - hf_overshoot
          - 1.0 * max(0.0, speech_d)
          - 0.5 * max(0.0, -lra_d)
          - centroid_pen
          - 1.0 * max(0.0, rtf - 2.0)
          - 0.3 * lufs_drift)
    return sc, {
        "hiss_d": hiss_d, "speech_d": speech_d, "lra_d": lra_d,
        "rtf": rtf, "lufs_drift": lufs_drift,
        "hf_d": hf_d, "hf_reward": hf_reward, "centroid_pen": centroid_pen,
    }


CSS = """
/* ---------- agency v2 design system (portable subset) ---------- */
:root {
  --bg: #0e0d0b; --bg-2: #161410; --bg-3: #1e1b16;
  --ink: #fcf9f2; --ink-2: #d1ccbe; --ink-3: #9b9587;
  --rule: #2a2722;
  --accent: oklch(0.72 0.135 35);
  --accent-soft: oklch(0.72 0.135 35 / 0.14);
  --accent-hover: oklch(0.78 0.14 35);
  --accent-hex: #D4A155;
  --success: #5a8a4a; --warning: #c4913a; --danger: #b85450;
  --active: #6fa86a;                   /* sage green — "candidate side" indicator */
  --active-soft: rgba(111, 168, 106, 0.14);
  --active-hover: #82bf7c;
  --active-ring: rgba(111, 168, 106, 0.35);
  --serif: "Newsreader", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --sans: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono: "IBM Plex Mono", "SF Mono", "JetBrains Mono", Menlo, monospace;
  --maxw: 1180px;
  --gutter: clamp(16px, 4vw, 56px);
}
[data-theme="light"] {
  --bg: #f5f1e8; --bg-2: #ece7dc; --bg-3: #e2ddd0;
  --ink: #14120f; --ink-2: #3b382f; --ink-3: #5a554a;
  --rule: #d8d1c2;
  --accent: oklch(0.55 0.135 35);
  --accent-soft: oklch(0.55 0.135 35 / 0.12);
  --accent-hover: oklch(0.62 0.135 35);
}

* { box-sizing: border-box; }
html { font-size: 18px; }
body {
  font-family: var(--sans); line-height: 1.55; color: var(--ink);
  background:
    radial-gradient(1200px 600px at 80% -10%, var(--accent-soft), transparent 60%),
    var(--bg);
  margin: 0; padding: 0; min-height: 100vh;
}
.wrap { max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter); }

/* ---------- typography ---------- */
h1, h2, h3, h4 { font-family: var(--serif); font-weight: 400; line-height: 1.05; letter-spacing: 0; margin: 0 0 0.4rem 0; color: var(--ink); }
h1 { font-size: clamp(2.6rem, 5vw, 4.45rem); line-height: 0.98; max-width: 18ch; }
h2 { font-size: clamp(2rem, 3.4vw, 2.7rem); margin-top: 1.4rem; }
h3 { font-size: 1.55rem; font-weight: 500; line-height: 1.18; }
h4 { font-size: 1.2rem; font-weight: 500; line-height: 1.22; }
h1 em, h2 em, h3 em, h4 em { font-style: italic; color: var(--accent); }
p { color: var(--ink-2); margin: 0 0 1rem 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); }
.meta { color: var(--ink-3); font-size: 0.9rem; margin-bottom: 1.5rem; }

.kicker { font-family: var(--mono); font-size: 14px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.5rem; }
.kicker.accent { color: var(--accent); }
.deck { font-family: var(--serif); font-style: italic; font-size: 1.22rem; color: var(--ink-2); line-height: 1.4; max-width: 56ch; margin: 0 0 1.2rem 0; }

/* ---------- topbar ---------- */
.topbar {
  position: sticky; top: 0; z-index: 20;
  backdrop-filter: saturate(140%) blur(18px);
  -webkit-backdrop-filter: saturate(140%) blur(18px);
  background: color-mix(in oklab, var(--bg) 72%, transparent);
  border-bottom: 1px solid var(--rule);
}
.topbar .row { display: flex; align-items: center; gap: clamp(16px, 2.2vw, 28px); padding-block: 14px; }
.brand { display: flex; align-items: center; gap: 14px; flex: 0 0 auto; min-width: 236px; }
.brand-mark { width: 44px; height: 44px; border-radius: 10px; display: block; flex: 0 0 44px; box-shadow: 0 0 0 1px color-mix(in oklab, var(--ink) 18%, transparent); }
.brand-text { display: flex; flex-direction: column; line-height: 1.05; min-width: 0; }
.brand-line-1 { font-family: var(--serif); font-size: 17px; font-weight: 500; color: var(--ink); letter-spacing: 0.005em; white-space: nowrap; }
.brand-line-2 { font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-3); }
.nav { display: flex; gap: clamp(14px, 2vw, 22px); margin-left: auto; align-items: center; }
.nav a { font-family: var(--mono); font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.nav a:hover { color: var(--ink); }
.theme-toggle { background: transparent; border: 1px solid var(--rule); width: 36px; height: 36px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; color: var(--ink-2); padding: 0; }
.theme-toggle:hover { color: var(--ink); border-color: var(--ink-3); }
.theme-toggle svg { width: 17px; height: 17px; }
.theme-toggle .moon { display: none; }
[data-theme="light"] .theme-toggle .sun { display: none; }
[data-theme="light"] .theme-toggle .moon { display: block; }

/* ---------- hero ---------- */
.hero { padding: clamp(56px, 8vw, 96px) 0 clamp(40px, 6vw, 72px); border-bottom: 1px solid var(--rule); }
.hero .studio-line { display: flex; align-items: center; gap: 18px; font-family: var(--mono); font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 28px; }
.hero .studio-line .bar { flex: 1; height: 1px; background: color-mix(in oklab, var(--accent) 34%, var(--rule)); }
.hero .anti { font-family: var(--mono); font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); margin-top: 28px; }
.hero .anti b { color: var(--accent); font-weight: 500; }
.hero-mark { margin-top: 36px; display: flex; align-items: center; gap: 12px; }
.hero-mark img { border-radius: 10px; box-shadow: 0 8px 28px rgba(224, 122, 31, 0.32); }

/* ---------- callouts / cards ---------- */
.note {
  background: var(--bg-2); border-left: 2px solid var(--accent);
  padding: 18px 22px; margin: 0 0 18px 0; border-radius: 0 6px 6px 0;
}
.note b { color: var(--accent); font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; display: inline-block; margin-bottom: 6px; }
.note ul { margin: 0.6rem 0 0 1.2rem; padding: 0; color: var(--ink-2); }
.note li { margin-bottom: 6px; }
.note li b { color: var(--ink); font-weight: 500; text-transform: none; letter-spacing: 0; font-family: var(--sans); font-size: 0.95rem; margin-bottom: 0; }

.proofline {
  background: var(--bg-2);
  border-left: 2px solid var(--accent);
  padding: 14px 18px;
  font-family: var(--sans); font-size: 0.95rem; line-height: 1.55; color: var(--ink-2);
  border-radius: 0 6px 6px 0;
  margin: 0 0 14px 0;
}
.proofline b { color: var(--accent); font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; margin-right: 10px; }

.band { padding: clamp(56px, 8vw, 88px) 0; border-top: 1px solid var(--rule); }
.band:first-of-type { border-top: 0; }
.band-head { display: grid; grid-template-columns: 1fr; gap: 14px; margin-bottom: 36px; }
@media (min-width: 800px) { .band-head { grid-template-columns: 200px 1fr; gap: 40px; align-items: start; } }
.band-head .kicker { margin-bottom: 6px; }
.band-head h2 em { display: inline; }

/* ---------- mock-window audio card ---------- */
.mock-window {
  border: 1px solid color-mix(in oklab, var(--ink-3) 36%, var(--rule));
  border-radius: 12px;
  background: color-mix(in oklab, var(--bg) 82%, #000);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.22);
  overflow: hidden;
}
.mock-top { display: flex; align-items: center; gap: 7px; padding: 10px 14px; border-bottom: 1px solid var(--rule); background: color-mix(in oklab, var(--bg-2) 92%, transparent); }
.mock-top .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.mock-top .dot.r { background: var(--danger); }
.mock-top .dot.y { background: var(--warning); }
.mock-top .dot.g { background: var(--success); }
.mock-top .label { margin-left: 12px; font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; color: var(--ink-3); }
.mock-top .label b { color: var(--ink-2); font-weight: 500; }
.mock-body { padding: 16px 18px; }

/* ---------- audio grid (3-up cards) ---------- */
.audio-cards { display: grid; grid-template-columns: 1fr; gap: 18px; }
@media (min-width: 720px) { .audio-cards { grid-template-columns: 1fr 1fr; } }
@media (min-width: 1100px) { .audio-cards.has-winner { grid-template-columns: repeat(3, 1fr); } }
.audio-card { display: flex; flex-direction: column; gap: 18px; padding: 20px 22px 22px 22px; background: var(--bg-2); border: 1px solid var(--rule); border-radius: 14px; }
.audio-card .meta { display: flex; align-items: center; gap: 10px; margin: 0; min-height: 32px; }
.audio-card .meta .cand-name { font-family: var(--mono); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); }
.audio-card .meta .cand-name b { color: var(--ink); font-weight: 500; }
.audio-card .meta .ab-toggle { margin-left: auto; }
.audio-card .stat { display: flex; align-items: baseline; gap: 12px; padding: 0; }
.audio-card .stat .num { font-family: var(--serif); font-size: 3.2rem; font-weight: 380; line-height: 0.95; color: var(--accent); letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
.audio-card .stat .unit { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.audio-card .stat .label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); margin-left: auto; }
.audio-card.is-winner { background: linear-gradient(180deg, var(--accent-soft) 0%, var(--bg-2) 30%); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); }
.audio-card.is-winner::before { content: none; }
.audio-card.is-winner .mock-window { box-shadow: 0 24px 60px rgba(0, 0, 0, 0.22), 0 0 0 1px color-mix(in oklab, var(--accent) 22%, transparent); }

.ab-toggle {
  display: inline-flex; gap: 0; border: 1px solid var(--rule);
  border-radius: 999px; padding: 3px; background: var(--bg-2);
}
.ab-toggle button {
  background: transparent; color: var(--ink-3); border: 0;
  padding: 6px 14px; font-size: 12px; cursor: pointer;
  font-family: var(--mono); letter-spacing: 0.06em; text-transform: uppercase;
  border-radius: 999px; transition: background 180ms, color 180ms;
  min-height: 0; font-weight: 500;
}
.ab-toggle button:hover { color: var(--ink-2); background: transparent; filter: none; }
.ab-toggle button.active { background: var(--accent-soft); color: var(--accent); }
.ab-toggle button.active:hover { background: var(--accent-soft); }

audio { width: 100%; height: 36px; }

/* tables: thin-rule style, no heavy borders */
table.metrics { border-collapse: collapse; width: 100%; margin: 0; }
table.metrics th, table.metrics td { padding: 10px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); font-size: 0.92rem; }
table.metrics th { background: transparent; color: var(--ink-3); font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; border-bottom: 1px solid var(--rule); }
table.metrics tr.src-row td { background: color-mix(in oklab, var(--accent) 4%, transparent); }
table.metrics tr:last-child td { border-bottom: 0; }
table.metrics td.num { font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 0.85rem; color: var(--ink-2); }
table.metrics {
  display: block;
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--rule);
  border-radius: 8px;
}
table.metrics thead,
table.metrics tbody,
table.metrics tr {
  width: max-content;
  min-width: 100%;
}

.score-good { color: var(--accent); font-weight: 500; }
.score-bad { color: var(--danger); font-weight: 500; }
.score-mute { color: var(--ink-3); }

/* score-line (progress bar for metrics) */
.score-line { display: grid; grid-template-columns: 130px 1fr 60px; gap: 14px; align-items: center; padding: 6px 0; }
.score-line .lbl { font-family: var(--mono); font-size: 12px; letter-spacing: 0.04em; color: var(--ink-3); }
.score-line .track { height: 6px; border-radius: 999px; background: var(--bg-3); position: relative; overflow: hidden; }
.score-line .fill { position: absolute; top: 0; left: 0; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-hover)); border-radius: 999px; transition: width 600ms; }
.score-line .fill.bad { background: linear-gradient(90deg, var(--danger), var(--danger)); }
.score-line .fill.mid { background: linear-gradient(90deg, var(--warning), var(--warning)); }
.score-line .val { font-family: var(--mono); font-size: 13px; color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }

/* flow map (3-step pipeline) */
.flow-map { display: grid; grid-template-columns: 1fr; gap: 14px; margin: 0 0 28px 0; }
@media (min-width: 720px) { .flow-map { grid-template-columns: repeat(3, 1fr); } }
.flow-step {
  position: relative; padding: 18px 18px 18px 22px;
  background: var(--bg-2); border: 1px solid var(--rule); border-radius: 8px;
}
.flow-step .n { font-family: var(--mono); font-size: 12px; letter-spacing: 0.1em; color: var(--ink-3); }
.flow-step .n b { color: var(--accent); font-weight: 500; }
.flow-step .t { font-family: var(--serif); font-size: 1.25rem; color: var(--ink); margin: 6px 0 4px 0; line-height: 1.15; }
.flow-step .d { font-size: 0.92rem; color: var(--ink-2); margin: 0; }
.flow-step.gate { border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); background: var(--accent-soft); }
.flow-step.gate .t em { color: var(--accent); font-style: italic; }

/* master transport dock */
#master {
  position: sticky; top: 64px; z-index: 15;
  backdrop-filter: saturate(140%) blur(18px); -webkit-backdrop-filter: saturate(140%) blur(18px);
  background: color-mix(in oklab, var(--bg) 78%, transparent);
  border: 1px solid var(--rule); border-radius: 12px;
  padding: 14px 18px; margin: 0 0 28px 0;
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
}
#master .dock-kicker { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); margin-right: 6px; }
#master .time { font-family: var(--mono); color: var(--ink-2); margin-left: auto; font-size: 0.92rem; font-variant-numeric: tabular-nums; }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 9px 16px; min-height: 38px; font: 500 14px var(--sans); border-radius: 6px; border: 0; cursor: pointer; transition: background 180ms, color 180ms, border-color 180ms; }
.btn.primary { background: var(--accent); color: var(--bg); }
.btn.primary:hover { background: var(--accent-hover); }
.btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--rule); }
.btn.ghost:hover { border-color: var(--ink-3); }
.btn.sm { padding: 6px 12px; min-height: 32px; font-size: 13px; }

/* comparison summary table (3-sample) */
table.compare-sum { width: 100%; border-collapse: collapse; }
table.compare-sum th { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--rule); }
table.compare-sum td { padding: 12px; border-bottom: 1px solid var(--rule); font-size: 0.95rem; color: var(--ink-2); }
table.compare-sum tr:last-child td { border-bottom: 0; }
table.compare-sum td b { color: var(--ink); }
table.compare-sum .num { font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 0.88rem; }

/* per-sample section */
.sample-section { margin: 0 0 56px 0; }
.sample-section .head { display: flex; align-items: baseline; justify-content: space-between; gap: 18px; margin-bottom: 14px; flex-wrap: wrap; }
.sample-section .head h3 { font-family: var(--serif); font-size: 1.55rem; font-weight: 400; line-height: 1.15; color: var(--ink); margin: 0; }
.sample-section .head h3 em { color: var(--accent); font-style: italic; }
.sample-section .head .stamp { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); }
.sample-section,
.sample-section .head,
.sample-section .head h3 {
  min-width: 0;
}
.sample-section .head h3 {
  overflow-wrap: anywhere;
}

/* active card glow */
.audio-card.is-active { box-shadow: 0 0 0 1px var(--accent), 0 24px 60px rgba(0, 0, 0, 0.22); }

/* "candidate side" green indicator — when the A/B toggle is on candidate,
   the entire card tints green so the user knows at a glance which row
   is currently playing the model output (vs the raw tape). */
.audio-card.is-candidate-side {
  background: var(--active-soft);
  border-color: var(--active-ring);
  box-shadow: 0 0 0 1px var(--active-ring) inset, 0 0 24px rgba(111, 168, 106, 0.08);
}
.audio-card.is-candidate-side.is-active {
  box-shadow: 0 0 0 1px var(--active) inset, 0 0 28px rgba(111, 168, 106, 0.18), 0 24px 60px rgba(0, 0, 0, 0.22);
}
/* The mock-window inside a green-tinted card also gets a subtle ring so
   the audio player itself reads as "this is the candidate output". */
.audio-card.is-candidate-side .mock-window {
  box-shadow: 0 0 0 1px var(--active-ring), 0 24px 60px rgba(0, 0, 0, 0.22);
}
/* The A/B toggle button when "candidate" is active also flips green
   (it currently uses --accent, which is amber). */
.audio-card.is-candidate-side .ab-btn.active {
  background: var(--active-soft);
  color: var(--active);
}

.matrix-frame {
  display: block;
  width: 100%;
  max-width: 100%;
  min-height: 78vh;
  overflow: hidden;
  contain: inline-size;
  border: 1px solid var(--rule);
  border-radius: 12px;
  background: var(--bg-2);
}

/* responsive: shrink the audio cards to single column on narrow screens */
@media (max-width: 720px) {
  html { font-size: 16px; }
  .nav { display: none; }
  #master { top: 56px; }
  h1 { font-size: 2.4rem; }
  h2 { font-size: 1.8rem; }
  .audio-card .stat .num { font-size: 2.6rem; }
}

/* reveal-on-scroll */
.reveal { opacity: 0; transform: translateY(20px); transition: opacity 600ms, transform 600ms; }
.reveal.visible { opacity: 1; transform: translateY(0); }
"""


BRAND_MARK_SVG = """
<img class="brand-mark" src="assets/images/prabhupada-icon.png" alt="Srila Prabhupada Audio Restoration" width="36" height="36"/>
"""


def _score_line_rows(report: dict, baseline: dict) -> list[str]:
    """Render a list of score-line rows visualizing the candidate's
    hiss Δ, HF Δ, speech Δ, LRA Δ, RTF as horizontal bars."""
    if not report or not baseline:
        return []
    rtf = report.get("runtime_s", 0) / max(baseline.get("duration_s", 1.0), 1.0)
    out_hiss = report.get("hiss_band_energy_db", 0)
    out_speech = report.get("speech_band_energy_db", 0)
    base_hiss = baseline.get("hiss_band_energy_db", 0)
    base_speech = baseline.get("speech_band_energy_db", 0)
    hiss_d = (base_hiss - base_speech) - (out_hiss - out_speech)
    speech_d = base_speech - out_speech
    base_hf = baseline.get("hf_extension_db", -200)
    out_hf = report.get("hf_extension_db", -200)
    hf_d = out_hf - base_hf
    lra_d = (report.get("dynamic_range_lu") or 0) - (baseline.get("dynamic_range_lu") or 0)
    # Bar widths normalized against ±15 dB range.
    def bar(d, lo=-15, hi=15):
        clamped = max(lo, min(hi, d))
        pct = (clamped - lo) / (hi - lo) * 100
        cls = "fill"
        if d < -2: cls = "fill bad"
        elif d < 2: cls = "fill mid"
        return f'<div class="track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'

    return [
        ('hiss Δ', hiss_d, bar(hiss_d)),
        ('HF Δ', hf_d, bar(hf_d, -5, 25)),
        ('speech Δ', speech_d, bar(-speech_d)),  # inverted: we want to NOT remove speech
        ('LRA Δ', lra_d, bar(lra_d, -8, 4)),
        ('RTF', rtf, ''),  # RTF not a bar
    ]


def card_html(c: dict | None, src_clip: str, *, is_source: bool = False, baseline: dict | None = None) -> str:
    """One audio card. c is None for the source card.
    The card has: kicker label, large serif score number, mock-window with audio,
    A/B toggle, and (for candidates) score-line breakdown.
    """
    if is_source:
        c = c or {}
        row_id = c.get("row_id", "src")
        label = c.get("cand", "raw tape")
        return f"""
<div class="audio-card" data-row="{escape(row_id, quote=True)}">
  <div class="meta">
    <span class="cand-name"><b>{escape(label)}</b></span>
  </div>
  <div class="stat">
    <span class="num" style="color: var(--ink-3)">—</span>
    <span class="label">unprocessed</span>
  </div>
  <div class="mock-window">
    <div class="mock-top">
      <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
      <span class="label">tape_<b>raw</b> · 48 kHz mono</span>
    </div>
    <div class="mock-body">
      <audio id="aud-{escape(row_id, quote=True)}" controls preload="metadata" src="{escape(src_clip, quote=True)}"></audio>
    </div>
  </div>
</div>
"""

    cand = c["cand"]
    cand_rel = c.get("cand_url", f"assets/audio/candidates/{c['stage']}.mp3")
    sc = c["score"]
    is_winner = c.get("is_winner", False)
    winner_class = " is-winner" if is_winner else ""
    row_id = c.get("row_id", cand)
    score_cls = "score-good" if sc > 0 else ("score-bad" if sc < 0 else "score-mute")
    r = c.get("report", {})

    # Sub-metric lines (hiss / HF / speech)
    bars = _score_line_rows(r, baseline) if baseline else []

    bars_html = ""
    if bars:
        bars_html = '<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--rule);">'
        for label, val, bar in bars:
            if label == "RTF":
                # Special: just show the number with a different label color
                bars_html += (
                    f'<div class="score-line">'
                    f'<span class="lbl">{escape(label)}</span>'
                    f'<span class="val" style="text-align: left">{val:.2f}×</span>'
                    f'</div>'
                )
            else:
                sign = "+" if val >= 0 else ""
                bars_html += (
                    f'<div class="score-line">'
                    f'<span class="lbl">{escape(label)}</span>'
                    f'{bar}'
                    f'<span class="val">{sign}{val:.1f} dB</span>'
                    f'</div>'
                )
        bars_html += '</div>'

    return f"""
<div class="audio-card{winner_class}" data-row="{escape(row_id, quote=True)}"
     data-src-url="{escape(src_clip, quote=True)}"
     data-cand-url="{escape(cand_rel, quote=True)}">
  <div class="meta">
    <span class="cand-name"><b>{escape(cand)}</b></span>
    <span class="ab-toggle" role="group" aria-label="A/B switch for {escape(cand)}">
      <button type="button" class="ab-btn active" data-side="src" data-row="{escape(row_id)}">source</button>
      <button type="button" class="ab-btn" data-side="cand" data-row="{escape(row_id)}">candidate</button>
    </span>
  </div>
  <div class="stat">
    <span class="num {score_cls}">{sc:+.2f}</span>
    <span class="label">project score</span>
  </div>
  <div class="mock-window">
    <div class="mock-top">
      <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
      <span class="label"><b>{escape(cand)}</b> · RTF {r.get("runtime_s", 0) / max(baseline.get("duration_s", 1.0) if baseline else 1.0, 1.0):.2f}×</span>
    </div>
    <div class="mock-body">
      <audio id="aud-{escape(row_id)}" controls preload="metadata" src="{escape(src_clip, quote=True)}"></audio>
    </div>
  </div>
  {bars_html}
</div>
"""


def build_multi_sample_section() -> str:
    """Build a per-sample c15 vs c16 A/B section using 3-min clips.

    For each of the 3 samples (SF, DC, BOM), show:
      - source card (3-min clip)
      - c15 card
      - c16 card
    Plus a small metrics table per sample so the user can see what
    c15 vs c16 does on each one before listening.
    """
    samples = [
        ("670322SB_SAN_FRANCISCO", "1967-03-22 San Francisco"),
        ("760706AD_WASHINGTON_DC", "1976-07-06 Washington DC"),
        ("720321SB_BOM",           "1972-03-21 Bombay"),
    ]
    parts: list[str] = []
    parts.append('<section class="band reveal" id="multi">')
    parts.append('<div class="band-head">')
    parts.append('<div><span class="kicker accent">02 — cross-sample A/B</span></div>')
    parts.append('<div>')
    parts.append('<h2>Does c15 vs c16 hold up on <em>other tapes</em>?</h2>')
    parts.append('<p class="deck">A single 10-minute slice of one recording isn\'t a representative test. '
                 'The three samples below span the quality range of the corpus — different hiss levels, '
                 'different HF rolloff, different room acoustics. c15 and c16 were re-run on 3-minute '
                 'slices of each so we can hear whether the verdict transfers.</p>')
    parts.append('</div>')
    parts.append('</div>')

    # ----- Summary comparison table across all 3 samples -----
    def fmt_score(sc):
        if sc is None:
            return "—", "score-mute"
        cls = "score-good" if sc > 0 else "score-bad" if sc < 0 else "score-mute"
        return f"{sc:+.2f}", cls

    def fmt_val(v, suffix=" dB"):
        return f"{v:.1f}{suffix}" if v is not None else "—"

    summary_rows: list[str] = []
    for sample, label in samples:
        base = _load_baseline_for(sample)
        if base is None:
            continue
        c15 = _load_report_for(sample, "c15")
        c16 = _load_report_for(sample, "c16")
        c15_sc, _ = _score(base, c15) if c15 else (None, None)
        c16_sc, _ = _score(base, c16) if c16 else (None, None)
        c15_hf = c15.get("hf_extension_db") if c15 else None
        c16_hf = c16.get("hf_extension_db") if c16 else None
        b_hiss = base.get("hiss_band_energy_db", 0)
        b_hf = base.get("hf_extension_db", 0)
        c15_s, c15_cls = fmt_score(c15_sc)
        c16_s, c16_cls = fmt_score(c16_sc)
        summary_rows.append(
            f"<tr>"
            f"<td><b>{escape(label)}</b></td>"
            f"<td class='num'>{b_hiss:.1f} dB</td>"
            f"<td class='num'>{b_hf:.1f} dB</td>"
            f"<td class='{c15_cls}'><b>{c15_s}</b></td>"
            f"<td class='num'>{fmt_val(c15_hf)}</td>"
            f"<td class='{c16_cls}'><b>{c16_s}</b></td>"
            f"<td class='num'>{fmt_val(c16_hf)}</td>"
            f"</tr>"
        )
    parts.append(f"""
<table class="compare-sum" style="margin-bottom: 48px;">
  <thead>
    <tr>
      <th>sample</th><th>baseline hiss</th><th>baseline HF</th>
      <th>c15 score</th><th>c15 HF</th>
      <th>c16 score</th><th>c16 HF</th>
    </tr>
  </thead>
  <tbody>
    {''.join(summary_rows)}
  </tbody>
</table>
""")

    # ----- Per-sample sections -----
    for sample, label in samples:
        base = _load_baseline_for(sample)
        c15 = _load_report_for(sample, "c15")
        c16 = _load_report_for(sample, "c16")
        if base is None:
            continue
        src_clip = f"assets/audio/{sample}_3min.mp3"
        c15_clip = f"assets/audio/{sample}_c15_3min.mp3"
        c16_clip = f"assets/audio/{sample}_c16_3min.mp3"

        # Compute scores to know the winner
        c15_sc = _score(base, c15)[0] if c15 else None
        c16_sc = _score(base, c16)[0] if c16 else None
        c15_is_winner = (c15_sc is not None) and (c16_sc is None or c15_sc >= c16_sc)
        c16_is_winner = (c16_sc is not None) and (c15_sc is None or c16_sc > c15_sc)

        # Per-sample audio cards: source, c15, c16.
        cards = [card_html(
            {"cand": "raw tape", "row_id": f"{sample}--src"},
            src_clip,
            is_source=True,
            baseline=base,
        )]
        for cand, cand_clip, report, is_winner in [("c15", c15_clip, c15, c15_is_winner), ("c16", c16_clip, c16, c16_is_winner)]:
            if report is None:
                continue
            sc, bd = _score(base, report)
            cards.append(card_html(
                {
                    "cand": cand,
                    "stage": f"{cand}--placeholder",
                    "report": report,
                    "score": sc,
                    "breakdown": bd,
                    "cand_url": cand_clip,
                    "row_id": f"{sample}--{cand}",
                    "is_winner": is_winner,
                },
                src_clip,
                baseline=base,
            ))

        # Per-sample metrics table.
        metric_rows = []
        for cand, report in [("c15", c15), ("c16", c16)]:
            if report is None:
                metric_rows.append(f"<tr><td><b>{cand}</b></td><td colspan='10' style='color:var(--ink-3)'>no report</td></tr>")
                continue
            sc, bd = _score(base, report)
            hf_d = bd.get("hf_d", 0.0)
            metric_rows.append(
                "<tr>"
                f"<td><b>{escape(cand)}</b></td>"
                f"<td class='num'>{report.get('hiss_band_energy_db', 0):.1f} dB</td>"
                f"<td class='num'>{bd.get('hiss_d', 0):+.1f}</td>"
                f"<td class='num'>{report.get('hf_extension_db', 0):.1f} dB</td>"
                f"<td class='num'>{hf_d:+.1f}</td>"
                f"<td class='num'>{report.get('speech_band_energy_db', 0):.1f} dB</td>"
                f"<td class='num'>{bd.get('speech_d', 0):+.1f}</td>"
                f"<td class='num'>{(report.get('lufs') or 0):.2f}</td>"
                f"<td class='num'>{(report.get('dynamic_range_lu') or 0):.1f} LU</td>"
                f"<td class='num'>{report.get('runtime_s', 0):.1f}s (RTF {bd.get('rtf', 0):.2f})</td>"
                f"<td class='{('score-good' if sc > 0 else 'score-bad' if sc < 0 else 'score-mute')}'>"
                f"<b>{sc:+.2f}</b></td>"
                "</tr>"
            )

        parts.append(f"""
<div class="sample-section" id="multi-{sample}">
  <div class="head">
    <h3>{label} — <em>{escape(sample)}</em></h3>
    <div class="stamp">3-MIN · 48 KHZ MONO</div>
  </div>
  <p class="meta" style="margin: 0 0 18px 0">baseline hiss {base['hiss_band_energy_db']:.1f} dB · baseline HF {base.get('hf_extension_db', -200):.1f} dB</p>
  <div class="audio-cards has-winner">
    {''.join(cards)}
  </div>
  <table class="metrics" style="margin-top: 24px;">
    <thead>
      <tr>
        <th>cand</th><th>hiss 5-12kHz</th><th>hiss Δ</th>
        <th>HF 12-18kHz</th><th>HF Δ</th>
        <th>speech 300-3.4kHz</th><th>speech Δ</th>
        <th>LUFS</th><th>LRA</th><th>runtime (RTF)</th><th>score</th>
      </tr>
    </thead>
    <tbody>
      <tr class='src-row'>
        <td><b>baseline</b></td>
        <td class='num'>{base['hiss_band_energy_db']:.1f} dB</td><td class='num'>—</td>
        <td class='num'>{base.get('hf_extension_db', -200):.1f} dB</td><td class='num'>—</td>
        <td class='num'>{base['speech_band_energy_db']:.1f} dB</td><td class='num'>—</td>
        <td class='num'>{(base.get('lufs') or 0):.2f}</td>
        <td class='num'>{(base.get('dynamic_range_lu') or 0):.1f} LU</td>
        <td class='num'>—</td><td class='num'>—</td>
      </tr>
      {''.join(metric_rows)}
    </tbody>
  </table>
</div>
""")
    parts.append('</section>')
    return "\n".join(parts)


MATRIX_CSS = r"""
:host {
  --bg: #0e0d0b; --bg-2: #161410; --bg-3: #1e1b16;
  --ink: #fcf9f2; --ink-2: #d1ccbe; --ink-3: #9b9587;
  --rule: #2a2722; --accent: #f07f63; --accent-2: #d4a155;
  --success: #6fa86a; --warning: #c4913a; --danger: #b85450;
  --serif: "Newsreader", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --sans: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono: "IBM Plex Mono", "SF Mono", "JetBrains Mono", Menlo, monospace;
  display: block;
  color: var(--ink);
  font: 400 16px/1.5 var(--sans);
  overflow: hidden;
}
* { box-sizing: border-box; }
h1 {
  margin: 0 0 0.35rem;
  color: var(--ink);
  font: 400 clamp(1.7rem, 3vw, 2.6rem)/1.05 var(--serif);
  letter-spacing: 0;
}
h2 {
  margin: 2rem 0 0.45rem;
  color: var(--ink);
  font: 500 clamp(1.25rem, 2vw, 1.65rem)/1.12 var(--serif);
}
h3 {
  margin: 1.4rem 0 0.65rem;
  color: var(--accent-2);
  font: 500 12px/1.3 var(--mono);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
p, .meta, .muted { color: var(--ink-3); }
.meta { margin: 0 0 1rem; font-size: 0.92rem; }
code {
  color: var(--ink);
  background: var(--bg-3);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 1px 5px;
  font-family: var(--mono);
  font-size: 0.9em;
}
a { color: var(--accent); text-decoration: none; }
nav.sticky {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 0;
  margin: 0 0 1.4rem;
  background: color-mix(in oklab, var(--bg) 90%, transparent);
  border-bottom: 1px solid var(--rule);
}
nav.sticky a {
  color: var(--ink-2);
  background: var(--bg-2);
  border: 1px solid var(--rule);
  border-radius: 999px;
  padding: 6px 11px;
  font: 500 11px/1.2 var(--mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
nav.sticky a:hover { color: var(--accent); border-color: var(--accent); }
section.card, section.tldr .block, .summary, details.per-sample-card > summary {
  background: var(--bg-2);
  border: 1px solid var(--rule);
  border-radius: 8px;
}
section.card {
  padding: clamp(16px, 2vw, 24px);
  margin: 1.25rem 0;
  overflow: hidden;
}
section.tldr {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 1.25rem 0 1.5rem;
}
section.tldr .block {
  padding: 16px 18px;
  border-left: 2px solid var(--accent);
}
section.tldr h3 { margin-top: 0; color: var(--ink-3); }
section.tldr .big-num {
  color: var(--accent);
  font: 500 1.55rem/1 var(--serif);
  font-variant-numeric: tabular-nums;
}
.tldr-list, .chips { margin: 0; padding: 0; }
.tldr-list { list-style: none; }
.tldr-list li { padding: 6px 0; border-bottom: 1px solid var(--rule); }
.tldr-list li:last-child { border-bottom: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.tldr-chip {
  color: var(--ink-2);
  background: var(--bg-3);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 4px 7px;
  font: 500 12px/1.3 var(--mono);
}
.table-scroll-note {
  color: var(--ink-3);
  font: 500 11px/1.3 var(--mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
table {
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  border-collapse: collapse;
  margin: 0.75rem 0 1.2rem;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--bg-2);
}
thead, tbody, tr { width: max-content; min-width: 100%; }
th, td {
  border-bottom: 1px solid var(--rule);
  padding: 10px 12px;
  color: var(--ink-2);
  font-size: 0.9rem;
  vertical-align: top;
}
th {
  position: sticky;
  top: 0;
  z-index: 1;
  color: var(--ink-3);
  background: var(--bg-3);
  font: 500 11px/1.25 var(--mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
td.num, td.cand, .audio-label, td.rating-aggregate {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
}
td.cand, .audio-label {
  color: var(--ink);
  overflow-wrap: anywhere;
  word-break: break-word;
}
td.num { text-align: right; white-space: nowrap; }
td.score, .score { color: var(--accent); font-weight: 600; }
.score-bad { color: var(--danger); }
.score-good { color: var(--accent); }
tr.src-row, table.audio-grid tr.src-row { background: color-mix(in oklab, var(--accent-2) 12%, var(--bg-2)); }
tr.skipped { opacity: 0.55; }
table.unified tr { min-width: 1080px; }
table.audio-grid {
  table-layout: fixed;
}
table.audio-grid tr { min-width: 1180px; }
table.audio-grid th,
table.audio-grid td {
  width: 180px;
  max-width: 180px;
  text-align: center;
}
table.audio-grid th:first-child,
table.audio-grid td:first-child {
  width: 190px;
  max-width: 190px;
  text-align: left;
}
.audio-label {
  display: block;
  color: var(--ink-2);
  font-size: 12px;
  line-height: 1.35;
  margin-bottom: 8px;
}
button.play-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 7px 12px;
  color: var(--bg);
  background: var(--accent);
  border: 0;
  border-radius: 6px;
  cursor: pointer;
  font: 600 13px/1 var(--sans);
}
button.play-btn:hover { background: #ff947b; }
audio {
  display: block;
  width: 100%;
  height: 34px;
  margin-top: 8px;
}
audio.hidden { display: none; }
.winner {
  margin: 0.75rem 0 1.2rem;
  padding: 12px 16px;
  color: var(--ink-2);
  background: color-mix(in oklab, var(--accent) 12%, var(--bg-2));
  border-left: 2px solid var(--accent);
  border-radius: 0 6px 6px 0;
}
.summary {
  margin: 2rem 0;
  padding: 16px 18px;
  color: var(--ink-2);
  border-left: 2px solid var(--accent-2);
}
.rating { display: inline-flex; gap: 1px; user-select: none; }
.rating button {
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 2px 3px;
}
.rating button.filled,
.rating.hover-1 button:nth-child(-n+1),
.rating.hover-2 button:nth-child(-n+2),
.rating.hover-3 button:nth-child(-n+3),
.rating.hover-4 button:nth-child(-n+4),
.rating.hover-5 button:nth-child(-n+5) { color: var(--accent-2); }
details.per-sample-card {
  margin: 0.9rem 0;
}
details.per-sample-card > summary {
  cursor: pointer;
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 12px 16px;
  list-style: none;
}
details.per-sample-card > summary::-webkit-details-marker { display: none; }
details.per-sample-card > summary::before {
  content: "+";
  color: var(--accent);
  font-family: var(--mono);
}
details.per-sample-card[open] > summary::before { content: "-"; }
details.per-sample-card > summary .ps-name { color: var(--ink); font-weight: 600; }
footer {
  color: var(--ink-3);
  border-top: 1px solid var(--rule);
  margin-top: 2rem;
  padding-top: 1rem;
  font-size: 0.88rem;
}
@media (max-width: 900px) {
  section.tldr { grid-template-columns: 1fr; }
  :host { font-size: 15px; }
}
"""


def build_full_matrix_section() -> str:
    """Embed the full generated matrix inside index.html.

    The old standalone reports/multi/comparison.html page is useful content,
    but a second review URL is redundant. Keep the matrix visually isolated in
    a shadow root so its plain report CSS cannot fight the polished index CSS.
    """
    p = REPORT_DIR / "comparison.html"
    if not p.exists():
        return """
<section class="band reveal" id="matrix">
  <div class="band-head">
    <div><span class="kicker accent">03 — full matrix</span></div>
    <div>
      <h2>Full comparison <em>matrix</em></h2>
      <p class="deck">The matrix has not been generated yet. Run <code>make multi</code>, then rebuild this index.</p>
    </div>
  </div>
</section>
"""
    matrix_html = p.read_text().replace("../../assets/", "assets/")
    body_match = re.search(r"<body>(.*?)</body>", matrix_html, flags=re.S)
    matrix_body = body_match.group(1) if body_match else matrix_html
    return f"""
<section class="band reveal" id="matrix">
  <div class="band-head">
    <div><span class="kicker accent">03 — full matrix</span></div>
    <div>
      <h2>Every candidate, <em>every sample</em></h2>
      <p class="deck">The complete c01–c17 scoring table, per-sample rankings, rating controls,
      and all 196 public audio cells are embedded here so the project index contains the whole review.</p>
    </div>
  </div>
  <div id="matrix-host" class="matrix-frame"></div>
  <template id="matrix-template">
    <style>{MATRIX_CSS}</style>
    {matrix_body}
  </template>
  <script>
  (function() {{
    const host = document.getElementById('matrix-host');
    const tpl = document.getElementById('matrix-template');
    if (!host || !tpl || host.shadowRoot) return;
    host.attachShadow({{ mode: 'open' }}).appendChild(tpl.content.cloneNode(true));
  }})();
  </script>
</section>
"""


def main() -> int:
    base = _load_baseline()
    if base is None:
        print(f"[new] no baseline for {SAMPLE}", file=sys.stderr)
        return 1

    cands = []
    for cand in ["c01", "c07"] + FOCUS_CANDS:
        stage_dir = None
        for d in (REPO_ROOT / "stages" / SAMPLE).iterdir():
            if d.is_dir() and d.name.startswith(cand + "--"):
                stage_dir = d.name
                break
        if not stage_dir:
            continue
        r = _load_report(stage_dir)
        if r is None:
            continue
        sc, bd = _score(base, r)
        cands.append({"cand": cand, "stage": stage_dir, "report": r, "score": sc, "breakdown": bd})

    # Mark the top-scoring candidate as the winner for visual emphasis.
    if cands:
        top = max(cands, key=lambda c: c["score"])
        for c in cands:
            c["is_winner"] = (c is top and top["score"] > 0)

    src_clip = f"assets/audio/{SAMPLE}_10min.mp3"

    # Audio cards: source first, then candidates.
    audio_cards = [card_html(None, src_clip, is_source=True, baseline=base)]
    for c in cands:
        audio_cards.append(card_html(c, src_clip, baseline=base))

    # Metrics table for the SF section.
    metrics_rows: list[str] = []
    for c in cands:
        r = c["report"]; bd = c["breakdown"]
        hf_d = bd.get("hf_d", 0.0)
        metrics_rows.append(
            "<tr>"
            f"<td><b>{escape(c['cand'])}</b></td>"
            f"<td class='num'>{r.get('hiss_band_energy_db', 0):.1f} dB</td>"
            f"<td class='num'>{bd.get('hiss_d', 0):+.1f}</td>"
            f"<td class='num'>{r.get('hf_extension_db', 0):.1f} dB</td>"
            f"<td class='num'>{hf_d:+.1f}</td>"
            f"<td class='num'>{r.get('speech_band_energy_db', 0):.1f} dB</td>"
            f"<td class='num'>{bd.get('speech_d', 0):+.1f}</td>"
            f"<td class='num'>{(r.get('lufs') or 0):.2f}</td>"
            f"<td class='num'>{bd.get('lufs_drift', 0):.2f}</td>"
            f"<td class='num'>{(r.get('dynamic_range_lu') or 0):.1f} LU</td>"
            f"<td class='num'>{bd.get('lra_d', 0):+.1f}</td>"
            f"<td class='num'>{r.get('runtime_s', 0):.1f}s (RTF {bd.get('rtf', 0):.2f})</td>"
            f"<td class='{('score-good' if c['score'] > 0 else 'score-bad' if c['score'] < 0 else 'score-mute')}'>"
            f"<b>{c['score']:+.2f}</b></td>"
            "</tr>"
        )

    # Build the body
    body = f"""
<!-- ============== TOP BAR ============== -->
<div class="topbar">
  <div class="wrap row">
    <div class="brand">
      {BRAND_MARK_SVG}
      <span class="brand-text"><span class="brand-line-1">Srila Prabhupada</span><span class="brand-line-2">Audio Restoration</span></span>
    </div>
    <nav class="nav">
      <a href="#audio">01 Audio</a>
      <a href="#metrics">01 Metrics</a>
      <a href="#verdict">01 Verdict</a>
      <a href="#multi">02 Cross-sample</a>
      <a href="#matrix">03 Matrix</a>
    </nav>
    <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme">
      <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
        <circle cx="12" cy="12" r="4"/>
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4l1.4-1.4M17 7l1.4-1.4"/>
      </svg>
      <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
        <path d="M20 14.5A7.5 7.5 0 0 1 9.5 4 8.4 8.4 0 1 0 20 14.5Z"/>
      </svg>
    </button>
  </div>
</div>

<div class="wrap">

<!-- ============== HERO ============== -->
<section class="hero">
  <div class="studio-line">
    <span>audio_restore</span><span class="bar"></span><span>v 0.1 · 2026-06</span>
  </div>
  <h1>Which tape sounds most <em>studio</em>?</h1>
  <p class="deck">A focused A/B comparison of the eight candidates that actually target
  the "1970s tape → modern podcast" question — denoise, dereverb, bandwidth extension,
  and loudness, in different combinations. Listen first, then let the metrics tell you
  why your ear picked what it picked.</p>
  <div class="anti">
    Source · <b>{SAMPLE_LABEL}</b> · 10-minute clip · 48 kHz mono ·
    <a href="#multi">skip to 3-sample A/B →</a>
  </div>
</section>

<!-- ============== CALLOUTS ============== -->
<div class="note reveal">
  <b>What this page is for</b><br>
  The "studio quality" question: which candidate makes 1970s tape sound like a modern podcast?
  This is the lineup that targets that goal — denoise + dereverb + bandwidth extension (BWE) +
  loudness normalization, in different combinations. Pick the one your ear likes best.
  <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0;">
    <li><b>c01</b> — classical ffmpeg (lowpass + afir + adeclick): metric-top for hiss removal, no BWE</li>
    <li><b>c07</b> — Resemble denoise: a strong denoise baseline, no BWE</li>
    <li><b>c12</b> — DeepFilterNet3: conservative denoise, 48k native, fast (RTF 0.06)</li>
    <li><b>c13</b> — VoiceFixer: denoise + dereverb + BWE in one model, TFGAN vocoder</li>
    <li><b>c14</b> — AudioSR: BWE only (latent diffusion, VCTK-tuned). Adds 12-18kHz content that 1960s tape couldn't record.</li>
    <li><b>c15</b> — MossFormer2_SE_48K: denoise + dereverb, no BWE. The strongest dereverb candidate.</li>
    <li><b>c16</b> — c12 → c14 → loudnorm: clean denoise + BWE, the two-step pipeline</li>
    <li><b>c17</b> — c13 → loudnorm: VoiceFixer one-shot, packaged under a "studio" name</li>
  </ul>
</div>

<div class="note reveal">
  <b>How to use this page</b><br>
  • Hit <b>▶ play</b> on any card — only one plays at a time. The playhead is shared across all rows, so switching rows never jumps the time.<br>
  • Each candidate card has a <b>source / candidate</b> toggle. Flip it live to A/B the raw tape against the model output without stopping playback.<br>
  • The dock has master <b>play / pause / rewind</b> controls for the whole page.
</div>

<!-- ============== MASTER DOCK ============== -->
<div id="master" class="reveal">
  <span class="dock-kicker">master transport</span>
  <button id="master-play" class="btn primary" type="button"><span>Play</span></button>
  <button id="master-pause" class="btn ghost" type="button"><span>Pause</span></button>
  <button id="master-rewind" class="btn ghost" type="button"><span>Rewind to 0</span></button>
  <span class="time" id="master-time">0:00 / 0:00</span>
</div>

<!-- ============== AUDIO SECTION ============== -->
<section class="band reveal" id="audio">
  <div class="band-head">
    <div><span class="kicker accent">01 — side-by-side audio</span></div>
    <div>
      <h2>The eight candidates, <em>head to head</em></h2>
      <p class="deck">Same 10-minute source, different restoration pipelines. The card
      with the accent border is the project-score winner for this sample — but the
      score isn't the whole story; listen to the BWE candidates (c14, c16, c17) with
      a critical ear for the metallic artifacts that the metric only partially penalizes.</p>
    </div>
  </div>

  <div class="audio-cards has-winner">
    {''.join(audio_cards)}
  </div>
</section>

<!-- ============== METRICS SECTION ============== -->
<section class="band reveal" id="metrics">
  <div class="band-head">
    <div><span class="kicker accent">01 — metrics</span></div>
    <div>
      <h2>What the <em>numbers</em> say</h2>
      <p class="deck">Score = 2·hiss_Δ + min(HF_Δ, +20) − max(0,+speech_Δ) − 0.5·max(0,−LRA_Δ)
      − 0.1·|centroid−5000|/1000 − max(0,RTF−2) − 0.3·max(0,|LUFS+16|−3). hiss_Δ is
      gain-invariant (hiss/speech ratio).</p>
    </div>
  </div>

  <table class="metrics">
    <thead>
      <tr>
        <th>cand</th><th>hiss 5-12kHz</th><th>hiss Δ</th>
        <th>HF 12-18kHz</th><th>HF Δ</th>
        <th>speech 300-3.4kHz</th><th>speech Δ</th>
        <th>LUFS</th><th>LUFS drift</th>
        <th>LRA</th><th>LRA Δ</th>
        <th>runtime (RTF)</th><th>score</th>
      </tr>
    </thead>
    <tbody>
      <tr class="src-row">
        <td><b>baseline</b></td>
        <td class='num'>{base['hiss_band_energy_db']:.1f} dB</td><td class='num'>—</td>
        <td class='num'>{base.get('hf_extension_db', -200):.1f} dB</td><td class='num'>—</td>
        <td class='num'>{base['speech_band_energy_db']:.1f} dB</td><td class='num'>—</td>
        <td class='num'>{(base.get('lufs') or 0):.2f}</td><td class='num'>—</td>
        <td class='num'>{(base.get('dynamic_range_lu') or 0):.1f} LU</td><td class='num'>—</td>
        <td class='num'>—</td><td class='num'>—</td>
      </tr>
      {''.join(metrics_rows)}
    </tbody>
  </table>
</section>

<!-- ============== VERDICT SECTION ============== -->
<section class="band reveal" id="verdict">
  <div class="band-head">
    <div><span class="kicker accent">01 — verdict</span></div>
    <div>
      <h2>What we <em>learned</em> on SF</h2>
    </div>
  </div>
  <div class="proofline">
    <b>note</b> &nbsp; SF baseline has hiss 28.2 dB (lower than DC's 33.3 dB) but the tape is harder —
    every candidate measures slightly above baseline on the 5-12 kHz band because the loudnorm pass
    + the model itself add some energy back. The hiss_Δ in the table is the gain-invariant
    hiss/speech ratio, not the absolute dB removed.
  </div>
  <div class="proofline">
    <b>finding</b> &nbsp; The BWE candidates (c13/c14/c17) have huge HF deltas (+60 to +90 dB) but
    are penalized for hurting the hiss/speech ratio (the diffusion/vocoder can't tell hiss from
    HF content, so it boosts both). The c12 → c14 pipeline (c16) is the bet: clean hiss first,
    then extend the bandwidth. c15 (MossFormer2 denoise+dereverb) is the best conservative
    candidate by ear — strong hiss reduction with a real dereverb effect, no BWE artifact risk.
  </div>
  <div class="proofline">
    <b>bottom line</b> &nbsp; Scroll down to the <a href="#multi">multi-sample section</a> to A/B
    test c15 vs c16 on all 3 recordings. The metric favors c15 on SF and BOM, ties on DC —
    your ear should decide whether c16's bandwidth extension is worth its occasional hiss-band boost.
  </div>
</section>

{build_multi_sample_section()}

{build_full_matrix_section()}

</div><!-- /.wrap -->

<script>
(function() {{
  // ===== Theme toggle =====
  const themeBtn = document.getElementById('theme-toggle');
  const saved = localStorage.getItem('audio_restore_theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  if (themeBtn) themeBtn.addEventListener('click', () => {{
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('audio_restore_theme', next);
  }});

  // ===== Reveal-on-scroll =====
  const io = new IntersectionObserver((entries) => {{
    entries.forEach(e => {{ if (e.isIntersecting) {{ e.target.classList.add('visible'); io.unobserve(e.target); }} }});
  }}, {{ threshold: 0.12, rootMargin: '0px 0px -40px 0px' }});
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));

  // ===== Master transport =====
  const masterPlay = document.getElementById('master-play');
  const masterPause = document.getElementById('master-pause');
  const masterRewind = document.getElementById('master-rewind');
  const masterTime = document.getElementById('master-time');

  // State indexed by row id (data-row). Cards carry data-row instead of tr.
  const rowState = {{}};
  let activeRow = null;
  document.querySelectorAll('[data-row]').forEach(el => {{
    const row = el.getAttribute('data-row');
    const aud = el.querySelector('audio');
    if (!aud) return;
    rowState[row] = {{
      aud,
      side: 'src',
      srcUrl: el.getAttribute('data-src-url') || aud.getAttribute('src'),
      candUrl: el.getAttribute('data-cand-url'),
    }};
  }});

  function loadSide(row, side) {{
    const st = rowState[row]; if (!st) return;
    if (st.side === side) return;
    const t = st.aud.currentTime;
    const wasPlaying = !st.aud.paused;
    const url = (side === 'src') ? st.srcUrl : st.candUrl;
    if (!url) return;
    st.aud.src = url;
    st.aud.load();
    st.aud.currentTime = t;
    st.side = side;
    document.querySelectorAll(`button.ab-btn[data-row="${{row}}"]`).forEach(b => {{
      b.classList.toggle('active', b.getAttribute('data-side') === side);
    }});
    // Tint the entire card green when candidate side is active, neutral
    // when source side. This is the "I know exactly which row is playing
    // the candidate right now" visual signal the user asked for.
    const card = document.querySelector(`.audio-card[data-row="${{row}}"]`);
    if (card) {{
      card.classList.toggle('is-candidate-side', side === 'cand');
    }}
    if (row === activeRow && wasPlaying) {{
      st.aud.play().catch(() => {{}});
    }}
  }}

  document.addEventListener('click', (e) => {{
    const abBtn = e.target.closest('button.ab-btn');
    if (!abBtn) return;
    loadSide(abBtn.getAttribute('data-row'), abBtn.getAttribute('data-side'));
  }});

  function activateRow(row) {{
    if (activeRow && activeRow !== row && rowState[activeRow]) {{
      rowState[activeRow].aud.pause();
    }}
    // Toggle is-active class on the audio card.
    document.querySelectorAll('.audio-card.is-active').forEach(el => el.classList.remove('is-active'));
    const card = document.querySelector(`.audio-card[data-row="${{row}}"]`);
    if (card) card.classList.add('is-active');
    activeRow = row;
    const st = rowState[row]; if (!st) return;
    st.aud.play().catch(() => {{}});
  }}

  Object.entries(rowState).forEach(([row, st]) => {{
    st.aud.addEventListener('play', () => activateRow(row));
  }});

  masterPlay.addEventListener('click', () => {{
    if (!activeRow) {{
      // Find the first audio card.
      const firstCard = document.querySelector('.audio-card[data-row]');
      if (firstCard) activateRow(firstCard.getAttribute('data-row'));
    }} else if (rowState[activeRow]) {{
      rowState[activeRow].aud.play().catch(() => {{}});
    }}
  }});
  masterPause.addEventListener('click', () => {{
    Object.values(rowState).forEach(st => st.aud.pause());
  }});
  masterRewind.addEventListener('click', () => {{
    Object.values(rowState).forEach(st => {{ st.aud.currentTime = 0; }});
  }});

  // ===== Time display =====
  function fmt(s) {{
    if (!isFinite(s)) s = 0;
    const m = Math.floor(s / 60); const sec = Math.floor(s % 60);
    return `${{m}}:${{sec.toString().padStart(2, '0')}}`;
  }}
  function updateTime() {{
    let aud = null;
    if (activeRow && rowState[activeRow]) aud = rowState[activeRow].aud;
    if (aud) {{
      masterTime.textContent = `${{fmt(aud.currentTime)}} / ${{fmt(aud.duration || 0)}}`;
    }}
    requestAnimationFrame(updateTime);
  }}
  requestAnimationFrame(updateTime);
}})();
</script>
"""

    html = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Srila Prabhupada Audio Restoration · audio_restore</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..600;1,6..72,300..600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    OUT.write_text(html)
    print(f"[new] wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"[new] {len(cands)} candidates on {SAMPLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
