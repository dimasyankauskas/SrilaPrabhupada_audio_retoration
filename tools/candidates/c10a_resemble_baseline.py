#!/usr/bin/env python3
"""
c10a_resemble_baseline — replicate c07's defaults exactly.

Knobs: chunk_seconds=30, overlap_seconds=1, preemphasis=0.97.
Use as the control: any difference vs. c07 should be ~0 (random seed
variance only).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the candidates dir to sys.path so the underscore-prefixed helper
# is importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _c10_core as core


if __name__ == "__main__":
    # Stage id parsed from filename ("c10a") so core.run() can look up
    # its preset. Stage name appears in the dir path on disk.
    stage_id = Path(__file__).stem.split("_", 1)[0]
    stage_name = Path(__file__).stem.split("_", 1)[1] if "_" in Path(__file__).stem else stage_id
    sys.exit(core.run(stage_id, stage_name))
