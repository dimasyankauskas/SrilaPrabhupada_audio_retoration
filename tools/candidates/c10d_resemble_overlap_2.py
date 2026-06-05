#!/usr/bin/env python3
"""
c10d_resemble_overlap_2 — c07 with overlap_seconds=2 (vs default 1).

Tests whether a longer crossfade between chunks reduces the audible
boundary seams. Doubles the compute per chunk but should not change
the model behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _c10_core as core


if __name__ == "__main__":
    stage_id = Path(__file__).stem.split("_", 1)[0]
    stage_name = Path(__file__).stem.split("_", 1)[1] if "_" in Path(__file__).stem else stage_id
    sys.exit(core.run(stage_id, stage_name))
