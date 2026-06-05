#!/usr/bin/env python3
"""
c10b_resemble_chunk_60 — c07 with chunk_seconds=60 (vs default 30).

Tests whether larger windows (less boundary crossfading) improve quality
or simply reduce compute.
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
