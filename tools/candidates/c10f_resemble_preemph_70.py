#!/usr/bin/env python3
"""
c10f_resemble_preemph_70 — c07 with preemphasis=0.70 (vs default 0.97).

Aggressive push on the pre-emphasis knob: a much flatter input spectrum.
If 0.85 helps, 0.70 might help more — or might break it entirely.
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
