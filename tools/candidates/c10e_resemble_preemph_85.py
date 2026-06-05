#!/usr/bin/env python3
"""
c10e_resemble_preemph_85 — c07 with preemphasis=0.85 (vs default 0.97).

Pre-emphasis is the high-pass filter applied to the input before the
STFT. Lower value = flatter input spectrum. The model was trained with
0.97, so moving away from that may degrade quality — but on tape audio
the input is already bright, so reducing pre-emphasis may help.
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
