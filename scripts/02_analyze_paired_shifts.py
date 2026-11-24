#!/usr/bin/env python3
"""Run paired compositional sensing, multi-omic flux, routing, gatekeeper, and persistence analyses.

Usage:
    python3 scripts/02_analyze_paired_shifts.py --data-dir ./data --processed-dir ./data/processed --results-dir ./results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from presentation_state.config import resolve_config  # noqa: E402
from modules import (  # noqa: E402
    mod1_concentration_decoupling,
    mod2_monte_degradative_flux,
    mod3_inflammatory_routing,
    mod4_hap1_gatekeepers,
    mod5_persistence_prediction,
    mod7_distributional_sensing,
    mod8_immunity_bridge,
    mod9_positional_anatomy,
)

MODULE_RUNNERS = {
    "1": ("Module 1 (Concentration Decoupling)", mod1_concentration_decoupling.run),
    "2": ("Module 2 (MONTE Degradative Flux & APPM)", mod2_monte_degradative_flux.run),
    "3": ("Module 3 (Inflammatory Routing & Censoring)", mod3_inflammatory_routing.run),
    "4": ("Module 4 (HAP1 11-KO & Sec61 Bypass)", mod4_hap1_gatekeepers.run),
    "5": ("Module 5 (LOBO Persistence Prediction)", mod5_persistence_prediction.run),
    "7": ("Module 7 (Distributional Sensing & Transfer)", mod7_distributional_sensing.run),
    "8": ("Module 8 (35-Contrast Roster, Kitagawa-Oaxaca & External Bridge)", mod8_immunity_bridge.run),
    "9": ("Module 9 (Positional Ablations, Delta-PSSM & Supertype JSD)", mod9_positional_anatomy.run),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--upstream-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--modules", default="1,2,3,4,5,7,8,9", help="Comma-separated module IDs to execute.")
    args = parser.parse_args()

    cfg = resolve_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        upstream_dir=args.upstream_dir,
        results_dir=args.results_dir,
    )
    selected = [m.strip() for m in args.modules.split(",") if m.strip()]
    for m_id in selected:
        if m_id not in MODULE_RUNNERS:
            raise ValueError(f"Unknown module id '{m_id}'; valid keys: {sorted(MODULE_RUNNERS)}")
        label, fn = MODULE_RUNNERS[m_id]
        print(f"[run ] {label}...")
        fn(cfg)
        print(f"[done] {label}")


if __name__ == "__main__":
    main()
