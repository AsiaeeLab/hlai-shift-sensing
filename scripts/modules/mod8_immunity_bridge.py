"""Module 8: Cohort Provenance, Symmetric Kitagawa-Oaxaca Decomposition, and External PRIDE Bridge."""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from presentation_state.config import PathResolver, resolve_config  # noqa: E402


def run(cfg: PathResolver | None = None) -> dict:
    """Copy or compute Module 8 outputs into cfg.results('exp8_immunity_bridge')."""
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp8_immunity_bridge")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If canonical repo script exists alongside staged external dataset tables, delegate or sync cleanly
    repo_cand = SCRIPT_DIR.parents[1] / "code" / "salvage" / "exp8_immunity_strengthening_and_external_bridge.py"
    canon_res = SCRIPT_DIR.parents[1] / "results" / "salvage" / "exp8_immunity_bridge"
    if (out_dir / "summary.json").exists():
        return json.loads((out_dir / "summary.json").read_text())
    if canon_res.exists() and (canon_res / "summary.json").exists() and canon_res.resolve() != out_dir.resolve():
        import shutil
        for fp in canon_res.iterdir():
            if fp.is_file():
                shutil.copy2(fp, out_dir / fp.name)
        return json.loads((out_dir / "summary.json").read_text())
    if repo_cand.exists():
        subprocess.run([sys.executable, str(repo_cand)], check=True)
        if canon_res.resolve() != out_dir.resolve():
            import shutil
            for fp in canon_res.iterdir():
                if fp.is_file():
                    shutil.copy2(fp, out_dir / fp.name)
        return json.loads((out_dir / "summary.json").read_text())
    raise FileNotFoundError(f"Missing Module 8 summary at {out_dir / 'summary.json'}")
