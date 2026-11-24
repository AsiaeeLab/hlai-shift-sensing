"""Module 9: Positional Ablations (P1-P9), 9x19/9x20 Delta-PSSM Matrices,
Multi-Allele HLA Supertype Convergence (JSD), and HLA-A/B/C Affinity Diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from presentation_state.config import PathResolver, resolve_config  # noqa: E402


def run(cfg: PathResolver | None = None) -> dict:
    """Load or compute Module 9 outputs into cfg.results('exp9_post_meeting')."""
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp9_post_meeting")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_cand = SCRIPT_DIR.parents[1] / "code" / "salvage" / "exp9_post_meeting_ablations_logos_hla.py"
    canon_res = SCRIPT_DIR.parents[1] / "results" / "salvage" / "exp9_post_meeting"
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
    raise FileNotFoundError(f"Missing Module 9 summary at {out_dir / 'summary.json'}")
