#!/usr/bin/env python3
"""Generate all main figures (1-5, graphical abstract), supplementary figures (S1-S5), and summary tables.

Usage:
    python3 scripts/04_make_figures_tables.py --results-dir ./results --fig-dir ./figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from presentation_state.config import PathResolver, resolve_config  # noqa: E402
from modules import fig_main, fig_supp  # noqa: E402


def _fals_dir(cfg: PathResolver) -> Path:
    d1 = cfg.results("falsification_controls")
    return d1 if d1.exists() else cfg.results("exp6_referee_falsifications")


def export_summary_tables(cfg: PathResolver) -> list[str]:
    tbl_dir = cfg.results("tables")
    tbl_dir.mkdir(parents=True, exist_ok=True)

    elast = pd.read_csv(cfg.results("exp1_concentration_decoupling", "perturbation_elasticity_summary.tsv"), sep="\t")
    f7a = pd.read_csv(cfg.results("exp7_distributional_sensing", "exp7a_followup_elasticity_and_ub.tsv"), sep="\t")
    disatt = pd.read_csv(_fals_dir(cfg) / "disattenuation_and_sibling_null.tsv", sep="\t")

    t1 = elast.merge(
        disatt[["background", "r2_fully_disattenuated", "multiplicity_noise_null_gt2x", "fold_over_noise_null"]],
        on="background",
        how="left",
    )
    t1.to_csv(tbl_dir / "table1_concentration_decoupling.csv", index=False)
    f7a.to_csv(tbl_dir / "table1b_followup_elasticity.csv", index=False)

    t2_cross = pd.read_csv(cfg.results("exp7_distributional_sensing", "exp7c_cross_lineage_study_transfer.tsv"), sep="\t")
    t2_res = pd.read_csv(cfg.results("exp7_distributional_sensing", "exp7c_resistance_and_cross_study_challenge.tsv"), sep="\t")
    t2_il = pd.read_csv(cfg.results("exp7_distributional_sensing", "exp7c_isobaric_il_sensitivity.tsv"), sep="\t")
    t2_cross.to_csv(tbl_dir / "table2a_cross_lineage_representations.csv", index=False)
    t2_res.to_csv(tbl_dir / "table2b_resistance_challenge.csv", index=False)
    t2_il.to_csv(tbl_dir / "table2c_isobaric_il_harmonization.csv", index=False)

    exported = [
        "table1_concentration_decoupling.csv",
        "table1b_followup_elasticity.csv",
        "table2a_cross_lineage_representations.csv",
        "table2b_resistance_challenge.csv",
        "table2c_isobaric_il_harmonization.csv",
    ]

    exp8_dir = cfg.results("exp8_immunity_bridge")
    if exp8_dir.exists():
        for fname in [
            "contrast_roster_35.csv",
            "folds_lolo_16lineages.csv",
            "folds_loso_strict_5studies.csv",
            "study_class_support_and_within_study_metrics.csv",
            "cross_transfer_all_representations_db_vs_il.csv",
            "kitagawa_oaxaca_locus_vs_motif_decomposition.csv",
            "kitagawa_oaxaca_per_contrast_35.csv",
            "external_frozen_decoder_bridge_results.csv",
            "external_aranha2026_pxd085506_proteome_state.csv",
            "external_aranha2026_pxd085470_repertoire_coverage.csv",
            "external_aranha2026_signal_span_plasticity.csv",
            "skmel28_mock_adjusted_did.csv",
            "skmel28_predictions_with_il_and_unseen_9mer_purge.csv",
            "tableS12_newey_unpaired_vs_paired_all11.csv",
            "tableS13_persistence_gene_disjoint_all12.csv",
            "persistence_endpoint_prevalence_and_comparators.csv",
            "persistence_factorial_specification_control.csv",
            "peptide_budget_curve_nested_vs_fixed_C.csv",
            "velz_ari_all_pairs.csv",
            "hap1_exact_signal_span_cleavage_audit.tsv",
        ]:
            src_f = exp8_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, tbl_dir / fname)
                exported.append(fname)

    exp9_dir = cfg.results("exp9_post_meeting")
    if exp9_dir.exists():
        for fname in [
            "positional_ablation_and_subsets.csv",
            "pssm_shift_and_decoder_logo_matrices.csv",
            "pssm_within_locus_ifng_shifts.csv",
            "baseline_supertype_convergence_jsd.csv",
            "hla_abc_affinity_diagnostics.csv",
            "hla_abc_share_shifts_35contrasts.csv",
        ]:
            src_f = exp9_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, tbl_dir / fname)
                exported.append(fname)

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = resolve_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        fig_dir=args.fig_dir,
    )
    main_manifest = fig_main.run(cfg.results_dir, cfg.fig_dir)
    supp_pdfs = fig_supp.run(cfg.results_dir, cfg.fig_dir)
    tables = export_summary_tables(cfg)
    print(json.dumps({"main": main_manifest, "supp_pdfs": supp_pdfs, "tables": tables}, indent=2))


if __name__ == "__main__":
    main()
