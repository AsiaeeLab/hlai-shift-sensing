#!/usr/bin/env python3
"""Independent numerical, mathematical, metadata, Kitagawa-Oaxaca, external-bridge,
and figure-hygiene consistency verifier for the standalone executable deposit.

Performs deterministic checks against staged cohort tables, Proposition 1 algebraic
rank calculations, Module 1-8 outputs, falsification controls, and generated figures.
Exits non-zero on failure.

Usage:
    python3 scripts/05_verify_results.py --data-dir ./data --processed-dir ./data/processed --results-dir ./results --fig-dir ./figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from presentation_state import representations  # noqa: E402
from presentation_state.config import PathResolver, resolve_config  # noqa: E402
from presentation_state.manifest_validator import validate_design_manifests  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checks(cfg: PathResolver) -> dict:
    checks = []

    def record(name: str, condition: bool, observed, expected) -> None:
        checks.append({"check": name, "passed": bool(condition), "observed": observed, "expected": expected})

    # 1. Preprocessed manifests (when present under PROCESSED_DIR)
    ann_man = cfg.annotations("manifest.json")
    if ann_man.exists():
        m1 = json.loads(ann_man.read_text())
        obs_t = sha256_file(cfg.annotations("gene_topology_summary.tsv.gz"))
        obs_h = sha256_file(cfg.annotations("monte_hlaii_strict.tsv.gz"))
        record("staged_gene_topology_sha256", obs_t == m1["topology"]["gene_topology_sha256"], obs_t, m1["topology"]["gene_topology_sha256"])
        record("staged_monte_hlaii_strict_sha256", obs_h == m1["monte_hlaii"]["monte_hlaii_strict_sha256"], obs_h, m1["monte_hlaii"]["monte_hlaii_strict_sha256"])

    fup_man = cfg.followup("audit_manifest.json")
    if fup_man.exists():
        m2 = json.loads(fup_man.read_text())
        obs_s = sha256_file(cfg.followup("stopfer2022_peptides.tsv.gz"))
        obs_b = sha256_file(cfg.followup("bernhardt2025_psilac_peptides.tsv.gz"))
        record("staged_stopfer2022_peptides_sha256", obs_s == m2["stopfer2022"]["peptides_sha256"], obs_s, m2["stopfer2022"]["peptides_sha256"])
        record("staged_bernhardt2025_peptides_sha256", obs_b == m2["bernhardt2025"]["peptides_sha256"], obs_b, m2["bernhardt2025"]["peptides_sha256"])

    # 2. Proposition 1 rank 67 / affine dimension 66 & exact GROUPS partition
    rank_phi, affine_dim = representations.geometry_check()
    record("proposition1_linear_rank_67", rank_phi == 67, rank_phi, 67)
    record("proposition1_simplex_affine_dim_66", affine_dim == 66, affine_dim, 66)
    record("exact_representations_groups", representations.GROUPS == ("AILMVFWYP", "CNQSTG", "DE", "RHK"), representations.GROUPS, ("AILMVFWYP", "CNQSTG", "DE", "RHK"))

    # 3. Ground-truth cohort guards
    res_root = cfg.results_dir
    velz_n = len(pd.read_csv(res_root / "exp1_concentration_decoupling/velz_microenvironment_patient_scores.tsv", sep="\t"))
    hap1_samps = pd.read_csv(cfg.processed("visibility_hap1_gene_unique", "samples.tsv"), sep="\t")
    tap1_date = str(hap1_samps[(hap1_samps["condition"] == "TAP1_KO") & (hap1_samps["modality"] == "peptide")]["run_dates"].iloc[0])
    chong_samps = pd.read_csv(cfg.processed("visibility_replication", "chong2018_gene_unique", "samples.tsv"), sep="\t")
    chong_n_trt = int(((chong_samps["included"] == True) & (chong_samps["modality"] == "peptide") & (chong_samps["condition"] == "IFNG")).sum())

    record("meta_velz_N28_exact", velz_n == 28, velz_n, 28)
    record("meta_hap1_run_date_20201127", tap1_date == "20201127", tap1_date, "20201127")
    record("meta_chong_n4_replicates", chong_n_trt == 4, chong_n_trt, 4)

    # 4. Module 1 checks
    e1_bench = pd.read_csv(res_root / "exp1_concentration_decoupling/decoding_benchmark_synthesis.tsv", sep="\t")
    e1_elast = pd.read_csv(res_root / "exp1_concentration_decoupling/perturbation_elasticity_summary.tsv", sep="\t")
    e1_sum = json.loads((res_root / "exp1_concentration_decoupling/summary.json").read_text())

    ba_loo = float(e1_bench.iloc[0]["primary_metric"])
    pair_conc = float(e1_bench.iloc[1]["primary_metric"])
    ari_rna_meth = float(e1_bench[e1_bench["representation"] == "RNA vs methylation"]["primary_metric"].iloc[0])
    ari_pep_rna = float(e1_bench[e1_bench["representation"] == "RNA vs coarse96"]["primary_metric"].iloc[0])
    r2_min_pct = 100.0 * float(e1_elast["r_squared"].min())
    r2_max_pct = 100.0 * float(e1_elast["r_squared"].max())
    ifng_elast = e1_elast[e1_elast["perturbation"] == "IFNG"]
    div_min_pct = 100.0 * float(ifng_elast["frac_genes_diverging_gt_2fold"].min())
    div_max_pct = 100.0 * float(ifng_elast["frac_genes_diverging_gt_2fold"].max())

    record("mod1_newey_loo_ba", abs(ba_loo - 0.500) < 1e-6, round(ba_loo, 4), 0.500)
    record("mod1_newey_paired_concordance", abs(pair_conc - 0.984375) < 1e-6, round(pair_conc, 4), 0.9844)
    record("mod1_velz_rna_methylation_ari", abs(ari_rna_meth - 0.6538) < 1e-3, round(ari_rna_meth, 4), 0.6538)
    record("mod1_velz_peptide_rna_ari", abs(ari_pep_rna - 0.0301) < 1e-3, round(ari_pep_rna, 4), 0.0301)
    record("mod1_velz_angio_hlai_collinearity", abs(e1_sum["velz_microenvironment"]["r_angio_vs_hlai_b2m"] - 0.7875) < 1e-3, round(e1_sum["velz_microenvironment"]["r_angio_vs_hlai_b2m"], 4), 0.7875)
    record("mod1_proteome_r2_bounds", abs(r2_min_pct - 1.115) < 0.05 and abs(r2_max_pct - 15.602) < 0.05, [round(r2_min_pct, 2), round(r2_max_pct, 2)], [1.12, 15.60])
    record("mod1_sibling_divergence_bounds", abs(div_min_pct - 26.69) < 0.1 and abs(div_max_pct - 60.80) < 0.1, [round(div_min_pct, 2), round(div_max_pct, 2)], [26.69, 60.80])

    # 5. Module 2 checks
    e2_reg = pd.read_csv(res_root / "exp2_monte_degradative_flux/across_gene_ptm_regression.tsv", sep="\t")
    e2_flux = pd.read_csv(res_root / "exp2_monte_degradative_flux/within_gene_across_patient_flux.tsv", sep="\t")
    e2_appm = pd.read_csv(res_root / "exp2_monte_degradative_flux/monte_appm_readout_tests.tsv", sep="\t")

    ub_hlai = float(e2_reg[(e2_reg["outcome"] == "log2_hlai_peptides") & (e2_reg["predictor"] == "log2_ub_sites")]["std_beta"].iloc[0])
    noip_hlai = e2_flux[(e2_flux["hla_class"] == "HLA-I") & (e2_flux["workflow"] == "noIP")].iloc[0]
    erap1_r = float(e2_appm[(e2_appm["hypothesis"] == "ERAP1 protein vs mean peptide length") & (e2_appm["workflow"] == "noIP")]["pearson_r"].iloc[0])
    appm_ft_r = float(e2_appm[(e2_appm["hypothesis"] == "APPM core protein vs log2(HLA-I breadth)") & (e2_appm["workflow"] == "FT")]["pearson_r"].iloc[0])

    record("mod2_across_gene_ub_hlai_beta", abs(ub_hlai - 0.1993) < 1e-3, round(ub_hlai, 4), 0.1993)
    record("mod2_within_gene_noip_ub_slope", abs(float(noip_hlai["beta_ub"]) - 0.0464) < 1e-3 and abs(float(noip_hlai["p_ub_gene_clustered"]) - 0.0122) < 5e-4, [round(float(noip_hlai["beta_ub"]), 4), round(float(noip_hlai["p_ub_gene_clustered"]), 4)], [0.0464, 0.0122])
    record("mod2_erap1_length_r", abs(erap1_r - (-0.6783)) < 1e-3, round(erap1_r, 4), -0.6783)
    record("mod2_appm_breadth_ft_r", abs(appm_ft_r - 0.7615) < 1e-3, round(appm_ft_r, 4), 0.7615)

    # 6. Module 3 checks
    e3_cens = pd.read_csv(res_root / "exp3_inflammatory_routing/censoring_bounds_all_programs.tsv", sep="\t")
    e3_sib = pd.read_csv(res_root / "exp3_inflammatory_routing/same_protein_hlab_vs_hlaa_siblings.tsv", sep="\t")
    e3_mot = pd.read_csv(res_root / "exp3_inflammatory_routing/proteolytic_motif_shifts_8_backgrounds.tsv", sep="\t")

    uwb_e2f = e3_cens[(e3_cens["background"] == "UWB1.289") & (e3_cens["program"] == "E2F_TARGETS")].iloc[0]
    ito58_sib = e3_sib[e3_sib["background"] == "ITO58M"].iloc[0]
    uwb_sib = e3_sib[e3_sib["background"] == "UWB1.289"].iloc[0]
    hydro_pos = int((e3_mot["hydro_minus_nonhydro_dPep"] > 0).sum())
    hydro_sig = int(((e3_mot["hydro_minus_nonhydro_dPep"] > 0) & (e3_mot["hydro_mwu_p"] < 0.05)).sum())

    record("mod3_uwb1289_censoring_sign_flip", float(uwb_e2f["theta_survivor"]) > 0.47 and float(uwb_e2f["theta_lod_min_bound"]) < -0.52, [round(float(uwb_e2f["theta_survivor"]), 3), round(float(uwb_e2f["theta_lod_min_bound"]), 3)], [0.478, -0.527])
    record("mod3_same_gene_hlab_divergence", abs(float(ito58_sib["mean_gene_dPep_B_minus_A"]) - 0.9839) < 1e-3 and abs(float(uwb_sib["mean_gene_dPep_B_minus_A"]) - 1.2423) < 1e-3, [round(float(ito58_sib["mean_gene_dPep_B_minus_A"]), 3), round(float(uwb_sib["mean_gene_dPep_B_minus_A"]), 3)], [0.984, 1.242])
    record("mod3_hydrophobic_cterm_7pos_6sig", hydro_pos == 7 and hydro_sig == 6, [hydro_pos, hydro_sig], [7, 6])

    # 7. Module 4 checks
    e4_ko = pd.read_csv(res_root / "exp4_hap1_gatekeepers/hap1_11ko_two_margin_summary.tsv", sep="\t").set_index("knockout")
    e4_ldr = pd.read_csv(res_root / "exp4_hap1_gatekeepers/hap1_leader_signal_peptide_sparing.tsv", sep="\t").set_index("knockout")
    e4_top = pd.read_csv(res_root / "exp4_hap1_gatekeepers/tap_ko_topology_matched_reanalysis.tsv", sep="\t")
    e4_hm2 = pd.read_csv(res_root / "exp4_hap1_gatekeepers/hallmark50_topology_in_tap2_ko.tsv", sep="\t")
    r_tap2_hm = float(np.corrcoef(e4_hm2["frac_signal_peptide_genes"], e4_hm2["gene_mean_retention"])[0, 1])

    erap1_dn = 100.0 * float(e4_ko.loc["ERAP1", "denovo_frac_10_11mer"])
    tap2_fold = float(2.0 ** e4_ldr.loc["TAP2", "surv_leader_minus_internal_log2"])
    t2_emt = e4_top[(e4_top["knockout"] == "TAP2") & (e4_top["program"] == "EPITHELIAL_MESENCHYMAL_TRANSITION")].iloc[0]

    record("mod4_erap1_denovo_10_11mer_pct", abs(erap1_dn - 78.11) < 0.1, round(erap1_dn, 2), 78.11)
    record("mod4_tap2_sec61_sparing_fold", abs(tap2_fold - 43.64) < 0.2, round(tap2_fold, 2), 43.64)
    record("mod4_tap2_hallmark49_sigpep_r", len(e4_hm2) == 49 and abs(r_tap2_hm - 0.688) < 2e-3, round(r_tap2_hm, 4), 0.688)
    record("mod4_tap2_emt_topology_collapse", abs(float(t2_emt["topology_matched_mean_pair_u"]) - 0.230) < 0.01, round(float(t2_emt["topology_matched_mean_pair_u"]), 2), 0.23)

    # 8. Module 5 checks (with exact 0.8686 expected value)
    e5_sum = json.loads((res_root / "exp5_persistence_prediction/summary.json").read_text())
    ret_mech = e5_sum["mean_auroc_by_endpoint"]["retained"]["Mechanistic_Censored"]
    ret_prot = e5_sum["mean_auroc_by_endpoint"]["retained"]["Intracellular_Prot0"]
    hp_mech = e5_sum["mean_auroc_by_endpoint"]["high_persist"]["Mechanistic_Censored"]
    hp_pep0 = e5_sum["mean_auroc_by_endpoint"]["high_persist"]["Baseline_MS_Pep0"]
    dur_aff = e5_sum["mean_auroc_by_endpoint"]["durable"]["MHCflurry_Affinity"]
    top20_mech = 100.0 * e5_sum["mean_top20_portfolio_by_endpoint"]["high_persist"]["Mechanistic_Censored"]

    record("mod5_total_baseline_peptides", e5_sum["n_total_baseline_peptides"] == 6998, e5_sum["n_total_baseline_peptides"], 6998)
    record("mod5_retention_auroc_mech_vs_prot0", abs(ret_mech - 0.7246) < 5e-4 and abs(ret_prot - 0.4546) < 5e-4, [round(ret_mech, 4), round(ret_prot, 4)], [0.7246, 0.4546])
    record("mod5_high_persist_pep0_and_affinity_durability", abs(hp_mech - 0.8716) < 5e-4 and abs(hp_pep0 - 0.8686) < 5e-4 and abs(dur_aff - 0.6351) < 5e-4 and abs(top20_mech - 98.75) < 0.05, [round(hp_mech, 4), round(hp_pep0, 4), round(dur_aff, 4), round(top20_mech, 2)], [0.8716, 0.8686, 0.6351, 98.75])

    # 9. Falsification controls checks
    fc_dir = res_root / "falsification_controls" if (res_root / "falsification_controls").exists() else res_root / "exp6_referee_falsifications"
    fc_sum = json.loads((fc_dir / "summary.json").read_text())
    k_dis = fc_sum.get("disattenuation", fc_sum.get("exp6a_disattenuation"))
    k_mon = fc_sum.get("monte_controls", fc_sum.get("exp6b_monte_controls"))
    k_new = fc_sum.get("newey_external", fc_sum.get("exp6c_newey_external"))
    k_sec = fc_sum.get("hap1_2x2_sec61", fc_sum.get("exp6d_hap1_2x2_sec61"))
    k_per = fc_sum.get("persistence_controls", fc_sum.get("exp6e_persistence_controls"))

    record("fc_disattenuated_r2_bounds", abs(100 * k_dis["disattenuated_r2_range"][0] - 1.25) < 0.1 and abs(100 * k_dis["disattenuated_r2_range"][1] - 17.77) < 0.1, [round(100 * x, 2) for x in k_dis["disattenuated_r2_range"]], [1.25, 17.77])
    record("fc_within_exact_allele_erap1", abs(k_mon["within_exact_allele_erap1_len_r"] - (-0.7452)) < 5e-4, round(k_mon["within_exact_allele_erap1_len_r"], 4), -0.7452)
    record("fc_lysine_controlled_ub_hlai", abs(k_mon["lysine_controlled_hlai_ub_beta"] - 0.1994) < 5e-4, round(k_mon["lysine_controlled_hlai_ub_beta"], 4), 0.1994)
    record("fc_newey_crc01_external_validation", k_new["crc01_n_genes"] == 1748 and abs(k_new["crc01_mean_B_minus_A"] - 1.102) < 1e-3, [k_new["crc01_n_genes"], round(k_new["crc01_mean_B_minus_A"], 3)], [1748, 1.102])
    record("fc_hap1_2x2_sec61_signal_sparing", abs(100 * k_sec["tap2_true_sigpep_ret"] - 77.14) < 0.05 and abs(100 * k_sec["tap2_cytosolic_nterm_ret"] - 15.77) < 0.05, [round(100 * k_sec["tap2_true_sigpep_ret"], 2), round(100 * k_sec["tap2_cytosolic_nterm_ret"], 2)], [77.14, 15.77])
    record("fc_mid_tercile_and_disjoint_gene", abs(k_per["mid_tercile_durable_mean_auroc_mech"] - 0.6234) < 5e-4 and abs(k_per["disjoint_gene_uwb1289_high_persist_auroc_mech"] - 0.8633) < 5e-4, [round(k_per["mid_tercile_durable_mean_auroc_mech"], 4), round(k_per["disjoint_gene_uwb1289_high_persist_auroc_mech"], 4)], [0.6234, 0.8633])

    # 10. Module 7 checks
    e7_sum = json.loads((res_root / "exp7_distributional_sensing/summary.json").read_text())
    e7a = e7_sum["exp7a_followup_elasticity_and_ub"]
    record("mod7a_skmel5_meki_and_dynamic_ub_null", abs(100.0 * e7a["skmel5_meki_prot_r2"] - 3.10) < 0.05 and abs(100.0 * e7a["skmel5_meki_sibling_gt2fold"] - 3.66) < 0.05 and e7a["skmel5_dynamic_ub"]["p_dUb"] > 0.25, [round(100.0 * e7a["skmel5_meki_prot_r2"], 2), round(e7a["skmel5_dynamic_ub"]["p_dUb"], 3)], [3.10, 0.269])
    mm24 = e7a["bernhardt_mamel63a_ifng"][0]
    record("mod7a_bernhardt_psilac_ifng_hlab_divergence", abs(mm24["mean_same_gene_B_minus_A"] - 0.688) < 1e-3 and abs(100.0 * mm24["frac_multi_genes_gt_2fold"] - 60.36) < 0.1, [round(mm24["mean_same_gene_B_minus_A"], 3), round(100.0 * mm24["frac_multi_genes_gt_2fold"], 2)], [0.688, 60.36])

    velz_rungs = {r["representation"]: r["prolif_rel_mse_vs_subgroup"] for r in e7_sum["exp7b_velz_ladder"]}
    record("mod7b_velz_all_6_rungs_fail_prolif", min(velz_rungs.values()) > 0.99 and max(velz_rungs.values()) < 1.01, [round(min(velz_rungs.values()), 3), round(max(velz_rungs.values()), 3)], [0.991, 1.008])

    cl_map = {r["representation"]: r for r in e7_sum["exp7c_transfer_and_resistance"]["cross_lineage_ifng_vs_mapki"]}
    record("mod7c_cross_lineage_pos180_vs_coarse96", abs(cl_map["R2_PosAA180"]["lolo_auroc"] - 0.948) < 1e-6 and abs(cl_map["R1_Coarse96"]["lolo_auroc"] - 0.724) < 1e-6 and abs(cl_map["R2_PosAA180"]["loso_strict_disjoint_lineage_auroc"] - 0.756) < 1e-6, [cl_map["R2_PosAA180"]["lolo_auroc"], cl_map["R1_Coarse96"]["lolo_auroc"], cl_map["R2_PosAA180"]["loso_strict_disjoint_lineage_auroc"]], [0.948, 0.724, 0.756])

    il_map = {r["representation"]: r for r in e7_sum["exp7c_transfer_and_resistance"]["isobaric_il_harmonization"]}
    record("mod7c_isobaric_il_harmonization", abs(il_map["R2_PosAA180_IL"]["lolo_auroc"] - 0.916) < 1e-6 and abs(il_map["R2_PosAA180_IL"]["loso_strict_disjoint_lineage_auroc"] - 0.788) < 1e-6, [il_map["R2_PosAA180_IL"]["lolo_auroc"], il_map["R2_PosAA180_IL"]["loso_strict_disjoint_lineage_auroc"]], [0.916, 0.788])

    res_rows = {(r["training_design"], r["representation"]): r for r in e7_sum["exp7c_transfer_and_resistance"]["resistance_challenge"]}
    ws_p180 = res_rows[("Bernhardt_MaMel63a+UKEMel105b_to_SKMel28_ZeroHLAOverlap", "R2_PosAA180")]
    ws_c96 = res_rows[("Bernhardt_MaMel63a+UKEMel105b_to_SKMel28_ZeroHLAOverlap", "R1_Coarse96")]
    cp_p180 = res_rows[("CrossPlatform_Stopfer2022_TMT_to_Bernhardt2025_pSILAC", "R2_PosAA180")]
    cp_mda = res_rows[("CrossPlatform_Stopfer2022_TMT_to_Bernhardt2025_pSILAC", "R5b_Prespecified_MDA_Axis")]
    record(
        "mod7c_skmel28_zero_hla_resistance_and_cross_platform",
        abs(ws_p180["auroc_all_skmel28_8runs"] - 1.0) < 1e-6
        and abs(ws_p180["exact_combinatorial_p_2of8"] - 1.0 / 28.0) < 1e-6
        and abs(ws_c96["auroc_all_skmel28_8runs"] - 0.250) < 1e-6
        and abs(cp_p180["auroc_all_skmel28_8runs"] - 2.0 / 3.0) < 1e-3
        and abs(cp_mda["auroc_unseen_bernhardt_14runs"] - 0.9167) < 1e-3,
        [ws_p180["auroc_all_skmel28_8runs"], round(cp_p180["auroc_all_skmel28_8runs"], 3), round(cp_mda["auroc_unseen_bernhardt_14runs"], 3)],
        [1.0, 0.667, 0.917],
    )

    abl_map = {(r["mode"], r["frac_genes_removed"]): r for r in e7_sum["exp7d_distributed_redundancy"]["ablation_summary"]}
    top1_r = abl_map[("Top_1_Peptide_Markers_Only", 0.0)]
    ab10_r = abl_map[("Collective_Distribution_After_TopGene_Ablation", 0.1)]
    record("mod7d_redundancy_ablation_vs_top1_marker", top1_r["zero_feature_test_contrasts"] == 34 and abs(ab10_r["lolo_auroc_pos180"] - 0.860) < 1e-3, [top1_r["zero_feature_test_contrasts"], ab10_r["lolo_auroc_pos180"]], [34, 0.860])

    # 11. Module 8 (Provenance Roster, Kitagawa-Oaxaca Decomposition & External PRIDE Bridge) checks
    r8 = res_root / "exp8_immunity_bridge"
    roster = pd.read_csv(r8 / "contrast_roster_35.csv")
    f_lolo = pd.read_csv(r8 / "folds_lolo_16lineages.csv")
    f_loso = pd.read_csv(r8 / "folds_loso_strict_5studies.csv")
    v_lolo = validate_design_manifests(roster, f_lolo, strict_study=False)
    v_loso = validate_design_manifests(roster, f_loso, strict_study=True)
    record("mod8_roster_35_and_manifests_valid", len(roster) == 35 and v_lolo["all_structural_checks_pass"] and v_loso["all_structural_checks_pass"], {"n": len(roster), "lolo": v_lolo["all_structural_checks_pass"], "loso": v_loso["all_structural_checks_pass"]}, {"n": 35, "lolo": True, "loso": True})

    ko_df = pd.read_csv(r8 / "kitagawa_oaxaca_locus_vs_motif_decomposition.csv")
    ko_il_loso = ko_df[(ko_df["regime"] == "MHCflurry35_Hard_500nM") & (ko_df["sequence_coding"] == "Isobaric_IL_Harmonized") & (ko_df["eval_split"] == "LOSO")].iloc[0]
    record("mod8_kitagawa_oaxaca_exact_identity", float(ko_df["max_identity_residual"].max()) < 1e-12 and float(ko_df["max_score_reconstruction_err"].max()) < 1e-12 and abs(float(ko_il_loso["pct_linear_gap_from_within_locus"]) - 76.78) < 0.1, round(float(ko_il_loso["pct_linear_gap_from_within_locus"]), 2), 76.78)

    ext_df = pd.read_csv(r8 / "external_frozen_decoder_bridge_results.csv")
    ec_rep = ext_df[ext_df["contrast"] == "ECN90_BetaCell_IFNa_vs_Basal_RepFreq351"].iloc[0]
    t2_pool = ext_df[ext_df["contrast"] == "HAP1_TAP2_KO_IFNg_vs_UNT_Pooled"].iloc[0]
    record("mod8_external_pride_bridge_pxd085470_pxd045265", float(t2_pool["frozen_score_PosAA180_IL"]) > 9.0 and float(ec_rep["frozen_score_PosAA180_IL"]) > 4.8 and float(ec_rep["dw_HLA_B"]) > 0.26, [round(float(t2_pool["frozen_score_PosAA180_IL"]), 3), round(float(ec_rep["frozen_score_PosAA180_IL"]), 3)], [9.062, 4.857])

    # 12. Figures & hygiene check + Module 9 positional anatomy & supertype checks
    mod_dir = SCRIPT_DIR / "modules"
    fig_s1 = (mod_dir / "fig_main.py").read_text()
    fig_s2 = (mod_dir / "fig_supp.py").read_text()
    has_set_title = (".set_title(" in fig_s1) or (".set_title(" in fig_s2)
    req_figs = [
        "graphical_abstract.pdf",
        "fig1_concentration_decoupling.pdf",
        "fig_distributional_sensing_scope.pdf",
        "fig_representation_anatomy_and_logos.pdf",
        "fig2_monte_degradative_flux.pdf",
        "fig3_inflammatory_censoring_routing.pdf",
        "fig4_hap1_appm_gatekeepers.pdf",
        "fig5_persistence_prediction.pdf",
        "figS1_disattenuation_and_noise_null.pdf",
        "figS2_monte_allele_and_jackknife.pdf",
        "figS3_stoichiometry_and_newey_validation.pdf",
        "figS4_sec61_2x2_and_tercile_persistence.pdf",
        "figS5_followup_temporal_and_locus.pdf",
        "figS6_within_locus_pssm_logos.pdf",
    ]
    figs_present = all((cfg.fig_dir / fn).exists() for fn in req_figs)
    record("all_14_figures_exist_and_no_subplot_titles", figs_present and (not has_set_title), {"figs_present": figs_present, "has_set_title": has_set_title}, {"figs_present": True, "has_set_title": False})

    r9 = res_root / "exp9_post_meeting"
    if r9.exists():
        ab9 = pd.read_csv(r9 / "positional_ablation_and_subsets.csv")
        ab9_il = ab9[ab9["sequence_coding"] == "Isobaric_IL_Harmonized"].set_index("subset_key")
        s9 = json.loads((r9 / "summary.json").read_text())
        aff9 = pd.read_csv(r9 / "hla_abc_affinity_diagnostics.csv")
        aff9_c = float(aff9[(aff9["source"] == "MHCflurry_16Lineages_BestDonorLocus") & (aff9["locus"] == "HLA-C")]["frac_gt_500nM"].iloc[0])
        record(
            "mod9_positional_anatomy_logos_and_supertype_convergence",
            abs(float(ab9_il.loc["Drop_P2", "loso_strict_auroc"]) - 0.704) < 1e-6
            and abs(float(ab9_il.loc["Drop_P1", "skmel28_8run_auroc"]) - 1.0 / 6.0) < 1e-4
            and abs(float(s9["supertype_convergence"]["anchor_jsd_reduction_pct"]) - 39.85) < 0.1
            and abs(100.0 * aff9_c - 11.57) < 0.1,
            [round(float(ab9_il.loc["Drop_P2", "loso_strict_auroc"]), 3), round(float(s9["supertype_convergence"]["anchor_jsd_reduction_pct"]), 2), round(100.0 * aff9_c, 2)],
            [0.704, 39.85, 11.57],
        )

    n_failed = sum(1 for c in checks if not c["passed"])
    return {
        "n_checks": len(checks),
        "n_passed": len(checks) - n_failed,
        "n_failed": n_failed,
        "all_passed": n_failed == 0,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--upstream-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = resolve_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        upstream_dir=args.upstream_dir,
        results_dir=args.results_dir,
        fig_dir=args.fig_dir,
    )
    report = run_checks(cfg)
    out_p = args.out if args.out else cfg.results("verification_audit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification complete: {report['n_passed']}/{report['n_checks']} checks passed.")
    if not report["all_passed"]:
        for c in report["checks"]:
            if not c["passed"]:
                print("FAILED:", json.dumps(c), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
