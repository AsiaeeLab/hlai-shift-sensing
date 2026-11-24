"""Module 3: Two-Margin Censored Inflammatory Remodeling, Allotype-Channel
Routing, and Immunoproteasome Cleavage Editing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats

from presentation_state.config import PathResolver, resolve_config


def bh_adjust(pvals: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_ord = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ord = np.clip(q_ord, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q_ord
    return out


def gene_first_weights(df: pd.DataFrame, gene_col: str = "target_gene") -> np.ndarray:
    counts = df.groupby(gene_col)[gene_col].transform("count").astype(float)
    n_genes = float(df[gene_col].nunique())
    if n_genes == 0:
        return np.array([], dtype=float)
    return (1.0 / (n_genes * counts)).values


def compute_censoring_bounds(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    bg_specs = [
        ("pdo", "ITO57T", "ITO57T", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.upstream("results", "visibility_final", "pdo", "ITO57T_IFNG_main")),
        ("pdo", "ITO58M", "ITO58M", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.upstream("results", "visibility_final", "pdo", "ITO58M_IFNG_main")),
        ("pdo", "ITO66T", "ITO66T", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.upstream("results", "visibility_final", "pdo", "ITO66T_IFNG_main")),
        ("chong", "UWB.1_289", "UWB1.289", cfg.processed("visibility_replication", "chong2018_gene_unique"), cfg.upstream("results", "visibility_final", "chong", "UWB_1_289_IFNG_main")),
    ]
    programs = [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_TGF_BETA_SIGNALING",
    ]

    rows = []
    for cohort, bg_raw, bg_clean, pdir, rdir in bg_specs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        ctrl_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
        trt_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "IFNG")]["sample_id"].tolist()

        psub = peps[peps["sample_id"].isin(ctrl_pep + trt_pep) & (peps["intensity"] > 0)].copy()
        psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])
        ctrl_med = float(psub[psub["sample_id"].isin(ctrl_pep)]["l2"].median())
        trt_vals = psub[psub["sample_id"].isin(trt_pep)]["l2"]
        trt_med = float(trt_vals.median())
        trt_lod_min = float(trt_vals.min())
        trt_lod_p01 = float(trt_vals.quantile(0.01))
        trt_lod_p05 = float(trt_vals.quantile(0.05))

        lig = pd.read_csv(rdir / "ligand_outcomes.tsv.gz", sep="\t").set_index("peptide")

        for prog in programs:
            po_path = rdir / f"{prog}_pair_outcomes.tsv"
            if not po_path.exists():
                continue
            po = pd.read_csv(po_path, sep="\t")
            if len(po) == 0:
                continue

            surv = po[po["both_survive"] == True].copy()
            w_surv = gene_first_weights(surv)
            theta_surv = float(np.dot(w_surv, surv["pair_u"].values)) if len(surv) > 0 else np.nan
            n_surv_genes = int(surv["target_gene"].nunique())

            t_n1 = lig.loc[po["target"], "n1"].values
            c_n1 = lig.loc[po["control"], "n1"].values
            t_pep0 = lig.loc[po["target"], "pep0"].values
            c_pep0 = lig.loc[po["control"], "pep0"].values
            t_pep1 = lig.loc[po["target"], "pep1"].values
            c_pep1 = lig.loc[po["control"], "pep1"].values

            def subst_theta(floor_val: float) -> float:
                tp1 = np.where(t_n1 == 0, floor_val, t_pep1)
                cp1 = np.where(c_n1 == 0, floor_val, c_pep1)
                tdA = tp1 - t_pep0
                cdA = cp1 - c_pep0
                pu = (tdA - po["target_dB"].values) - (cdA - po["control_dB"].values)
                valid = np.isfinite(pu)
                if valid.sum() == 0:
                    return np.nan
                sub_po = po.loc[valid].copy()
                w_all = gene_first_weights(sub_po)
                return float(np.dot(w_all, pu[valid]))

            if len(surv) >= 2:
                u_min, u_max = float(surv["pair_u"].quantile(0.05)), float(surv["pair_u"].quantile(0.95))
                w_base = gene_first_weights(po)
                u_low = np.where(po["both_survive"], po["pair_u"], u_min)
                u_high = np.where(po["both_survive"], po["pair_u"], u_max)
                manski_q05_low = float(np.dot(w_base, u_low))
                manski_q95_high = float(np.dot(w_base, u_high))
            else:
                manski_q05_low, manski_q95_high = np.nan, np.nan

            w_b = gene_first_weights(po)
            t_ret = float(np.dot(w_b, po["target_survives"].astype(float)))
            c_ret = float(np.dot(w_b, po["control_survives"].astype(float)))

            theta_lod = subst_theta(trt_lod_min)
            theta_p01 = subst_theta(trt_lod_p01)
            theta_p05 = subst_theta(trt_lod_p05)

            rows.append(
                {
                    "cohort": cohort,
                    "background": bg_clean,
                    "program": prog.replace("HALLMARK_", ""),
                    "n_baseline_pairs": int(len(po)),
                    "n_surviving_pairs": int(len(surv)),
                    "pair_attrition_fraction": float(1.0 - len(surv) / len(po)),
                    "n_surviving_genes": n_surv_genes,
                    "qualified_10gene": bool(n_surv_genes >= 10),
                    "target_retention": t_ret,
                    "control_retention": c_ret,
                    "retention_diff": float(t_ret - c_ret),
                    "control_median_log2_ms": ctrl_med,
                    "treated_median_log2_ms": trt_med,
                    "theta_survivor": theta_surv,
                    "theta_lod_min_bound": theta_lod,
                    "theta_lod_p01_bound": theta_p01,
                    "theta_lod_p05_bound": theta_p05,
                    "manski_p05_lower": manski_q05_low,
                    "manski_p95_upper": manski_q95_high,
                    "sign_identified_across_lod_bounds": bool(
                        np.isfinite(theta_surv)
                        and np.isfinite(theta_lod)
                        and (np.sign(theta_surv) == np.sign(theta_lod))
                        and (np.sign(theta_surv) == np.sign(theta_p01))
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "censoring_bounds_all_programs.tsv", sep="\t", index=False)

    proc = pd.read_csv(cfg.upstream("results", "visibility_final_checks", "processing_protein_changes.tsv"), sep="\t")
    proc["background"] = proc["background"].replace({"UWB.1_289": "UWB1.289", "UWB.1 289": "UWB1.289"})
    proc.to_csv(out_dir / "appm_positive_control_proteomics.tsv", sep="\t", index=False)
    return df


def analyze_allotype_routing_and_siblings(cfg: PathResolver, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = pd.read_csv(cfg.upstream("results", "visibility_final_checks", "processing_protein_changes.tsv"), sep="\t")
    proc["background"] = proc["background"].replace({"UWB.1_289": "UWB1.289", "UWB.1 289": "UWB1.289"})
    hla_dB = proc[proc["gene"].isin(["HLA-A", "HLA-B", "HLA-C"])].pivot(index="background", columns="gene", values="dB")

    bg_specs = [
        ("ITO57T", "ITO57T", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.processed("visibility_binding", "gene_unique", "pdo57t_best.tsv.gz")),
        ("ITO58M", "ITO58M", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.processed("visibility_binding", "gene_unique", "pdo58m_best.tsv.gz")),
        ("ITO66T", "ITO66T", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), cfg.processed("visibility_binding", "gene_unique", "pdo66t_best.tsv.gz")),
        ("UWB.1_289", "UWB1.289", cfg.processed("visibility_replication", "chong2018_gene_unique"), cfg.processed("visibility_binding", "gene_unique", "chong2018_best.tsv.gz")),
    ]

    locus_rows = []
    sibling_rows = []
    for bg_raw, bg_clean, pdir, bfile in bg_specs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        bdf = pd.read_csv(bfile, sep="\t").drop_duplicates("peptide")

        ctrl_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
        trt_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "IFNG")]["sample_id"].tolist()
        psub = peps[peps["sample_id"].isin(ctrl_pep + trt_pep) & (peps["intensity"] > 0)].copy()
        psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])

        p0 = psub[psub["sample_id"].isin(ctrl_pep)].groupby(["peptide", "gene"]).agg(pep0=("l2", "mean"), n0=("l2", "count"))
        p1 = psub[psub["sample_id"].isin(trt_pep)].groupby(["peptide", "gene"]).agg(pep1=("l2", "mean"), n1=("l2", "count"))
        pm = p0.join(p1, how="outer").reset_index().merge(bdf, on="peptide", how="inner")
        pm["n0"] = pm["n0"].fillna(0).astype(int)
        pm["n1"] = pm["n1"].fillna(0).astype(int)
        pm["locus"] = "HLA-" + pm["allele"].astype(str).str.extract(r"([ABC])\*")[0]
        min_c = int(np.ceil(2 * len(ctrl_pep) / 3))
        min_t = int(np.ceil(2 * len(trt_pep) / 3))
        pm["base_supp"] = pm["n0"] >= min_c
        pm["trt_supp"] = pm["n1"] >= min_t
        pm["dPep"] = pm["pep1"] - pm["pep0"]

        base_df = pm[pm["base_supp"]]
        surv_df = pm[pm["base_supp"] & pm["trt_supp"]].copy()
        denovo_df = pm[(pm["n0"] == 0) & pm["trt_supp"]]
        med_shift = float(surv_df["dPep"].median())
        surv_df["rel_dPep"] = surv_df["dPep"] - med_shift

        base_shares = base_df["locus"].value_counts(normalize=True).to_dict()
        denovo_shares = denovo_df["locus"].value_counts(normalize=True).to_dict()

        for loc in ["HLA-A", "HLA-B", "HLA-C"]:
            s_loc = surv_df[surv_df["locus"] == loc]
            b_loc = base_df[base_df["locus"] == loc]
            d_prot_val = (
                float(hla_dB.loc[bg_clean, loc])
                if (bg_clean in hla_dB.index and loc in hla_dB.columns and pd.notna(hla_dB.loc[bg_clean, loc]))
                else np.nan
            )
            b_sh = float(base_shares.get(loc, 0.0))
            dn_sh = float(denovo_shares.get(loc, 0.0))
            locus_rows.append(
                {
                    "background": bg_clean,
                    "locus": loc,
                    "n_baseline": int(len(b_loc)),
                    "n_surviving": int(len(s_loc)),
                    "retention": float(b_loc["trt_supp"].mean()) if len(b_loc) else np.nan,
                    "baseline_share": b_sh,
                    "denovo_share": dn_sh,
                    "denovo_to_baseline_ratio": float(dn_sh / b_sh) if b_sh > 0 else np.nan,
                    "mean_dPep": float(s_loc["dPep"].mean()) if len(s_loc) else np.nan,
                    "median_dPep": float(s_loc["dPep"].median()) if len(s_loc) else np.nan,
                    "mean_rel_dPep": float(s_loc["rel_dPep"].mean()) if len(s_loc) else np.nan,
                    "measured_heavy_chain_dProt": d_prot_val,
                }
            )

        ab = surv_df[surv_df["locus"].isin(["HLA-A", "HLA-B"]) & surv_df["gene"].notna() & (~surv_df["gene"].str.contains(";"))]
        gene_diffs = []
        n_pairs = 0
        for gene, sub in ab.groupby("gene"):
            if sub["locus"].nunique() < 2:
                continue
            b_vals = sub[sub["locus"] == "HLA-B"]["rel_dPep"].values
            a_vals = sub[sub["locus"] == "HLA-A"]["rel_dPep"].values
            diffs = [float(bv - av) for bv in b_vals for av in a_vals]
            n_pairs += len(diffs)
            gene_diffs.append(float(np.mean(diffs)))

        if gene_diffs:
            g_arr = np.array(gene_diffs)
            _, w_p = stats.wilcoxon(g_arr)
            dprot_ba = (
                float(hla_dB.loc[bg_clean, "HLA-B"] - hla_dB.loc[bg_clean, "HLA-A"])
                if bg_clean in hla_dB.index and pd.notna(hla_dB.loc[bg_clean, "HLA-B"]) and pd.notna(hla_dB.loc[bg_clean, "HLA-A"])
                else np.nan
            )
            sibling_rows.append(
                {
                    "background": bg_clean,
                    "n_cross_locus_pairs": int(n_pairs),
                    "n_genes": int(len(g_arr)),
                    "mean_gene_dPep_B_minus_A": float(g_arr.mean()),
                    "median_gene_dPep_B_minus_A": float(np.median(g_arr)),
                    "frac_genes_B_gt_A": float((g_arr > 0).mean()),
                    "wilcoxon_two_sided_p": float(w_p),
                    "measured_dProt_B_minus_A": dprot_ba,
                }
            )

    loc_df = pd.DataFrame(locus_rows)
    sib_df = pd.DataFrame(sibling_rows)
    loc_df.to_csv(out_dir / "locus_channel_reallocation.tsv", sep="\t", index=False)
    sib_df.to_csv(out_dir / "same_protein_hlab_vs_hlaa_siblings.tsv", sep="\t", index=False)
    return loc_df, sib_df


def analyze_proteolytic_motifs_8_backgrounds(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    rows = []

    bg_specs = [
        ("MHC1-TIP PDO", "ITO57T", "ITO57T", "IFNG vs Control", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("MHC1-TIP PDO", "ITO58M", "ITO58M", "IFNG vs Control", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("MHC1-TIP PDO", "ITO66T", "ITO66T", "IFNG vs Control", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Chong Ovarian", "UWB.1_289", "UWB1.289", "IFNG vs Control", cfg.processed("visibility_replication", "chong2018_gene_unique")),
    ]
    for cohort, bg_raw, bg_clean, contrast, pdir in bg_specs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        ctrl_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
        trt_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "IFNG")]["sample_id"].tolist()
        psub = peps[peps["sample_id"].isin(ctrl_pep + trt_pep) & (peps["intensity"] > 0)].copy()
        psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])
        p0 = psub[psub["sample_id"].isin(ctrl_pep)].groupby("peptide").agg(pep0=("l2", "mean"), n0=("l2", "count"))
        p1 = psub[psub["sample_id"].isin(trt_pep)].groupby("peptide").agg(pep1=("l2", "mean"), n1=("l2", "count"))
        min_c = int(np.ceil(2 * len(ctrl_pep) / 3))
        min_t = int(np.ceil(2 * len(trt_pep) / 3))
        surv = p0.join(p1, how="inner")
        surv = surv[(surv["n0"] >= min_c) & (surv["n1"] >= min_t)].reset_index()
        surv["dPep"] = surv["pep1"] - surv["pep0"]
        hydro = surv["peptide"].str[-1].isin(list("FWYLIVMA"))
        pro = surv["peptide"].str[3:7].str.contains("P")
        u_h = stats.mannwhitneyu(surv.loc[hydro, "dPep"], surv.loc[~hydro, "dPep"])
        u_p = stats.mannwhitneyu(surv.loc[pro, "dPep"], surv.loc[~pro, "dPep"])
        rows.append(
            {
                "cohort": cohort,
                "background": bg_clean,
                "contrast": contrast,
                "n_surviving_peptides": int(len(surv)),
                "hydro_cterm_mean_dPep": float(surv.loc[hydro, "dPep"].mean()),
                "nonhydro_cterm_mean_dPep": float(surv.loc[~hydro, "dPep"].mean()),
                "hydro_minus_nonhydro_dPep": float(surv.loc[hydro, "dPep"].mean() - surv.loc[~hydro, "dPep"].mean()),
                "hydro_mwu_p": float(u_h.pvalue),
                "core_pro_mean_dPep": float(surv.loc[pro, "dPep"].mean()),
                "non_core_pro_mean_dPep": float(surv.loc[~pro, "dPep"].mean()),
                "core_pro_minus_nonpro_dPep": float(surv.loc[pro, "dPep"].mean() - surv.loc[~pro, "dPep"].mean()),
                "core_pro_mwu_p": float(u_p.pvalue),
            }
        )

    new_peps = pd.read_csv(cfg.processed("newey2019_replicates", "peptides_long.tsv.gz"), sep="\t")
    for org in ["CRC01", "CRC04", "CRC05", "CRC08"]:
        sub = new_peps[(new_peps["organoid"] == org) & (new_peps["intensity"] > 0)].copy()
        sub["l2"] = np.log2(sub["intensity"])
        p0 = sub[sub["condition"] == "TRAM"].groupby("peptide").agg(pep0=("l2", "mean"), n0=("l2", "count"))
        p1 = sub[sub["condition"] == "IFNg"].groupby("peptide").agg(pep1=("l2", "mean"), n1=("l2", "count"))
        surv = p0.join(p1, how="inner")
        surv = surv[(surv["n0"] >= 2) & (surv["n1"] >= 2)].reset_index()
        surv["dPep"] = surv["pep1"] - surv["pep0"]
        hydro = surv["peptide"].str[-1].isin(list("FWYLIVMA"))
        pro = surv["peptide"].str[3:7].str.contains("P")
        u_h = stats.mannwhitneyu(surv.loc[hydro, "dPep"], surv.loc[~hydro, "dPep"])
        u_p = stats.mannwhitneyu(surv.loc[pro, "dPep"], surv.loc[~pro, "dPep"])
        rows.append(
            {
                "cohort": "Newey CRC PDO",
                "background": org,
                "contrast": "IFNG vs Trametinib (same era)",
                "n_surviving_peptides": int(len(surv)),
                "hydro_cterm_mean_dPep": float(surv.loc[hydro, "dPep"].mean()),
                "nonhydro_cterm_mean_dPep": float(surv.loc[~hydro, "dPep"].mean()),
                "hydro_minus_nonhydro_dPep": float(surv.loc[hydro, "dPep"].mean() - surv.loc[~hydro, "dPep"].mean()),
                "hydro_mwu_p": float(u_h.pvalue),
                "core_pro_mean_dPep": float(surv.loc[pro, "dPep"].mean()),
                "non_core_pro_mean_dPep": float(surv.loc[~pro, "dPep"].mean()),
                "core_pro_minus_nonpro_dPep": float(surv.loc[pro, "dPep"].mean() - surv.loc[~pro, "dPep"].mean()),
                "core_pro_mwu_p": float(u_p.pvalue),
            }
        )

    df = pd.DataFrame(rows)
    df["hydro_bh_q"] = bh_adjust(df["hydro_mwu_p"].values)
    df["core_pro_bh_q"] = bh_adjust(df["core_pro_mwu_p"].values)
    df.to_csv(out_dir / "proteolytic_motif_shifts_8_backgrounds.tsv", sep="\t", index=False)
    return df


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp3_inflammatory_routing")
    out_dir.mkdir(parents=True, exist_ok=True)

    cens_df = compute_censoring_bounds(cfg, out_dir)
    loc_df, _ = analyze_allotype_routing_and_siblings(cfg, out_dir)
    motif_df = analyze_proteolytic_motifs_8_backgrounds(cfg, out_dir)

    uwb_e2f = cens_df[(cens_df["background"] == "UWB1.289") & (cens_df["program"] == "E2F_TARGETS")].iloc[0]
    summary = {
        "uwb1289_e2f_theta_survivor": float(uwb_e2f["theta_survivor"]),
        "uwb1289_e2f_theta_lod_min": float(uwb_e2f["theta_lod_min_bound"]),
        "uwb1289_e2f_theta_lod_p01": float(uwb_e2f["theta_lod_p01_bound"]),
        "uwb1289_e2f_sign_identified": bool(uwb_e2f["sign_identified_across_lod_bounds"]),
        "ito58m_hlab_denovo_ratio": float(
            loc_df[(loc_df["background"] == "ITO58M") & (loc_df["locus"] == "HLA-B")]["denovo_to_baseline_ratio"].iloc[0]
        ),
        "uwb1289_hlab_denovo_ratio": float(
            loc_df[(loc_df["background"] == "UWB1.289") & (loc_df["locus"] == "HLA-B")]["denovo_to_baseline_ratio"].iloc[0]
        ),
        "hydro_cterm_positive_backgrounds": int((motif_df["hydro_minus_nonhydro_dPep"] > 0).sum()),
        "total_ifng_backgrounds_tested": int(len(motif_df)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
