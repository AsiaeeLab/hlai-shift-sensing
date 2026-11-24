"""Module 1: Testing the Static Concentration-Readout Hypothesis (y = Cx).

Integrates and audits:
1A. Cross-donor vs. paired within-donor decoding & RNA/methylation/peptide clustering
    across Stopfer (melanoma), Newey (CRC PDOs), and Velz (28 medulloblastomas).
1B. Microenvironmental & HLA-I/B2M collinearity dissection of the exploratory Velz
    Angiogenesis signal (with 10,000-permutation tests and partial correlations).
1C. Direct proteome-matched perturbation variance decomposition (dPep ~ dProt) and
    within-protein sibling-peptide divergence vs. biological-replicate spread across
    ITO57T, ITO58M, ITO66T, UWB1.289 (IFN-gamma), and A375 (TGF-beta).
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

SEED = 20261116


def perm_corr_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    x0 = x - x.mean()
    y0 = y - y.mean()
    obs = abs(float((x0 @ y0) / np.sqrt((x0 @ x0) * (y0 @ y0))))
    cnt = 1
    for _ in range(n_perm):
        yp = rng.permutation(y0)
        r = abs(float((x0 @ yp) / np.sqrt((x0 @ x0) * (yp @ yp))))
        if r >= obs - 1e-12:
            cnt += 1
    return cnt / (n_perm + 1)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Pearson correlation between x and y given matrix/vector z."""
    if z.ndim == 1:
        z = z[:, None]
    Z = np.column_stack([np.ones(len(x)), z])
    bx, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    by, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    r, _ = stats.pearsonr(rx, ry)
    df = len(x) - 2 - z.shape[1]
    t = r * np.sqrt(df / max(1e-15, 1.0 - r * r))
    p = float(2.0 * stats.t.sf(abs(t), df=df))
    return float(r), p


def analyze_velz_microenvironment(cfg: PathResolver, out_dir: Path) -> dict:
    velz_rna = pd.read_csv(cfg.processed("velz2025", "RNA_gene_tpm.tsv.gz"), sep="\t").set_index("sample_id")
    velz_ga = pd.read_csv(cfg.processed("velz2025", "gene_annotation.tsv"), sep="\t")
    velz_s = pd.read_csv(cfg.processed("velz2025", "samples.tsv"), sep="\t").set_index("sample_id")
    sym_to_ensg = velz_ga[velz_ga["symbol_is_unique"] == True].set_index("gene_symbol")["gene_id"].to_dict()

    def score_genes(syms: list[str]) -> pd.Series:
        cols = [sym_to_ensg[s] for s in syms if s in sym_to_ensg and sym_to_ensg[s] in velz_rna.columns]
        return np.log2(1.0 + velz_rna[cols]).mean(axis=1)

    covs = pd.DataFrame(index=velz_rna.index)
    covs["molecular_subgroup"] = velz_s.loc[velz_rna.index, "molecular_subgroup"]
    covs["hla_i_b2m_mrna"] = score_genes(["HLA-A", "HLA-B", "HLA-C", "B2M"])
    covs["appm_mrna"] = score_genes(["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9", "PSMB10", "ERAP1"])
    covs["endothelial_mrna"] = score_genes(["CDH5", "VWF", "KDR", "FLT1", "CLDN5", "TIE1", "ENG"])
    covs["ecm_stromal_mrna"] = score_genes(["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP"])
    covs["myeloid_mrna"] = score_genes(["CD68", "CD163", "C1QA", "C1QB", "AIF1", "CSF1R"])
    covs["tcell_mrna"] = score_genes(["CD3D", "CD3E", "CD8A", "CD8B", "GZMA", "PRF1"])

    ang_pred = pd.read_csv(
        cfg.upstream("results", "velz_regularization_programs", "same_coarse_target_predictions.tsv"), sep="\t"
    )
    ang = (
        ang_pred[(ang_pred["model"] == "regularized_coarse96") & (ang_pred["program"] == "Angiogenesis")]
        .set_index("sample_id")
        .loc[velz_rna.index]
    )
    covs["angiogenesis_target"] = ang["target"]
    covs["angiogenesis_coarse96_pred"] = ang["prediction"]
    covs["angiogenesis_subgroup_ref"] = ang["subgroup_prediction"]
    covs.reset_index().to_csv(out_dir / "velz_microenvironment_patient_scores.tsv", sep="\t", index=False)

    rows = []
    for cname in ["hla_i_b2m_mrna", "appm_mrna", "endothelial_mrna", "myeloid_mrna", "ecm_stromal_mrna", "tcell_mrna"]:
        x = covs[cname].values
        yt = covs["angiogenesis_target"].values
        yp = covs["angiogenesis_coarse96_pred"].values
        r_t, p_t = stats.pearsonr(x, yt)
        rho_t, sp_t = stats.spearmanr(x, yt)
        perm_p_t = perm_corr_pvalue(x, yt, n_perm=10000, seed=SEED + len(rows))
        r_p, p_p = stats.pearsonr(x, yp)
        sub_dummies = pd.get_dummies(covs["molecular_subgroup"], drop_first=True, dtype=float).values
        pr_t, pp_t = partial_corr(x, yt, sub_dummies)
        rows.append(
            {
                "covariate": cname,
                "pearson_r_target": float(r_t),
                "pearson_p_target": float(p_t),
                "perm_p_target": float(perm_p_t),
                "spearman_rho_target": float(rho_t),
                "spearman_p_target": float(sp_t),
                "within_subgroup_partial_r_target": float(pr_t),
                "within_subgroup_partial_p_target": float(pp_t),
                "pearson_r_coarse96_pred": float(r_p),
                "pearson_p_coarse96_pred": float(p_p),
            }
        )
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out_dir / "velz_angiogenesis_collinearity.tsv", sep="\t", index=False)

    sub_dummies = pd.get_dummies(covs["molecular_subgroup"], drop_first=True, dtype=float).values
    r_raw, p_raw = stats.pearsonr(covs["angiogenesis_coarse96_pred"].values, covs["angiogenesis_target"].values)
    r_sub, p_sub = partial_corr(
        covs["angiogenesis_coarse96_pred"].values, covs["angiogenesis_target"].values, sub_dummies
    )
    r_env, p_env = partial_corr(
        covs["angiogenesis_coarse96_pred"].values,
        covs["angiogenesis_target"].values,
        np.column_stack(
            [
                sub_dummies,
                covs["hla_i_b2m_mrna"].values,
                covs["endothelial_mrna"].values,
                covs["myeloid_mrna"].values,
            ]
        ),
    )
    return {
        "r_pred_target_marginal": r_raw,
        "p_pred_target_marginal": p_raw,
        "r_pred_target_given_subgroup": r_sub,
        "p_pred_target_given_subgroup": p_sub,
        "r_pred_target_given_subgroup_and_microenv": r_env,
        "p_pred_target_given_subgroup_and_microenv": p_env,
        "r_angio_vs_hlai_b2m": float(corr_df.loc[corr_df["covariate"] == "hla_i_b2m_mrna", "pearson_r_target"].iloc[0]),
        "p_angio_vs_hlai_b2m": float(corr_df.loc[corr_df["covariate"] == "hla_i_b2m_mrna", "pearson_p_target"].iloc[0]),
    }


def analyze_perturbation_elasticity(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    configs = [
        ("MHC1-TIP Colorectal PDO", "ITO57T", "ITO57T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("MHC1-TIP Colorectal PDO", "ITO58M", "ITO58M", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("MHC1-TIP Colorectal PDO", "ITO66T", "ITO66T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Chong Ovarian Carcinoma", "UWB.1_289", "UWB1.289", "IFNG", cfg.processed("visibility_replication", "chong2018_gene_unique")),
        ("MHC1-TIP Melanoma", "A375", "A375", "TGFB", cfg.processed("visibility_mhc1tip_gene_unique", "a375")),
    ]

    summary_rows = []
    sibling_rows = []
    rng = np.random.default_rng(SEED)

    for study, bg_raw, bg_clean, trt, pdir in configs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        prots = pd.read_csv(pdir / "proteins.tsv.gz", sep="\t")

        ctrl_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
        trt_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == trt)]["sample_id"].tolist()
        ctrl_prot = samps[(samps["modality"] == "protein") & (samps["condition"] == "control")]["sample_id"].tolist()
        trt_prot = samps[(samps["modality"] == "protein") & (samps["condition"] == trt)]["sample_id"].tolist()

        psub = peps[peps["sample_id"].isin(ctrl_pep + trt_pep) & (peps["intensity"] > 0)].copy()
        psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])

        rep_sd_c = psub[psub["sample_id"].isin(ctrl_pep)].groupby("peptide")["l2"].std().dropna()
        rep_sd_t = psub[psub["sample_id"].isin(trt_pep)].groupby("peptide")["l2"].std().dropna()
        expected_null_diff_sd = float(
            np.sqrt((rep_sd_c.median() ** 2) / len(ctrl_pep) + (rep_sd_t.median() ** 2) / len(trt_pep)) * np.sqrt(2.0)
        )

        p0 = psub[psub["sample_id"].isin(ctrl_pep)].groupby(["peptide", "gene"]).agg(pep0=("l2", "mean"), n0=("l2", "count"))
        p1 = psub[psub["sample_id"].isin(trt_pep)].groupby(["peptide", "gene"]).agg(pep1=("l2", "mean"), n1=("l2", "count"))
        pm = p0.join(p1, how="outer").reset_index()
        pm["n0"] = pm["n0"].fillna(0).astype(int)
        pm["n1"] = pm["n1"].fillna(0).astype(int)
        pm = pm[pm["gene"].notna() & (pm["gene"] != "") & (~pm["gene"].str.contains(";"))]

        pr_sub = prots[prots["sample_id"].isin(ctrl_prot + trt_prot) & np.isfinite(prots["log2_abundance"])]
        pr0 = pr_sub[pr_sub["sample_id"].isin(ctrl_prot)].groupby("gene").agg(prot0=("log2_abundance", "mean"), pn0=("log2_abundance", "count"))
        pr1 = pr_sub[pr_sub["sample_id"].isin(trt_prot)].groupby("gene").agg(prot1=("log2_abundance", "mean"), pn1=("log2_abundance", "count"))
        prm = pr0.join(pr1, how="inner")
        min_pc = int(np.ceil(2 * len(ctrl_prot) / 3))
        min_pt = int(np.ceil(2 * len(trt_prot) / 3))
        prm = prm[(prm["pn0"] >= min_pc) & (prm["pn1"] >= min_pt)].copy()
        prm["dProt"] = prm["prot1"] - prm["prot0"]

        df = pm.merge(prm.reset_index()[["gene", "prot0", "prot1", "dProt"]], on="gene", how="inner")
        min_c = int(np.ceil(2 * len(ctrl_pep) / 3))
        min_t = int(np.ceil(2 * len(trt_pep) / 3))
        df["base_supp"] = df["n0"] >= min_c
        df["trt_supp"] = df["n1"] >= min_t
        df["dPep"] = df["pep1"] - df["pep0"]

        base_df = df[df["base_supp"]]
        surv = df[df["base_supp"] & df["trt_supp"]].copy()
        denovo = df[(df["n0"] == 0) & df["trt_supp"]]

        lr = stats.linregress(surv["dProt"], surv["dPep"])
        rho, rho_p = stats.spearmanr(surv["dProt"], surv["dPep"])

        genes = surv["gene"].unique()
        g_idx = [np.where(surv["gene"].values == g)[0] for g in genes]
        boot_betas, boot_r2 = [], []
        dp_arr = surv["dProt"].values
        dy_arr = surv["dPep"].values
        for _ in range(1000):
            pick = rng.integers(0, len(genes), size=len(genes))
            idx = np.concatenate([g_idx[i] for i in pick])
            b_lr = stats.linregress(dp_arr[idx], dy_arr[idx])
            boot_betas.append(b_lr.slope)
            boot_r2.append(b_lr.rvalue ** 2)

        multi = surv.groupby("gene").filter(lambda g: len(g) >= 2)
        ranges = multi.groupby("gene")["dPep"].apply(lambda s: float(s.max() - s.min()))
        for g_name, r_val in ranges.items():
            sibling_rows.append(
                {
                    "background": bg_clean,
                    "perturbation": trt,
                    "gene": g_name,
                    "within_gene_dPep_range": r_val,
                    "expected_replicate_noise_sd": expected_null_diff_sd,
                }
            )

        summary_rows.append(
            {
                "study": study,
                "background": bg_clean,
                "perturbation": trt,
                "n_baseline_peptides": int(len(base_df)),
                "n_surviving_peptides": int(len(surv)),
                "retention_rate": float(len(surv) / len(base_df)),
                "n_denovo_peptides": int(len(denovo)),
                "repertoire_churn_ratio": float((len(base_df) - len(surv) + len(denovo)) / len(base_df)),
                "elasticity_beta": float(lr.slope),
                "elasticity_se_ols": float(lr.stderr),
                "elasticity_ci95_low": float(np.percentile(boot_betas, 2.5)),
                "elasticity_ci95_high": float(np.percentile(boot_betas, 97.5)),
                "r_squared": float(lr.rvalue ** 2),
                "r_squared_ci95_low": float(np.percentile(boot_r2, 2.5)),
                "r_squared_ci95_high": float(np.percentile(boot_r2, 97.5)),
                "unexplained_variance_fraction": float(1.0 - lr.rvalue ** 2),
                "ols_p": float(lr.pvalue),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "n_multi_ligand_genes": int(len(ranges)),
                "median_within_gene_dPep_range": float(ranges.median()),
                "frac_genes_diverging_gt_2fold": float((ranges > 1.0).mean()),
                "expected_replicate_diff_sd": expected_null_diff_sd,
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "perturbation_elasticity_summary.tsv", sep="\t", index=False)
    pd.DataFrame(sibling_rows).to_csv(out_dir / "within_gene_sibling_ranges.tsv", sep="\t", index=False)
    return sum_df


def summarize_prior_decoding_benchmarks(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    """Extract benchmark metrics from upstream Newey paired/LOO and Velz ARI tables."""
    rows = []
    s_new = pd.read_csv(cfg.upstream("results", "newey_ifng_vs_tram", "summary.tsv"), sep="\t")
    c96_s = s_new[s_new["model"] == "coarse96"].iloc[0]
    p_new = pd.read_csv(cfg.upstream("results", "paired_ranking", "summary.tsv"), sep="\t")
    c96_p = p_new[p_new["model"] == "coarse96"].iloc[0]
    aa_p = p_new[p_new["model"] == "aa20"].iloc[0]

    rows.append(
        {
            "task": "Newey CRC PDO single-sample classification (IFNg vs MEKi, 4 PDOs)",
            "representation": "Coarse P4-P7 chemistry (96 features, rank 66)",
            "primary_metric_name": "Balanced accuracy",
            "primary_metric": float(c96_s["mean_balanced_accuracy"]),
            "reference_value": 0.500,
            "secondary_metric_name": "Within-PDO AUROC",
            "secondary_metric": float(c96_s["mean_auroc"]),
        }
    )
    rows.append(
        {
            "task": "Newey CRC PDO paired same-background ordering (4 PDOs)",
            "representation": "Coarse P4-P7 chemistry (96 features)",
            "primary_metric_name": "Pair concordance",
            "primary_metric": float(c96_p["mean_concordance"]),
            "reference_value": 0.500,
            "secondary_metric_name": "AA composition baseline concordance",
            "secondary_metric": float(aa_p["mean_concordance"]),
        }
    )

    v_ari = pd.read_csv(cfg.upstream("manuscript", "figures", "state_selectivity_ari.tsv"), sep="\t")
    for _, r in v_ari[v_ari["k"] == 4].iterrows():
        rows.append(
            {
                "task": f"Velz medulloblastoma k=4 partition agreement ({r['first']} vs {r['second']}, N=28)",
                "representation": f"{r['first']} vs {r['second']}",
                "primary_metric_name": "Adjusted Rand Index (ARI)",
                "primary_metric": float(r["score"]),
                "reference_value": 0.000,
                "secondary_metric_name": "BH q-value",
                "secondary_metric": float(r["q_bh"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "decoding_benchmark_synthesis.tsv", sep="\t", index=False)
    return df


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp1_concentration_decoupling")
    out_dir.mkdir(parents=True, exist_ok=True)

    velz_info = analyze_velz_microenvironment(cfg, out_dir)
    elast_df = analyze_perturbation_elasticity(cfg, out_dir)
    summarize_prior_decoding_benchmarks(cfg, out_dir)

    summary = {
        "velz_microenvironment": velz_info,
        "perturbation_r2_min": float(elast_df["r_squared"].min()),
        "perturbation_r2_max": float(elast_df["r_squared"].max()),
        "ifng_diverging_gt2fold_min": float(
            elast_df[elast_df["perturbation"] == "IFNG"]["frac_genes_diverging_gt_2fold"].min()
        ),
        "ifng_diverging_gt2fold_max": float(
            elast_df[elast_df["perturbation"] == "IFNG"]["frac_genes_diverging_gt_2fold"].max()
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
