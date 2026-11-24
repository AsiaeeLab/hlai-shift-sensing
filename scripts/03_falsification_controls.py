#!/usr/bin/env python3
"""Run disattenuation, multiplicity-matched noise nulls, econometric IV/2-way FE, and 2x2 factorial controls.

Usage:
    python3 scripts/03_falsification_controls.py --data-dir ./data --processed-dir ./data/processed --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from presentation_state.config import PathResolver, resolve_config  # noqa: E402
from modules.mod2_monte_degradative_flux import PATIENTS_DASH, build_gene_patient_matrix  # noqa: E402
from modules.mod4_hap1_gatekeepers import load_canonical_sequences  # noqa: E402
from modules.mod5_persistence_prediction import build_background_table  # noqa: E402

SEED = 20261116


def run_disattenuation_and_sibling_null(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    specs = [
        ("ITO57T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO57T"),
        ("ITO58M", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO58M"),
        ("ITO66T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO66T"),
        ("UWB1.289", "IFNG", cfg.processed("visibility_replication", "chong2018_gene_unique"), "UWB.1_289"),
        ("A375", "TGFB", cfg.processed("visibility_mhc1tip_gene_unique", "a375"), "A375"),
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for bg_clean, pert, pdir, bg_raw in specs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        prots = pd.read_csv(pdir / "proteins.tsv.gz", sep="\t")

        c_pep = sorted(samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist())
        t_pep = sorted(samps[(samps["modality"] == "peptide") & (samps["condition"] == pert)]["sample_id"].tolist())
        c_prt = sorted(samps[(samps["modality"] == "protein") & (samps["condition"] == "control")]["sample_id"].tolist())
        t_prt = sorted(samps[(samps["modality"] == "protein") & (samps["condition"] == pert)]["sample_id"].tolist())

        psub = peps[peps["sample_id"].isin(c_pep + t_pep) & (peps["intensity"] > 0)].copy()
        psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])

        p_mat = psub.pivot_table(index=["peptide", "gene"], columns="sample_id", values="l2")
        min_c = int(np.ceil(2 * len(c_pep) / 3))
        min_t = int(np.ceil(2 * len(t_pep) / 3))
        n0 = p_mat[c_pep].notna().sum(axis=1)
        n1 = p_mat[t_pep].notna().sum(axis=1)
        p_surv = p_mat[(n0 >= min_c) & (n1 >= min_t)].copy()

        var_c = p_surv[c_pep].var(axis=1, ddof=1)
        var_t = p_surv[t_pep].var(axis=1, ddof=1)
        nc_i = p_surv[c_pep].notna().sum(axis=1)
        nt_i = p_surv[t_pep].notna().sum(axis=1)
        se2_dpep = (var_c / nc_i + var_t / nt_i).dropna()
        mean_err_var_pep = float(se2_dpep.mean())

        dpep_obs = (p_surv[t_pep].mean(axis=1) - p_surv[c_pep].mean(axis=1)).rename("dPep").reset_index()

        pr_mat = prots[prots["sample_id"].isin(c_prt + t_prt) & np.isfinite(prots["log2_abundance"])].pivot_table(
            index="gene", columns="sample_id", values="log2_abundance"
        )
        pn0 = pr_mat[c_prt].notna().sum(axis=1)
        pn1 = pr_mat[t_prt].notna().sum(axis=1)
        pr_ok = pr_mat[(pn0 >= int(np.ceil(2 * len(c_prt) / 3))) & (pn1 >= int(np.ceil(2 * len(t_prt) / 3)))].copy()
        dprot_obs = (pr_ok[t_prt].mean(axis=1) - pr_ok[c_prt].mean(axis=1)).rename("dProt")
        pvar_c = pr_ok[c_prt].var(axis=1, ddof=1)
        pvar_t = pr_ok[t_prt].var(axis=1, ddof=1)
        se2_dprot = (pvar_c / pr_ok[c_prt].notna().sum(axis=1) + pvar_t / pr_ok[t_prt].notna().sum(axis=1)).dropna()

        m = dpep_obs.merge(dprot_obs.reset_index(), on="gene", how="inner")
        m_se = p_surv.copy()
        m_se["se_dpep"] = np.sqrt((var_c / nc_i + var_t / nt_i).fillna(mean_err_var_pep))
        m = m.merge(m_se[["se_dpep"]].reset_index(), on=["peptide", "gene"], how="left")

        tot_var_pep = float(m["dPep"].var(ddof=1))
        rel_pep = float(np.clip(1.0 - float((m["se_dpep"] ** 2).median()) / tot_var_pep, 0.1, 1.0))

        m_se_p = se2_dprot.reindex(m["gene"]).dropna()
        tot_var_prot = float(m["dProt"].var(ddof=1))
        rel_prot = float(np.clip(1.0 - float(m_se_p.median()) / tot_var_prot, 0.1, 1.0))

        r_raw = float(np.corrcoef(m["dProt"], m["dPep"])[0, 1])
        r2_raw = r_raw ** 2
        r2_of_reliable_pep = float(np.clip(r2_raw / rel_pep, 0.0, 1.0))
        r2_fully_disattenuated = float(np.clip(r2_raw / (rel_pep * rel_prot), 0.0, 1.0))

        g_counts = m.groupby("gene").size()
        multi_genes = g_counts[g_counts >= 2].index
        m_multi = m[m["gene"].isin(multi_genes)].copy()
        obs_ranges = m_multi.groupby("gene")["dPep"].agg(lambda x: float(x.max() - x.min()))
        obs_gt2x = float((obs_ranges > 1.0).mean())

        gene_codes = m_multi["gene"].astype("category").cat.codes.values
        n_mg = int(gene_codes.max()) + 1
        ses = m_multi["se_dpep"].values
        null_gt2x_draws = []
        for _ in range(500):
            noise = rng.normal(0.0, ses)
            df_sim = pd.DataFrame({"g": gene_codes, "z": noise})
            sim_rng = df_sim.groupby("g")["z"].max() - df_sim.groupby("g")["z"].min()
            null_gt2x_draws.append(float((sim_rng > 1.0).mean()))
        null_gt2x_mean = float(np.mean(null_gt2x_draws))
        null_gt2x_p95 = float(np.percentile(null_gt2x_draws, 95))

        rows.append(
            {
                "background": bg_clean,
                "perturbation": pert,
                "n_surviving_peptides": int(len(m)),
                "dPep_reliability_ceiling": rel_pep,
                "dProt_reliability": rel_prot,
                "r2_raw": r2_raw,
                "r2_share_of_reliable_dPep": r2_of_reliable_pep,
                "r2_fully_disattenuated": r2_fully_disattenuated,
                "n_multi_ligand_genes": int(n_mg),
                "obs_frac_gt2x_split": obs_gt2x,
                "multiplicity_noise_null_gt2x": null_gt2x_mean,
                "multiplicity_noise_null_p95": null_gt2x_p95,
                "excess_biological_gt2x_split": obs_gt2x - null_gt2x_mean,
                "fold_over_noise_null": obs_gt2x / max(null_gt2x_mean, 1e-4),
            }
        )
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "disattenuation_and_sibling_null.tsv", sep="\t", index=False)
    return df_out


def two_way_demean(df: pd.DataFrame, cols: list[str], g_col: str = "gene", p_col: str = "patient", n_iter: int = 30) -> pd.DataFrame:
    """Alternating projections (Gauss-Seidel) to remove exact gene and patient fixed effects on unbalanced panel."""
    out = df.copy()
    for c in cols:
        v = out[c].astype(float).copy()
        for _ in range(n_iter):
            v = v - v.groupby(out[g_col]).transform("mean")
            v = v - v.groupby(out[p_col]).transform("mean")
        out[c + "_dm"] = v
    return out


def cgm_two_way_se(X: np.ndarray, res: np.ndarray, g_ids: np.ndarray, p_ids: np.ndarray) -> np.ndarray:
    """Cameron-Gelbach-Miller (2011) two-way clustered covariance (gene + patient - intersection)."""
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    scores = X * res[:, None]

    def meat_1way(clusters):
        df_s = pd.DataFrame(scores)
        df_s["_cl"] = clusters
        grp = df_s.groupby("_cl").sum().values
        C = len(grp)
        scale = (C / max(C - 1.0, 1.0)) * ((n - 1.0) / max(n - k, 1.0))
        return scale * (grp.T @ grp)

    V_g = bread @ meat_1way(g_ids) @ bread
    V_p = bread @ meat_1way(p_ids) @ bread
    V_gp = ((n - 1.0) / max(n - k, 1.0)) * (bread @ (scores.T @ scores) @ bread)
    V_2way = V_g + V_p - V_gp
    diag = np.maximum(np.diag(V_2way), np.diag(V_g) * 0.25)
    return np.sqrt(diag)


def run_monte_controls(cfg: PathResolver, out_dir: Path) -> dict:
    mdir = cfg.processed("visibility_monte")
    res2 = cfg.results("exp2_monte_degradative_flux")
    af = pd.read_csv(res2 / "across_gene_ptm_features.tsv.gz", sep="\t")
    gene_seqs = load_canonical_sequences(cfg)
    k_counts = {g: seq.count("K") for g, seq in gene_seqs.items()}
    af["log2_lys_count"] = np.log2(1.0 + af["gene"].map(k_counts).fillna(af["mw"] / 110.0 * 0.058))

    x_cols = [
        "log2_mw",
        "log2_lys_count",
        "log2_spec_abundance",
        "log2_ub_sites",
        "log2_phos_sites",
        "log2_acetyl_sites",
        "secretory_er_membrane",
    ]
    X_raw = af[x_cols].values
    Xz = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=0)
    X = np.column_stack([np.ones(len(af)), Xz])
    lys_rows = []
    for y_col in ["log2_hlai_peptides", "log2_hlaii_peptides", "hlai_minus_hlaii"]:
        yv = af[y_col].values
        yz = (yv - yv.mean()) / yv.std(ddof=0)
        b, _, _, _ = np.linalg.lstsq(X, yz, rcond=None)
        res = yz - X @ b
        h_diag = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
        u_hc3 = res / np.maximum(1.0 - h_diag, 1e-6)
        cov = np.linalg.inv(X.T @ X) @ (X.T @ ((u_hc3 ** 2)[:, None] * X)) @ np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_st = b / se
        pv = 2.0 * stats.t.sf(np.abs(t_st), df=len(af) - X.shape[1])
        for cname, bi, sei, ti, pi in zip(x_cols, b[1:], se[1:], t_st[1:], pv[1:]):
            lys_rows.append({"outcome": y_col, "predictor": cname, "std_beta": float(bi), "hc3_se": float(sei), "t_stat": float(ti), "p_value": float(pi)})
    pd.DataFrame(lys_rows).to_csv(out_dir / "monte_lysine_controlled_regression.tsv", sep="\t", index=False)

    pep1 = pd.read_csv(mdir / "peptides_global_source_strict.tsv.gz", sep="\t")
    hlai_int = np.log2(pep1.groupby(["gene", "sample_id"])["intensity"].sum()).rename("hla").reset_index().rename(columns={"sample_id": "patient"})
    prot_f = pd.read_csv(mdir / "proteome_feature_annotations.tsv.gz", sep="\t", low_memory=False)
    prot_m = pd.read_csv(mdir / "proteome_log2_ratios.tsv.gz", sep="\t")
    ub_f = pd.read_csv(mdir / "ubiquitylome_feature_annotations.tsv.gz", sep="\t", low_memory=False)
    ub_m = pd.read_csv(mdir / "ubiquitylome_log2_ratios.tsv.gz", sep="\t")

    p_ft = build_gene_patient_matrix(prot_f, prot_m, "")[0].stack().rename("prot_FT").reset_index().rename(columns={"level_1": "patient"})
    p_no = build_gene_patient_matrix(prot_f, prot_m, ".N")[0].stack().rename("prot_noIP").reset_index().rename(columns={"level_1": "patient"})
    u_no = build_gene_patient_matrix(ub_f, ub_m, ".N")[0].stack().rename("ub_noIP").reset_index().rename(columns={"level_1": "patient"})

    pan_no = hlai_int.merge(p_no, on=["gene", "patient"]).merge(u_no, on=["gene", "patient"]).merge(p_ft, on=["gene", "patient"]).dropna()
    pan_no = two_way_demean(pan_no, ["hla", "prot_noIP", "ub_noIP", "prot_FT"])
    X_no = np.column_stack([pan_no["prot_noIP_dm"].values, pan_no["ub_noIP_dm"].values])
    y_no = pan_no["hla_dm"].values
    b_no, _, _, _ = np.linalg.lstsq(X_no, y_no, rcond=None)
    res_no = y_no - X_no @ b_no
    se_2w = cgm_two_way_se(X_no, res_no, pan_no["gene"].values, pan_no["patient"].values)
    Z_iv = np.column_stack([pan_no["prot_FT_dm"].values, pan_no["ub_noIP_dm"].values])
    X_hat = Z_iv @ np.linalg.lstsq(Z_iv, X_no, rcond=None)[0]
    b_iv = np.linalg.lstsq(X_hat, y_no, rcond=None)[0]
    prot_reliability = float(np.corrcoef(pan_no["prot_noIP_dm"], pan_no["prot_FT_dm"])[0, 1])

    raw_h1 = pd.read_csv(
        mdir / "canonical_hlai_author_rows.tsv.gz",
        sep="\t",
        usecols=["sample_id", "peptide", "length", "allele", "primary_eligible", "detected_positive_signal"],
    )
    raw_h1 = raw_h1[raw_h1["primary_eligible"] & raw_h1["detected_positive_signal"] & raw_h1["allele"].notna()].copy()
    pt_df = pd.read_csv(res2 / "monte_patient_appm_and_ligandome.tsv", sep="\t")
    clin = pd.read_csv(mdir / "clinical.tsv", sep="\t")
    pt_full = pt_df.merge(clin, on="sample_id")

    al_n = raw_h1.groupby("allele")["sample_id"].nunique()
    shared_alleles = al_n[al_n >= 2].index.tolist()
    al_panel = (
        raw_h1[raw_h1["allele"].isin(shared_alleles)]
        .groupby(["allele", "sample_id"])
        .agg(mean_len=("length", "mean"), frac_10_12=("length", lambda x: float(np.mean(x >= 10))), n_peps=("peptide", "nunique"))
        .reset_index()
        .merge(pt_df[["sample_id", "erap1_prot_noIP", "erap1_prot_FT"]], on="sample_id")
    )
    al_panel["len_within_allele"] = al_panel["mean_len"] - al_panel.groupby("allele")["mean_len"].transform("mean")
    al_panel["f10_within_allele"] = al_panel["frac_10_12"] - al_panel.groupby("allele")["frac_10_12"].transform("mean")
    al_panel["erap1_within_allele"] = al_panel["erap1_prot_noIP"] - al_panel.groupby("allele")["erap1_prot_noIP"].transform("mean")
    al_panel.to_csv(out_dir / "monte_within_exact_allele_erap1.tsv", sep="\t", index=False)

    r_wa, p_wa = stats.pearsonr(al_panel["erap1_within_allele"], al_panel["len_within_allele"])
    Xa = al_panel[["erap1_within_allele"]].values
    ya = al_panel["len_within_allele"].values
    ba = float((Xa[:, 0] @ ya) / (Xa[:, 0] @ Xa[:, 0]))
    resa = ya - Xa[:, 0] * ba
    p_scores = pd.DataFrame({"s": Xa[:, 0] * resa, "pt": al_panel["sample_id"]}).groupby("pt")["s"].sum().values
    G_pt = len(p_scores)
    se_pt_cl = float(np.sqrt((G_pt / (G_pt - 1.0)) * np.sum(p_scores ** 2) / ((Xa[:, 0] @ Xa[:, 0]) ** 2)))
    p_wa_pt_clustered = float(2.0 * stats.t.sf(abs(ba / se_pt_cl), df=G_pt - 1))

    jk_rows = []
    for drop_pt in pt_full["sample_id"]:
        sub = pt_full[pt_full["sample_id"] != drop_pt]
        jk_rows.append(
            {
                "omitted_patient": drop_pt,
                "r_appm_breadth_noIP": float(stats.pearsonr(sub["appm_core_prot_noIP"], sub["log2_n_hlai_peptides"])[0]),
                "r_appm_breadth_FT": float(stats.pearsonr(sub["appm_core_prot_FT"], sub["log2_n_hlai_peptides"])[0]),
                "r_erap1_len_noIP": float(stats.pearsonr(sub["erap1_prot_noIP"], sub["mean_peptide_length"])[0]),
            }
        )
    jk_df = pd.DataFrame(jk_rows)
    jk_df.to_csv(out_dir / "monte_jackknife_sensitivity.tsv", sep="\t", index=False)

    def partial_r(x, y, z):
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        rx = x[ok] - np.polyval(np.polyfit(z[ok], x[ok], 1), z[ok])
        ry = y[ok] - np.polyval(np.polyfit(z[ok], y[ok], 1), z[ok])
        r = float(stats.pearsonr(rx, ry)[0])
        df = int(ok.sum()) - 3
        t_stat = r * np.sqrt(df / max(1e-12, 1.0 - r * r))
        p = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        return r, p

    r_pur, p_pur = partial_r(
        pt_full["appm_core_prot_noIP"].values,
        pt_full["log2_n_hlai_peptides"].values,
        pd.to_numeric(pt_full["Tumor_Purity"], errors="coerce").values,
    )
    r_imm, p_imm = partial_r(
        pt_full["appm_core_prot_noIP"].values,
        pt_full["log2_n_hlai_peptides"].values,
        pd.to_numeric(pt_full["ESTIMATE ImmuneScore"], errors="coerce").values,
    )

    ub_lys_row = [r for r in lys_rows if r["outcome"] == "log2_hlai_peptides" and r["predictor"] == "log2_ub_sites"][0]
    ub_diff_row = [r for r in lys_rows if r["outcome"] == "hlai_minus_hlaii" and r["predictor"] == "log2_ub_sites"][0]
    return {
        "lysine_controlled_hlai_ub_beta": ub_lys_row["std_beta"],
        "lysine_controlled_hlai_ub_p": ub_lys_row["p_value"],
        "lysine_controlled_diff_ub_beta": ub_diff_row["std_beta"],
        "lysine_controlled_diff_ub_p": ub_diff_row["p_value"],
        "exact_2way_noip_beta_prot": float(b_no[0]),
        "exact_2way_noip_se2way_prot": float(se_2w[0]),
        "exact_2way_noip_beta_ub": float(b_no[1]),
        "exact_2way_noip_se2way_ub": float(se_2w[1]),
        "prot_ft_noip_reliability": prot_reliability,
        "iv_2sls_beta_prot": float(b_iv[0]),
        "iv_2sls_beta_ub": float(b_iv[1]),
        "within_exact_allele_n_alleles": len(shared_alleles),
        "within_exact_allele_n_obs": len(al_panel),
        "within_exact_allele_erap1_len_r": float(r_wa),
        "within_exact_allele_erap1_len_p_ols": float(p_wa),
        "within_exact_allele_erap1_len_p_patient_clustered": p_wa_pt_clustered,
        "jackknife_appm_breadth_noip_range": [float(jk_df["r_appm_breadth_noIP"].min()), float(jk_df["r_appm_breadth_noIP"].max())],
        "jackknife_erap1_len_noip_range": [float(jk_df["r_erap1_len_noIP"].min()), float(jk_df["r_erap1_len_noIP"].max())],
        "appm_breadth_partial_r_purity": r_pur,
        "appm_breadth_partial_p_purity": p_pur,
        "appm_breadth_partial_r_immune": r_imm,
        "appm_breadth_partial_p_immune": p_imm,
    }


def run_molar_stoichiometry_and_newey_siblings(cfg: PathResolver, out_dir: Path) -> pd.DataFrame:
    pdir = cfg.processed("visibility_mhc1tip_gene_unique", "pdo")
    samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
    prots = pd.read_csv(pdir / "proteins.tsv.gz", sep="\t")
    res3 = cfg.results("exp3_inflammatory_routing")
    sib_orig = pd.read_csv(res3 / "same_protein_hlab_vs_hlaa_siblings.tsv", sep="\t").set_index("background")
    loc_orig = pd.read_csv(res3 / "locus_channel_reallocation.tsv", sep="\t")

    molar_rows = []
    for bg in ["ITO57T", "ITO66T", "ITO58M"]:
        c_s = samps[(samps["included"] == True) & (samps["background"] == bg) & (samps["modality"] == "protein") & (samps["condition"] == "control")]["sample_id"]
        t_s = samps[(samps["included"] == True) & (samps["background"] == bg) & (samps["modality"] == "protein") & (samps["condition"] == "IFNG")]["sample_id"]
        sub = prots[prots["gene"].isin(["HLA-A", "HLA-B", "HLA-C", "TAPBP"])]
        p0 = sub[sub["sample_id"].isin(c_s)].groupby("gene")["log2_abundance"].mean()
        p1 = sub[sub["sample_id"].isin(t_s)].groupby("gene")["log2_abundance"].mean()
        dM_A = float(2.0 ** p1["HLA-A"] - 2.0 ** p0["HLA-A"])
        dM_B = float(2.0 ** p1["HLA-B"] - 2.0 ** p0["HLA-B"])
        b_dn = float(loc_orig[(loc_orig["background"] == bg) & (loc_orig["locus"] == "HLA-B")]["denovo_to_baseline_ratio"].iloc[0])
        molar_rows.append(
            {
                "background": bg,
                "HLA_A_prot0_log2": float(p0["HLA-A"]),
                "HLA_B_prot0_log2": float(p0["HLA-B"]),
                "baseline_A_over_B_fold": float(2.0 ** (p0["HLA-A"] - p0["HLA-B"])),
                "dLog2_Prot_B_minus_A": float((p1["HLA-B"] - p0["HLA-B"]) - (p1["HLA-A"] - p0["HLA-A"])),
                "delta_molar_B_over_A_ratio": dM_B / dM_A,
                "post_IFNG_B_over_A_molar_ratio": float(2.0 ** (p1["HLA-B"] - p1["HLA-A"])),
                "hlab_denovo_share_ratio": b_dn,
                "same_gene_dPep_B_minus_A": float(sib_orig.loc[bg, "mean_gene_dPep_B_minus_A"]),
            }
        )
    pd.DataFrame(molar_rows).to_csv(out_dir / "pdo_molar_heavy_chain_stoichiometry.tsv", sep="\t", index=False)

    nl = pd.read_csv(cfg.processed("newey2023", "peptides_long.tsv.gz"), sep="\t")
    nl = nl[nl["detected"] & nl["intensity"].gt(0) & nl["gene"].notna() & (~nl["gene"].astype(str).str.contains(";"))].copy()
    nl["log2_int"] = np.log2(nl["intensity"])
    nl["locus"] = nl["predicted_hla"].astype(str).str.extract(r"HLA\.([ABC])")[0]
    piv = nl.pivot_table(index=["organoid", "peptide", "gene", "locus"], columns="condition", values="log2_int").reset_index()

    ext_rows = []
    for org, g in piv.groupby("organoid"):
        if "UT" in g.columns and "IFNg" in g.columns:
            sub = g.dropna(subset=["UT", "IFNg"]).copy()
            sub["dPep"] = sub["IFNg"] - sub["UT"]
            sub_ab = sub[sub["locus"].isin(["A", "B"])].copy()
            gl = sub_ab.groupby(["gene", "locus"])["dPep"].mean().unstack().dropna(subset=["A", "B"])
            if len(gl) >= 10:
                diff = gl["B"] - gl["A"]
                _, pw = stats.wilcoxon(diff)
                mg_genes = set(gl.index)
                pan = sub_ab[sub_ab["gene"].isin(mg_genes)].copy()
                pan["is_B"] = (pan["locus"] == "B").astype(float)
                for col in ["dPep", "is_B", "UT"]:
                    pan[col + "_gm"] = pan[col] - pan.groupby("gene")[col].transform("mean")
                X_anc = np.column_stack([pan["is_B_gm"].values, pan["UT_gm"].values])
                b_anc, _, _, _ = np.linalg.lstsq(X_anc, pan["dPep_gm"].values, rcond=None)
                ext_rows.append(
                    {
                        "cohort": "Newey2023 CRC PDO (external)",
                        "background": org,
                        "n_genes_both_A_and_B": int(len(diff)),
                        "mean_gene_dPep_B_minus_A": float(diff.mean()),
                        "pep0_controlled_ancova_B_minus_A": float(b_anc[0]),
                        "frac_genes_B_gt_A": float((diff > 0).mean()),
                        "wilcoxon_two_sided_p": float(pw),
                    }
                )
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(out_dir / "newey_external_hlab_sibling_validation.tsv", sep="\t", index=False)
    return ext_df


def run_hap1_2x2_sec61_and_scale_free_lengths(cfg: PathResolver, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hdir = cfg.processed("visibility_hap1_gene_unique")
    samps = pd.read_csv(hdir / "samples.tsv", sep="\t")
    samps = samps[samps["included"] == True]
    peps = pd.read_csv(hdir / "peptides.tsv.gz", sep="\t")
    topo = pd.read_csv(cfg.annotations("gene_topology_summary.tsv.gz"), sep="\t").set_index("gene")
    gene_seqs = load_canonical_sequences(cfg)

    peps_pos = peps[peps["normalized_intensity_positive_only"].notna() & (peps["normalized_intensity_positive_only"] > 0)].copy()
    peps_pos["log2_pep"] = np.log2(peps_pos["normalized_intensity_positive_only"])
    wt_pep_s = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()

    wt = (
        peps_pos[peps_pos["sample_id"].isin(wt_pep_s)]
        .groupby(["peptide", "gene"])
        .agg(pep0=("log2_pep", "mean"), n0=("log2_pep", "count"))
        .reset_index()
    )
    wt = wt[wt["n0"] >= 2].copy()
    wt["start_aa"] = wt.apply(
        lambda r: (gene_seqs.get(str(r["gene"]), "").find(str(r["peptide"])) + 1) if gene_seqs.get(str(r["gene"]), "") else 0,
        axis=1,
    )
    wt = wt[wt["start_aa"] >= 1].copy()
    sig_genes = set(topo.index[topo["has_signal_peptide"].astype(bool)])
    wt["nterm_1_35aa"] = wt["start_aa"] <= 35
    wt["uniprot_signal_peptide"] = wt["gene"].isin(sig_genes)

    fac_rows = []
    for ko in ["TAP1_KO", "TAP2_KO", "TAPBP_KO", "ERAP1_KO"]:
        ks = samps[(samps["modality"] == "peptide") & (samps["condition"] == ko)]["sample_id"].tolist()
        kd = peps_pos[peps_pos["sample_id"].isin(ks)].groupby("peptide").agg(pep1=("log2_pep", "mean"), n1=("log2_pep", "count")).reset_index()
        m = wt.merge(kd, on="peptide", how="left")
        m["n1"] = m["n1"].fillna(0).astype(int)
        m["retained_n2"] = m["n1"] >= 2
        m["detected_n1"] = m["n1"] >= 1
        m["dPep"] = m["pep1"] - m["pep0"]
        for nt in [True, False]:
            for sp in [True, False]:
                sub = m[(m["nterm_1_35aa"] == nt) & (m["uniprot_signal_peptide"] == sp)]
                surv2 = sub[sub["retained_n2"]]
                fac_rows.append(
                    {
                        "knockout": ko.replace("_KO", ""),
                        "nterm_1_35aa": nt,
                        "uniprot_signal_peptide": sp,
                        "n_baseline_peptides": int(len(sub)),
                        "retention_n2_of_3": float(sub["retained_n2"].mean()),
                        "detection_n1_of_3": float(sub["detected_n1"].mean()),
                        "n_surviving_n2": int(len(surv2)),
                        "surv_median_dPep_log2": float(surv2["dPep"].median()) if len(surv2) else np.nan,
                        "surv_mean_dPep_log2": float(surv2["dPep"].mean()) if len(surv2) else np.nan,
                    }
                )
    fac_df = pd.DataFrame(fac_rows)
    fac_df.to_csv(out_dir / "hap1_2x2_sec61_signal_peptide_factorial.tsv", sep="\t", index=False)

    peps_pos["len"] = peps_pos["peptide"].str.len()
    p811 = peps_pos[peps_pos["len"].isin([8, 9, 10, 11])].merge(samps[["sample_id", "condition", "run_dates"]], on="sample_id")
    wt_long_frac = float((p811[p811["condition"] == "control"]["len"] >= 10).mean())
    wt_odds = wt_long_frac / (1.0 - wt_long_frac)
    lor_rows = []
    for cond, g in p811.groupby("condition"):
        if cond == "control":
            continue
        lf = float((g["len"] >= 10).mean())
        odds = lf / max(1.0 - lf, 1e-6)
        rep_lf = g.groupby("sample_id")["len"].apply(lambda x: float((x >= 10).mean())).values
        lor_rows.append(
            {
                "knockout": cond.replace("_KO", ""),
                "run_date": str(g["run_dates"].iloc[0]),
                "ko_10_11mer_share": lf,
                "wt_10_11mer_share": wt_long_frac,
                "within_repertoire_10_11mer_odds_ratio_vs_wt": odds / wt_odds,
                "rep_min_10_11mer_share": float(rep_lf.min()),
                "rep_max_10_11mer_share": float(rep_lf.max()),
            }
        )
    lor_df = pd.DataFrame(lor_rows).sort_values("within_repertoire_10_11mer_odds_ratio_vs_wt", ascending=False)
    lor_df.to_csv(out_dir / "hap1_batch_invariant_length_odds_ratios.tsv", sep="\t", index=False)
    return fac_df, lor_df


def run_persistence_stratified_and_disjoint_gene(cfg: PathResolver, out_dir: Path) -> dict:
    topo = pd.read_csv(cfg.annotations("gene_topology_summary.tsv.gz"), sep="\t")
    sec_map = topo.set_index("gene")["is_secretory_er_membrane"].to_dict()
    dfs = [
        build_background_table("MHC1-TIP PDO", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO57T", "ITO57T", cfg.processed("visibility_binding", "gene_unique", "pdo57t_best.tsv.gz"), sec_map),
        build_background_table("MHC1-TIP PDO", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO58M", "ITO58M", cfg.processed("visibility_binding", "gene_unique", "pdo58m_best.tsv.gz"), sec_map),
        build_background_table("MHC1-TIP PDO", cfg.processed("visibility_mhc1tip_gene_unique", "pdo"), "ITO66T", "ITO66T", cfg.processed("visibility_binding", "gene_unique", "pdo66t_best.tsv.gz"), sec_map),
        build_background_table("Chong Ovarian", cfg.processed("visibility_replication", "chong2018_gene_unique"), "UWB.1_289", "UWB1.289", cfg.processed("visibility_binding", "gene_unique", "chong2018_best.tsv.gz"), sec_map),
    ]
    all_df = pd.concat(dfs, ignore_index=True)
    z_feats = ["pep0_z", "full_detect0_z", "prot0_z", "aff_z", "locus_B_z", "cterm_hydro_z", "cterm_acidic_z", "core_pro_z", "len9_z"]

    tercile_rows = []
    disjoint_rows = []

    for target_bin, target_cont in [("retained", "post_level_cens_z"), ("high_persist", "post_level_cens_z"), ("durable", "dPep_cens_z")]:
        for bg in ["ITO57T", "ITO58M", "ITO66T", "UWB1.289"]:
            tr = all_df[all_df["background"] != bg]
            te = all_df[all_df["background"] == bg].copy()
            reg = RidgeCV(alphas=np.logspace(0, 4, 25)).fit(tr[z_feats], tr[target_cont])
            te["s_mech"] = reg.predict(te[z_feats])
            te["tercile"] = pd.qcut(te["pep0_z"], 3, labels=["Low_Pep0", "Mid_Pep0", "High_Pep0"])

            for tlab, g in te.groupby("tercile", observed=False):
                if g[target_bin].nunique() == 2:
                    tercile_rows.append(
                        {
                            "endpoint": target_bin,
                            "held_out_background": bg,
                            "pep0_tercile": str(tlab),
                            "n_peptides": int(len(g)),
                            "auroc_mech": float(roc_auc_score(g[target_bin], g["s_mech"])),
                            "auroc_pep0": float(roc_auc_score(g[target_bin], g["pep0_z"])),
                            "auroc_aff": float(roc_auc_score(g[target_bin], g["aff_z"])),
                            "auroc_prot0": float(roc_auc_score(g[target_bin], g["prot0_z"])),
                        }
                    )

            tr_genes = set(tr["gene"].unique())
            te_disj = te[~te["gene"].isin(tr_genes)]
            if len(te_disj) >= 50 and te_disj[target_bin].nunique() == 2:
                disjoint_rows.append(
                    {
                        "endpoint": target_bin,
                        "held_out_background": bg,
                        "n_disjoint_peptides": int(len(te_disj)),
                        "n_disjoint_genes": int(te_disj["gene"].nunique()),
                        "auroc_mech": float(roc_auc_score(te_disj[target_bin], te_disj["s_mech"])),
                        "auroc_pep0": float(roc_auc_score(te_disj[target_bin], te_disj["pep0_z"])),
                        "auroc_aff": float(roc_auc_score(te_disj[target_bin], te_disj["aff_z"])),
                        "auroc_prot0": float(roc_auc_score(te_disj[target_bin], te_disj["prot0_z"])),
                    }
                )

    t_df = pd.DataFrame(tercile_rows)
    d_df = pd.DataFrame(disjoint_rows)
    t_df.to_csv(out_dir / "persistence_within_pep0_terciles.tsv", sep="\t", index=False)
    d_df.to_csv(out_dir / "persistence_disjoint_gene_transfer.tsv", sep="\t", index=False)

    mid_ret = t_df[(t_df["endpoint"] == "retained") & (t_df["pep0_tercile"] == "Mid_Pep0")]
    mid_dur = t_df[(t_df["endpoint"] == "durable") & (t_df["pep0_tercile"] == "Mid_Pep0")]
    uwb_disj_hp = d_df[(d_df["endpoint"] == "high_persist") & (d_df["held_out_background"] == "UWB1.289")].iloc[0]
    return {
        "mid_tercile_retained_mean_auroc_mech": float(mid_ret["auroc_mech"].mean()),
        "mid_tercile_retained_mean_auroc_pep0": float(mid_ret["auroc_pep0"].mean()),
        "mid_tercile_durable_mean_auroc_mech": float(mid_dur["auroc_mech"].mean()),
        "mid_tercile_durable_mean_auroc_pep0": float(mid_dur["auroc_pep0"].mean()),
        "disjoint_gene_uwb1289_n_peptides": int(uwb_disj_hp["n_disjoint_peptides"]),
        "disjoint_gene_uwb1289_high_persist_auroc_mech": float(uwb_disj_hp["auroc_mech"]),
        "disjoint_gene_uwb1289_high_persist_auroc_prot0": float(uwb_disj_hp["auroc_prot0"]),
    }


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("falsification_controls")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_6a = run_disattenuation_and_sibling_null(cfg, out_dir)
    res_6b = run_monte_controls(cfg, out_dir)
    df_6c = run_molar_stoichiometry_and_newey_siblings(cfg, out_dir)
    fac_6d, lor_6d = run_hap1_2x2_sec61_and_scale_free_lengths(cfg, out_dir)
    res_6e = run_persistence_stratified_and_disjoint_gene(cfg, out_dir)

    tap1_q1 = fac_6d[(fac_6d["knockout"] == "TAP1") & (fac_6d["nterm_1_35aa"] == True) & (fac_6d["uniprot_signal_peptide"] == True)].iloc[0]
    tap1_q2 = fac_6d[(fac_6d["knockout"] == "TAP1") & (fac_6d["nterm_1_35aa"] == True) & (fac_6d["uniprot_signal_peptide"] == False)].iloc[0]
    tap2_q1 = fac_6d[(fac_6d["knockout"] == "TAP2") & (fac_6d["nterm_1_35aa"] == True) & (fac_6d["uniprot_signal_peptide"] == True)].iloc[0]
    tap2_q2 = fac_6d[(fac_6d["knockout"] == "TAP2") & (fac_6d["nterm_1_35aa"] == True) & (fac_6d["uniprot_signal_peptide"] == False)].iloc[0]

    summary = {
        "disattenuation": {
            "raw_r2_range": [float(df_6a["r2_raw"].min()), float(df_6a["r2_raw"].max())],
            "disattenuated_r2_range": [float(df_6a["r2_fully_disattenuated"].min()), float(df_6a["r2_fully_disattenuated"].max())],
            "ifng_fold_over_noise_null_range": [
                float(df_6a[df_6a["perturbation"] == "IFNG"]["fold_over_noise_null"].min()),
                float(df_6a[df_6a["perturbation"] == "IFNG"]["fold_over_noise_null"].max()),
            ],
        },
        "monte_controls": res_6b,
        "newey_external": {
            "crc01_n_genes": int(df_6c[df_6c["background"] == "CRC01"]["n_genes_both_A_and_B"].iloc[0]),
            "crc01_mean_B_minus_A": float(df_6c[df_6c["background"] == "CRC01"]["mean_gene_dPep_B_minus_A"].iloc[0]),
            "crc01_ancova_pep0_adj_B_minus_A": float(df_6c[df_6c["background"] == "CRC01"]["pep0_controlled_ancova_B_minus_A"].iloc[0]),
        },
        "hap1_2x2_sec61": {
            "tap1_true_sigpep_ret": float(tap1_q1["retention_n2_of_3"]),
            "tap1_cytosolic_nterm_ret": float(tap1_q2["retention_n2_of_3"]),
            "tap1_true_sigpep_med_dPep": float(tap1_q1["surv_median_dPep_log2"]),
            "tap1_cytosolic_nterm_med_dPep": float(tap1_q2["surv_median_dPep_log2"]),
            "tap2_true_sigpep_ret": float(tap2_q1["retention_n2_of_3"]),
            "tap2_cytosolic_nterm_ret": float(tap2_q2["retention_n2_of_3"]),
            "tap2_true_sigpep_med_dPep": float(tap2_q1["surv_median_dPep_log2"]),
            "tap2_cytosolic_nterm_med_dPep": float(tap2_q2["surv_median_dPep_log2"]),
            "erap1_batch_invariant_10_11mer_or": float(lor_6d[lor_6d["knockout"] == "ERAP1"]["within_repertoire_10_11mer_odds_ratio_vs_wt"].iloc[0]),
        },
        "persistence_controls": res_6e,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--upstream-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = resolve_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        upstream_dir=args.upstream_dir,
        results_dir=args.results_dir,
    )
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
