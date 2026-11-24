"""Module 2: Degradative Flux (K-epsilon-GG Ubiquitylation), PTM Compartment
Bifurcation, and In Vivo APPM Readout in Primary Human Lung Adenocarcinoma (MONTE).
"""

from __future__ import annotations

import itertools
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

PATIENTS_DOT = [
    "C3N.02145",
    "C3N.00199",
    "C3N.00169",
    "C3N.00547",
    "C3N.00579",
    "C3N.01016",
    "C3L.01632",
    "C3N.01024",
    "C3N.01416",
    "C3L.02549",
]
PATIENTS_DASH = [p.replace(".", "-") for p in PATIENTS_DOT]


def exact_signflip_p(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    obs = abs(vals.mean())
    cnt = 0
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        m = abs(float(np.mean(vals * np.array(signs))))
        if m >= obs - 1e-12:
            cnt += 1
    return cnt / (2 ** n)


def perm_corr_p(x: np.ndarray, y: np.ndarray, n_perm: int = 50000, seed: int = SEED) -> float:
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


def build_gene_patient_matrix(feat_df: pd.DataFrame, val_df: pd.DataFrame, suffix: str = "") -> tuple[pd.DataFrame, pd.Series]:
    f_ok = feat_df[feat_df["source_gene_unambiguous"] == True][["id", "gene"]].copy()
    m = f_ok.merge(val_df, left_on="id", right_on="feature_id", how="inner")
    cols = [p + suffix for p in PATIENTS_DOT]
    site_counts = m.groupby("gene")["id"].nunique().rename("n_sites")
    gmean = m.groupby("gene")[cols].mean()
    gmean.columns = PATIENTS_DASH
    return gmean, site_counts


def run_across_gene_ptm_model(
    pep1: pd.DataFrame,
    pep2: pd.DataFrame,
    prot_f: pd.DataFrame,
    ub_f: pd.DataFrame,
    ph_f: pd.DataFrame,
    ac_f: pd.DataFrame,
    topo: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    pf = (
        prot_f[prot_f["source_gene_unambiguous"] == True]
        .groupby("gene")
        .agg(
            mw=("protein_mw", "median"),
            spectra=("numSpectraProteinObserved", "sum"),
            peps_uniq=("numPepsUnique", "sum"),
        )
    )
    ub_sites = ub_f[ub_f["source_gene_unambiguous"] == True].groupby("gene")["id"].nunique()
    ph_sites = ph_f[ph_f["source_gene_unambiguous"] == True].groupby("gene")["id"].nunique()
    ac_sites = ac_f[ac_f["source_gene_unambiguous"] == True].groupby("gene")["id"].nunique()

    pf["log2_mw"] = np.log2(pf["mw"])
    pf["log2_spec_abundance"] = np.log2((pf["spectra"] + 1.0) / (pf["mw"] / 1000.0))
    pf["log2_ub_sites"] = np.log2(1.0 + ub_sites.reindex(pf.index).fillna(0.0))
    pf["log2_phos_sites"] = np.log2(1.0 + ph_sites.reindex(pf.index).fillna(0.0))
    pf["log2_acetyl_sites"] = np.log2(1.0 + ac_sites.reindex(pf.index).fillna(0.0))

    topo_idx = topo.set_index("gene")
    pf["secretory_er_membrane"] = topo_idx["is_secretory_er_membrane"].astype(float).reindex(pf.index).fillna(0.0)

    pf["log2_hlai_peptides"] = np.log2(1.0 + pep1.groupby("gene")["peptide"].nunique().reindex(pf.index).fillna(0.0))
    pf["log2_hlaii_peptides"] = np.log2(1.0 + pep2.groupby("gene")["peptide"].nunique().reindex(pf.index).fillna(0.0))
    pf["hlai_minus_hlaii"] = pf["log2_hlai_peptides"] - pf["log2_hlaii_peptides"]
    pf = pf.dropna().copy()
    pf.reset_index().to_csv(out_dir / "across_gene_ptm_features.tsv.gz", sep="\t", index=False)

    x_cols = [
        "log2_mw",
        "log2_spec_abundance",
        "log2_ub_sites",
        "log2_phos_sites",
        "log2_acetyl_sites",
        "secretory_er_membrane",
    ]
    X_std = (pf[x_cols] - pf[x_cols].mean()) / pf[x_cols].std()
    X_des = np.column_stack([np.ones(len(pf)), X_std.values])

    rows = []
    for y_col in ["log2_hlai_peptides", "log2_hlaii_peptides", "hlai_minus_hlaii"]:
        yv = ((pf[y_col] - pf[y_col].mean()) / pf[y_col].std()).values
        b, _, _, _ = np.linalg.lstsq(X_des, yv, rcond=None)
        res = yv - X_des @ b
        n, k = X_des.shape
        h = np.sum(X_des * (X_des @ np.linalg.inv(X_des.T @ X_des)), axis=1)
        meat = X_des.T @ ((res / (1.0 - h))[:, None] ** 2 * X_des)
        bread = np.linalg.inv(X_des.T @ X_des)
        cov_hc3 = bread @ meat @ bread
        se = np.sqrt(np.diag(cov_hc3))
        t_stat = b / se
        p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=n - k)
        r2 = float(1.0 - (res @ res) / (yv @ yv))
        for cname, bi, sei, ti, pi in zip(x_cols, b[1:], se[1:], t_stat[1:], p_val[1:]):
            rows.append(
                {
                    "outcome": y_col,
                    "n_genes": int(n),
                    "model_r2": r2,
                    "predictor": cname,
                    "std_beta": float(bi),
                    "hc3_se": float(sei),
                    "ci95_low": float(bi - 1.96 * sei),
                    "ci95_high": float(bi + 1.96 * sei),
                    "t_stat": float(ti),
                    "p_value": float(pi),
                }
            )
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(out_dir / "across_gene_ptm_regression.tsv", sep="\t", index=False)
    return reg_df


def run_within_gene_across_patient_flux(
    pep1: pd.DataFrame,
    pep2: pd.DataFrame,
    prot_ft: pd.DataFrame,
    prot_noip: pd.DataFrame,
    ub_ft: pd.DataFrame,
    ub_noip: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    hlai_int = np.log2(pep1.groupby(["gene", "sample_id"])["intensity"].sum()).unstack()
    hlai_norm = hlai_int.sub(hlai_int.median(axis=0), axis=1)
    hlai_rel = hlai_norm.sub(hlai_norm.mean(axis=1), axis=0)

    hlaii_int = np.log2(pep2.groupby(["gene", "sample_id"])["intensity"].sum()).unstack()
    hlaii_norm = hlaii_int.sub(hlaii_int.median(axis=0), axis=1)
    hlaii_rel = hlaii_norm.sub(hlaii_norm.mean(axis=1), axis=0)

    results = []
    pt_slope_rows = []
    for hla_name, hla_rel in [("HLA-I", hlai_rel), ("HLA-II", hlaii_rel)]:
        for wf_name, pr_mat, ub_mat in [("FT", prot_ft, ub_ft), ("noIP", prot_noip, ub_noip)]:
            p_s = pr_mat.sub(pr_mat.mean(axis=1), axis=0).stack().rename("prot")
            u_s = ub_mat.sub(ub_mat.mean(axis=1), axis=0).stack().rename("ub")
            h_s = hla_rel.stack().rename("hla")
            panel = pd.concat([h_s, p_s, u_s], axis=1).dropna()

            X = np.column_stack([np.ones(len(panel)), panel["prot"].values, panel["ub"].values])
            y = panel["hla"].values
            b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            n, k = X.shape
            r2 = float(1.0 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))

            bread = np.linalg.inv(X.T @ X)
            genes = panel.index.get_level_values(0)
            df_x = pd.DataFrame(X * res[:, None], index=genes)
            g_sums = df_x.groupby(level=0).sum().values
            G = len(g_sums)
            scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
            cov_cl = scale * (bread @ (g_sums.T @ g_sums) @ bread)
            se_cl = np.sqrt(np.diag(cov_cl))
            t_cl = b / se_cl
            p_cl = 2.0 * stats.t.sf(np.abs(t_cl), df=G - 1)

            pt_b_prot, pt_b_ub = [], []
            for pt in PATIENTS_DASH:
                if pt in panel.index.get_level_values(1):
                    sub = panel.xs(pt, level=1)
                    if len(sub) >= 20:
                        Xp = np.column_stack([np.ones(len(sub)), sub["prot"].values, sub["ub"].values])
                        bp, _, _, _ = np.linalg.lstsq(Xp, sub["hla"].values, rcond=None)
                        pt_b_prot.append(bp[1])
                        pt_b_ub.append(bp[2])
                        pt_slope_rows.append(
                            {
                                "hla_class": hla_name,
                                "workflow": wf_name,
                                "patient": pt,
                                "n_genes": int(len(sub)),
                                "beta_prot": float(bp[1]),
                                "beta_ub": float(bp[2]),
                            }
                        )

            results.append(
                {
                    "hla_class": hla_name,
                    "workflow": wf_name,
                    "n_pairs": int(n),
                    "n_genes": int(G),
                    "joint_r2": r2,
                    "beta_prot": float(b[1]),
                    "se_prot_gene_clustered": float(se_cl[1]),
                    "p_prot_gene_clustered": float(p_cl[1]),
                    "patients_positive_prot": int(np.sum(np.array(pt_b_prot) > 0)),
                    "signflip_p_prot": float(exact_signflip_p(np.array(pt_b_prot))),
                    "beta_ub": float(b[2]),
                    "se_ub_gene_clustered": float(se_cl[2]),
                    "p_ub_gene_clustered": float(p_cl[2]),
                    "patients_positive_ub": int(np.sum(np.array(pt_b_ub) > 0)),
                    "signflip_p_ub": float(exact_signflip_p(np.array(pt_b_ub))),
                }
            )

    hlai_p = np.log2(pep1.pivot(index="peptide", columns="sample_id", values="intensity"))
    hlai_p_norm = hlai_p.sub(hlai_p.median(axis=0), axis=1)
    hlai_p_sub = hlai_p_norm[hlai_p_norm.notna().sum(axis=1) >= 4].copy()
    hlai_p_rel = hlai_p_sub.sub(hlai_p_sub.mean(axis=1), axis=0)
    pep_gene = pep1.drop_duplicates("peptide").set_index("peptide")["gene"].to_dict()

    same_pep_rows = []
    for pep_seq, row in hlai_p_rel.iterrows():
        g = pep_gene.get(pep_seq)
        if g in prot_noip.index and g in ub_noip.index:
            p_cent = prot_noip.loc[g] - prot_noip.loc[g].mean()
            u_cent = ub_noip.loc[g] - ub_noip.loc[g].mean()
            for pt, val in row.dropna().items():
                if np.isfinite(p_cent[pt]) and np.isfinite(u_cent[pt]):
                    same_pep_rows.append((pep_seq, g, pt, float(val), float(p_cent[pt]), float(u_cent[pt])))
    sp_df = pd.DataFrame(same_pep_rows, columns=["peptide", "gene", "patient", "hla_pep", "prot", "ub"])
    Xsp = np.column_stack([np.ones(len(sp_df)), sp_df["prot"].values, sp_df["ub"].values])
    ysp = sp_df["hla_pep"].values
    bsp, _, _, _ = np.linalg.lstsq(Xsp, ysp, rcond=None)
    rss = np.sum((ysp - Xsp @ bsp) ** 2)
    tss = np.sum((ysp - ysp.mean()) ** 2)
    sp_df.to_csv(out_dir / "identical_peptide_across_patient_noip.tsv.gz", sep="\t", index=False)

    res_df = pd.DataFrame(results)
    res_df["identical_peptide_n_obs"] = int(len(sp_df))
    res_df["identical_peptide_n_peptides"] = int(sp_df["peptide"].nunique())
    res_df["identical_peptide_r2"] = float(1.0 - rss / tss)
    res_df["identical_peptide_beta_prot"] = float(bsp[1])
    res_df["identical_peptide_beta_ub"] = float(bsp[2])
    res_df.to_csv(out_dir / "within_gene_across_patient_flux.tsv", sep="\t", index=False)
    pd.DataFrame(pt_slope_rows).to_csv(out_dir / "per_patient_flux_slopes.tsv", sep="\t", index=False)
    return res_df


def run_tumor_appm_readout(
    pep1: pd.DataFrame, prot_f: pd.DataFrame, prot_m: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    f_ok = prot_f[prot_f["gene"].notna()][["id", "gene"]].copy()
    m_pr = f_ok.merge(prot_m, left_on="id", right_on="feature_id", how="inner")
    pr_ft = m_pr.groupby("gene")[PATIENTS_DOT].mean()
    pr_ft.columns = PATIENTS_DASH
    pr_noip = m_pr.groupby("gene")[[p + ".N" for p in PATIENTS_DOT]].mean()
    pr_noip.columns = PATIENTS_DASH

    pep1 = pep1.copy()
    pep1["is_long_10_12"] = (pep1["length"] >= 10).astype(float)
    pep1["is_hydro_c"] = pep1["peptide"].str[-1].isin(list("FWYLIVMA")).astype(float)
    pep1["locus"] = pep1["allele"].astype(str).str.extract(r"^([ABC])")[0]

    pt_df = pd.DataFrame(index=PATIENTS_DASH)
    pt_df["n_hlai_peptides"] = pep1.groupby("sample_id")["peptide"].nunique()
    pt_df["log2_n_hlai_peptides"] = np.log2(pt_df["n_hlai_peptides"])
    pt_df["mean_peptide_length"] = pep1.groupby("sample_id")["length"].mean()
    pt_df["frac_long_10_12"] = pep1.groupby("sample_id")["is_long_10_12"].mean()
    pt_df["frac_hydro_cterm"] = pep1.groupby("sample_id")["is_hydro_c"].mean()
    frac_b = pep1.groupby("sample_id")["locus"].apply(lambda s: (s == "B").mean())
    frac_a = pep1.groupby("sample_id")["locus"].apply(lambda s: (s == "A").mean())
    pt_df["log2_hlab_over_hlaa_peptides"] = np.log2(frac_b / frac_a)

    for wf, pr in [("FT", pr_ft), ("noIP", pr_noip)]:
        pt_df[f"erap1_prot_{wf}"] = pr.loc["ERAP1"]
        ap_genes = [g for g in ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C"] if g in pr.index]
        pt_df[f"appm_core_prot_{wf}"] = pr.loc[ap_genes].mean(axis=0)
        pt_df[f"hlab_minus_hlaa_prot_{wf}"] = pr.loc["HLA-B"] - pr.loc["HLA-A"]
        pt_df[f"ip_minus_cp_prot_{wf}"] = (
            pr.loc[["PSMB8", "PSMB9", "PSMB10"]].mean(axis=0) - pr.loc[["PSMB5", "PSMB6", "PSMB7"]].mean(axis=0)
        )

    pt_df.reset_index().rename(columns={"index": "sample_id"}).to_csv(
        out_dir / "monte_patient_appm_and_ligandome.tsv", sep="\t", index=False
    )

    tests = []
    pairs_to_test = [
        ("APPM core protein vs log2(HLA-I breadth)", "appm_core_prot", "log2_n_hlai_peptides"),
        ("ERAP1 protein vs mean peptide length", "erap1_prot", "mean_peptide_length"),
        ("ERAP1 protein vs untrimmed 10-12mer fraction", "erap1_prot", "frac_long_10_12"),
        ("(HLA-B - HLA-A) protein vs log2(HLA-B / HLA-A ligands)", "hlab_minus_hlaa_prot", "log2_hlab_over_hlaa_peptides"),
        ("(Immunoproteasome - Constitutive) vs Hydrophobic C-term", "ip_minus_cp_prot", "frac_hydro_cterm"),
    ]
    for idx_t, (desc, x_base, y_col) in enumerate(pairs_to_test):
        for wf in ["FT", "noIP"]:
            x = pt_df[f"{x_base}_{wf}"].values
            y = pt_df[y_col].values
            r, p = stats.pearsonr(x, y)
            rho, sp = stats.spearmanr(x, y)
            pp = perm_corr_p(x, y, n_perm=50000, seed=SEED + 100 * idx_t + (0 if wf == "FT" else 1))
            tests.append(
                {
                    "hypothesis": desc,
                    "workflow": wf,
                    "n_patients": len(pt_df),
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "perm_p_two_sided": float(pp),
                    "spearman_rho": float(rho),
                    "spearman_p": float(sp),
                }
            )
    t_df = pd.DataFrame(tests)
    t_df.to_csv(out_dir / "monte_appm_readout_tests.tsv", sep="\t", index=False)
    return t_df


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp2_monte_degradative_flux")
    out_dir.mkdir(parents=True, exist_ok=True)

    monte_dir = cfg.processed("visibility_monte")
    pep1 = pd.read_csv(monte_dir / "peptides_global_source_strict.tsv.gz", sep="\t")
    pep2 = pd.read_csv(cfg.annotations("monte_hlaii_strict.tsv.gz"), sep="\t")
    topo = pd.read_csv(cfg.annotations("gene_topology_summary.tsv.gz"), sep="\t")

    prot_f = pd.read_csv(monte_dir / "proteome_feature_annotations.tsv.gz", sep="\t", low_memory=False)
    prot_m = pd.read_csv(monte_dir / "proteome_log2_ratios.tsv.gz", sep="\t")
    ub_f = pd.read_csv(monte_dir / "ubiquitylome_feature_annotations.tsv.gz", sep="\t", low_memory=False)
    ub_m = pd.read_csv(monte_dir / "ubiquitylome_log2_ratios.tsv.gz", sep="\t")
    ph_f = pd.read_csv(monte_dir / "phosphoproteome_feature_annotations.tsv.gz", sep="\t", low_memory=False)
    ac_f = pd.read_csv(monte_dir / "acetylome_feature_annotations.tsv.gz", sep="\t", low_memory=False)

    prot_ft, _ = build_gene_patient_matrix(prot_f, prot_m, "")
    prot_noip, _ = build_gene_patient_matrix(prot_f, prot_m, ".N")
    ub_ft, _ = build_gene_patient_matrix(ub_f, ub_m, "")
    ub_noip, _ = build_gene_patient_matrix(ub_f, ub_m, ".N")

    reg_df = run_across_gene_ptm_model(pep1, pep2, prot_f, ub_f, ph_f, ac_f, topo, out_dir)
    flux_df = run_within_gene_across_patient_flux(pep1, pep2, prot_ft, prot_noip, ub_ft, ub_noip, out_dir)
    appm_df = run_tumor_appm_readout(pep1, prot_f, prot_m, out_dir)

    summary = {
        "n_genes_across_gene_model": int(reg_df["n_genes"].iloc[0]),
        "hlai_ub_std_beta": float(
            reg_df[(reg_df["outcome"] == "log2_hlai_peptides") & (reg_df["predictor"] == "log2_ub_sites")]["std_beta"].iloc[0]
        ),
        "within_gene_noip_hlai_beta_ub": float(
            flux_df[(flux_df["hla_class"] == "HLA-I") & (flux_df["workflow"] == "noIP")]["beta_ub"].iloc[0]
        ),
        "within_gene_noip_hlai_p_ub": float(
            flux_df[(flux_df["hla_class"] == "HLA-I") & (flux_df["workflow"] == "noIP")]["p_ub_gene_clustered"].iloc[0]
        ),
        "identical_peptide_r2": float(flux_df["identical_peptide_r2"].iloc[0]),
        "erap1_vs_mean_len_noip_r": float(
            appm_df[(appm_df["hypothesis"] == "ERAP1 protein vs mean peptide length") & (appm_df["workflow"] == "noIP")]["pearson_r"].iloc[0]
        ),
        "appm_vs_breadth_ft_r": float(
            appm_df[(appm_df["hypothesis"] == "APPM core protein vs log2(HLA-I breadth)") & (appm_df["workflow"] == "FT")]["pearson_r"].iloc[0]
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
