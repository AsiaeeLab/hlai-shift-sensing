"""Module 5: Zero-Shot Out-of-Background Prediction of Inflammation-Resilient
Antigen Persistence & Diagnosis of Cross-Platform Censoring Simpson's Paradox.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from presentation_state.config import PathResolver, resolve_config

SEED = 20261116


def zscore(s: pd.Series) -> pd.Series:
    sd = float(s.std())
    return (s - float(s.mean())) / sd if sd > 1e-12 else s * 0.0


def build_background_table(
    cohort: str, pdir: Path, bg_raw: str, bg_clean: str, bfile: Path, sec_map: dict[str, bool]
) -> pd.DataFrame:
    samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
    samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw)]
    peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
    prots = pd.read_csv(pdir / "proteins.tsv.gz", sep="\t")
    bdf = pd.read_csv(bfile, sep="\t").drop_duplicates("peptide")

    ctrl_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
    trt_pep = samps[(samps["modality"] == "peptide") & (samps["condition"] == "IFNG")]["sample_id"].tolist()
    ctrl_prot = samps[(samps["modality"] == "protein") & (samps["condition"] == "control")]["sample_id"].tolist()

    psub = peps[peps["sample_id"].isin(ctrl_pep + trt_pep) & (peps["intensity"] > 0)].copy()
    psub["l2"] = psub["log2_abundance"] if "log2_abundance" in psub.columns else np.log2(psub["intensity"])

    p0 = psub[psub["sample_id"].isin(ctrl_pep)].groupby(["peptide", "gene"]).agg(pep0=("l2", "mean"), n0=("l2", "count")).reset_index()
    p1 = psub[psub["sample_id"].isin(trt_pep)].groupby("peptide").agg(pep1=("l2", "mean"), n1=("l2", "count")).reset_index()

    min_c = int(np.ceil(2 * len(ctrl_pep) / 3))
    min_t = int(np.ceil(2 * len(trt_pep) / 3))
    df = p0[p0["n0"] >= min_c].merge(p1, on="peptide", how="left")
    df["n1"] = df["n1"].fillna(0).astype(int)
    df["retained"] = (df["n1"] >= min_t).astype(int)

    pr0 = (
        prots[prots["sample_id"].isin(ctrl_prot) & np.isfinite(prots["log2_abundance"])]
        .groupby("gene")
        .agg(prot0=("log2_abundance", "mean"), pn0=("log2_abundance", "count"))
        .reset_index()
    )
    pr0 = pr0[pr0["pn0"] >= int(np.ceil(2 * len(ctrl_prot) / 3))]
    df = df.merge(pr0, on="gene", how="inner").merge(bdf, on="peptide", how="inner")

    lod1 = float(psub[psub["sample_id"].isin(trt_pep)]["l2"].quantile(0.01))
    pep1_cens = df["pep1"].fillna(lod1)
    df["post_level_cens_z"] = zscore(pep1_cens)
    df["dPep_cens_z"] = zscore(pep1_cens - df["pep0"])

    med_shift = float((df.loc[df["retained"] == 1, "pep1"] - df.loc[df["retained"] == 1, "pep0"]).median())
    df["rel_dPep"] = (df["pep1"] - df["pep0"]) - med_shift
    med_p1 = float(df.loc[df["retained"] == 1, "pep1"].median())
    df["high_persist"] = ((df["retained"] == 1) & (df["pep1"] >= med_p1)).astype(int)
    df["durable"] = ((df["retained"] == 1) & (df["rel_dPep"] >= -1.0)).astype(int)

    df["raw_log2_aff"] = -np.log2(np.clip(df["affinity_nM"], 1.0, 50000.0))
    df["raw_full_detect0"] = (df["n0"] == len(ctrl_pep)).astype(float)
    df["raw_locus_B"] = df["allele"].astype(str).str.contains(r"B\*").astype(float)
    df["raw_cterm_hydro"] = df["peptide"].str[-1].isin(list("FWYLIVMA")).astype(float)
    df["raw_cterm_acidic"] = df["peptide"].str[-1].isin(list("DE")).astype(float)
    df["raw_core_pro"] = df["peptide"].str[3:7].str.contains("P").astype(float)
    df["raw_len9"] = (df["peptide"].str.len() == 9).astype(float)
    sec_genes = {k for k, v in sec_map.items() if bool(v)}
    df["raw_is_sec"] = df["gene"].isin(sec_genes).astype(float)

    df["pep0_z"] = zscore(df["pep0"])
    df["prot0_z"] = zscore(df["prot0"])
    df["full_detect0_z"] = zscore(df["raw_full_detect0"])
    df["aff_z"] = zscore(df["raw_log2_aff"])
    df["locus_B_z"] = zscore(df["raw_locus_B"])
    df["cterm_hydro_z"] = zscore(df["raw_cterm_hydro"])
    df["cterm_acidic_z"] = zscore(df["raw_cterm_acidic"])
    df["core_pro_z"] = zscore(df["raw_core_pro"])
    df["len9_z"] = zscore(df["raw_len9"])
    df["is_sec_z"] = zscore(df["raw_is_sec"])

    df["cohort"] = cohort
    df["background"] = bg_clean
    return df


def bootstrap_auroc_ci(
    y: np.ndarray, score: np.ndarray, genes: np.ndarray, n_boot: int = 1000, seed: int = SEED
) -> tuple[float, float]:
    """Exact gene-cluster bootstrap 95% CI via multinomial cluster weights."""
    rng = np.random.default_rng(seed)
    _, g_inv = np.unique(genes, return_inverse=True)
    n_g = int(g_inv.max()) + 1
    order = np.argsort(score, kind="mergesort")
    s_ord = score[order]
    y_ord = y[order].astype(float)
    g_ord = g_inv[order]
    mult = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=n_boot).astype(float)
    W = mult[:, g_ord]
    W_pos = W * y_ord[None, :]
    W_neg = W * (1.0 - y_ord[None, :])
    tot_pos = W_pos.sum(axis=1)
    tot_neg = W_neg.sum(axis=1)
    uniq_s, grp_idx = np.unique(s_ord, return_inverse=True)
    if len(uniq_s) == len(s_ord):
        cum_neg = np.cumsum(W_neg, axis=1) - 0.5 * W_neg
        u_stat = np.sum(W_pos * cum_neg, axis=1)
    else:
        n_u = len(uniq_s)
        G_mat = np.zeros((len(s_ord), n_u), dtype=float)
        G_mat[np.arange(len(s_ord)), grp_idx] = 1.0
        gp_pos = W_pos @ G_mat
        gp_neg = W_neg @ G_mat
        cum_neg = np.cumsum(gp_neg, axis=1) - 0.5 * gp_neg
        u_stat = np.sum(gp_pos * cum_neg, axis=1)
    valid = (tot_pos > 0) & (tot_neg > 0)
    aucs = u_stat[valid] / (tot_pos[valid] * tot_neg[valid])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp5_persistence_prediction")
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = pd.read_csv(cfg.annotations("gene_topology_summary.tsv.gz"), sep="\t")
    sec_map = topo.set_index("gene")["is_secretory_er_membrane"].to_dict()

    dfs = [
        build_background_table(
            "MHC1-TIP PDO",
            cfg.processed("visibility_mhc1tip_gene_unique", "pdo"),
            "ITO57T",
            "ITO57T",
            cfg.processed("visibility_binding", "gene_unique", "pdo57t_best.tsv.gz"),
            sec_map,
        ),
        build_background_table(
            "MHC1-TIP PDO",
            cfg.processed("visibility_mhc1tip_gene_unique", "pdo"),
            "ITO58M",
            "ITO58M",
            cfg.processed("visibility_binding", "gene_unique", "pdo58m_best.tsv.gz"),
            sec_map,
        ),
        build_background_table(
            "MHC1-TIP PDO",
            cfg.processed("visibility_mhc1tip_gene_unique", "pdo"),
            "ITO66T",
            "ITO66T",
            cfg.processed("visibility_binding", "gene_unique", "pdo66t_best.tsv.gz"),
            sec_map,
        ),
        build_background_table(
            "Chong Ovarian",
            cfg.processed("visibility_replication", "chong2018_gene_unique"),
            "UWB.1_289",
            "UWB1.289",
            cfg.processed("visibility_binding", "gene_unique", "chong2018_best.tsv.gz"),
            sec_map,
        ),
    ]
    all_df = pd.concat(dfs, ignore_index=True)

    z_feats = [
        "pep0_z",
        "full_detect0_z",
        "prot0_z",
        "aff_z",
        "locus_B_z",
        "cterm_hydro_z",
        "cterm_acidic_z",
        "core_pro_z",
        "len9_z",
    ]
    raw_feats = [
        "pep0_z",
        "raw_full_detect0",
        "prot0_z",
        "raw_log2_aff",
        "raw_locus_B",
        "raw_cterm_hydro",
        "raw_cterm_acidic",
        "raw_core_pro",
        "raw_len9",
    ]
    virt_feats = ["prot0_z", "aff_z", "locus_B_z", "cterm_hydro_z", "cterm_acidic_z", "len9_z"]

    eval_rows = []
    portfolio_rows = []
    coef_rows = []

    for target_bin, target_cont in [
        ("retained", "post_level_cens_z"),
        ("high_persist", "post_level_cens_z"),
        ("durable", "dPep_cens_z"),
    ]:
        for bg in ["ITO57T", "ITO58M", "ITO66T", "UWB1.289"]:
            tr = all_df[all_df["background"] != bg]
            te = all_df[all_df["background"] == bg].copy()

            mech_reg = RidgeCV(alphas=np.logspace(0, 4, 25)).fit(tr[z_feats], tr[target_cont])
            s_mech = mech_reg.predict(te[z_feats])

            virt_reg = RidgeCV(alphas=np.logspace(0, 4, 25)).fit(tr[virt_feats], tr[target_cont])
            s_virt = virt_reg.predict(te[virt_feats])

            unstrat_clf = make_pipeline(
                StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 15), cv=3, max_iter=1000, random_state=SEED)
            ).fit(tr[raw_feats], tr[target_bin])
            s_unstrat = unstrat_clf.predict_proba(te[raw_feats])[:, 1]

            methods = [
                ("Mechanistic_Censored", s_mech),
                ("Baseline_MS_Pep0", te["pep0_z"].values),
                ("MHCflurry_Affinity", te["aff_z"].values),
                ("Virtual_Prot0_plus_Aff", s_virt),
                ("Intracellular_Prot0", te["prot0_z"].values),
                ("Unstratified_Pooled_Logit", s_unstrat),
            ]

            y_true = te[target_bin].values
            genes_te = te["gene"].values
            for m_idx, (mname, sc) in enumerate(methods):
                auc = float(roc_auc_score(y_true, sc))
                ap = float(average_precision_score(y_true, sc))
                lo, hi = bootstrap_auroc_ci(y_true, sc, genes_te, n_boot=1000, seed=SEED + 10 * m_idx)
                eval_rows.append(
                    {
                        "endpoint": target_bin,
                        "held_out_background": bg,
                        "n_peptides": int(len(te)),
                        "base_rate": float(y_true.mean()),
                        "method": mname,
                        "auroc": auc,
                        "auroc_ci95_low": lo,
                        "auroc_ci95_high": hi,
                        "auprc": ap,
                    }
                )
                order = np.argsort(-sc)
                for K in [5, 10, 15, 20, 30, 50]:
                    if K <= len(te):
                        portfolio_rows.append(
                            {
                                "endpoint": target_bin,
                                "held_out_background": bg,
                                "method": mname,
                                "budget_K": K,
                                "portfolio_success_rate": float(y_true[order[:K]].mean()),
                            }
                        )

        mech_full = RidgeCV(alphas=np.logspace(0, 4, 25)).fit(all_df[z_feats], all_df[target_cont])
        unstrat_full = make_pipeline(
            StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 15), cv=4, max_iter=1000, random_state=SEED)
        ).fit(all_df[raw_feats], all_df[target_bin])
        u_coefs = unstrat_full.named_steps["logisticregressioncv"].coef_[0]
        for zf, mc, uc in zip(z_feats, mech_full.coef_, u_coefs):
            coef_rows.append(
                {
                    "endpoint": target_bin,
                    "feature": zf.replace("_z", ""),
                    "within_cohort_censored_coef": float(mc),
                    "unstratified_pooled_logit_coef": float(uc),
                }
            )

    eval_df = pd.DataFrame(eval_rows)
    port_df = pd.DataFrame(portfolio_rows)
    coef_df = pd.DataFrame(coef_rows)
    eval_df.to_csv(out_dir / "lobo_persistence_metrics.tsv", sep="\t", index=False)
    port_df.to_csv(out_dir / "lobo_portfolio_curves.tsv", sep="\t", index=False)
    coef_df.to_csv(out_dir / "simpson_paradox_coefficients.tsv", sep="\t", index=False)

    mean_summary = eval_df.groupby(["endpoint", "method"])["auroc"].mean().unstack().round(4).to_dict("index")
    top20_summary = (
        port_df[port_df["budget_K"] == 20]
        .groupby(["endpoint", "method"])["portfolio_success_rate"]
        .mean()
        .unstack()
        .round(4)
        .to_dict("index")
    )
    summary = {
        "n_total_baseline_peptides": int(len(all_df)),
        "mean_auroc_by_endpoint": mean_summary,
        "mean_top20_portfolio_by_endpoint": top20_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
