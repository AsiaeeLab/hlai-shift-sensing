"""Module 7: Direct Empirical Evaluation of Distributional Cellular-State Sensing.

Implements the five-part public-data distributional follow-up across nine multimodal
human cohorts:
  7A. External multi-omic perturbation elasticity & dynamic ubiquitylation tests.
  7B. Bounded 6-rung representation ladder (with nested inner-fold penalty selection).
  7C. Cross-intervention, cross-lineage, I=L MS-isobaric harmonization, within-vs-between
      HLA locus decomposition, SK-Mel-28 wt vs dr resistance challenge, and continuous
      program-shift recovery.
  7D. Distributed-mixture vs few-marker redundancy tests (gene-disjoint 20% splits,
      top-gene ablations, and subsampled peptide budgets M = 25..2000).
  7E. Temporal trajectories in Bernhardt 2025 pSILAC & Stopfer 2022 xenografts.
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold

from presentation_state.config import PathResolver, resolve_config
from presentation_state.representations import AMINO_ACIDS, CLASS, sequence_map

SEED = 20261116
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
AA_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
MDA_PANEL = {"PMEL", "DCT", "TYRP1", "MLANA", "TYR", "SOX10", "RAB38", "MITF"}


def load_hallmark_sets(cfg: PathResolver) -> dict[str, set[str]]:
    candidates = [
        cfg.processed("visibility_source_audit", "hallmark_2023.2_four_programs_resolved.gmt"),
        cfg.raw("visibility_mhc1tip", "MB_39_TGFb_mhc1tip", "h.all.v2023.2.Hs.symbols.gmt"),
    ]
    for root in (cfg.processed_dir, cfg.data_dir / "processed", cfg.raw_dir):
        if root.exists():
            for dirpath, _, fnames in os.walk(root, followlinks=True):
                for fn in sorted(fnames):
                    if "hallmark" in fn.lower() and (fn.endswith(".gmt") or fn.endswith(".gmt.gz")):
                        candidates.append(Path(dirpath) / fn)
    sets: dict[str, set[str]] = {}
    for p in candidates:
        if p.exists():
            opener = (lambda f: gzip.open(f, "rt")) if str(p).endswith(".gz") else (lambda f: open(f, "r"))
            with opener(p) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        sets[parts[0].replace("HALLMARK_", "")] = set(parts[2:])
            if sets:
                return sets
    return sets


def encode_9mer_features(peps: list[str]) -> dict[str, np.ndarray]:
    """Compute fixed sequence feature matrices (n_peps x d) for canonical 9-mers."""
    n = len(peps)
    coarse96, _ = sequence_map(peps, "coarse96")
    aa20, _ = sequence_map(peps, "aa20")
    pos80, _ = sequence_map(peps, "position80")
    anchor16, _ = sequence_map(peps, "anchor16")
    pos180 = np.zeros((n, 180), dtype=float)
    anchor40 = np.zeros((n, 40), dtype=float)
    krl_count = np.zeros((n, 3), dtype=float)
    for r, p in enumerate(peps):
        for pos in range(9):
            pos180[r, 20 * pos + AA_IDX[p[pos]]] = 1.0
        anchor40[r, AA_IDX[p[1]]] = 1.0
        anchor40[r, 20 + AA_IDX[p[8]]] = 1.0
        krl_count[r, 0] = p.count("K")
        krl_count[r, 1] = p.count("R")
        krl_count[r, 2] = p.count("L")
    return {
        "coarse96": coarse96,
        "aa20": aa20,
        "pos80": pos80,
        "anchor16": anchor16,
        "anchor40": anchor40,
        "pos180": pos180,
        "krl_count": krl_count,
    }


class _IsometricLogReg:
    def __init__(self, clf, v_proj):
        self.clf = clf
        self.v_proj = v_proj

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(Z @ self.v_proj if self.v_proj is not None else Z)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.clf.predict(Z @ self.v_proj if self.v_proj is not None else Z)


def fit_nested_grouped_logreg(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c_grid=C_GRID):
    """Select L2 penalty C via inner leave-one-group-out log-loss, then refit on all training data."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    keep = (sd > 1e-12).ravel()
    if not keep.any():
        keep = np.ones(X.shape[1], dtype=bool)
        sd = np.ones_like(sd)
    Z_full = (X[:, keep] - mu[:, keep]) / sd[:, keep]
    v_proj = None
    if Z_full.shape[1] > 250:
        u_mat, s_vals, vt_mat = np.linalg.svd(Z_full, full_matrices=False)
        Z_work = u_mat * s_vals[None, :]
        v_proj = vt_mat.T
    else:
        Z_work = Z_full

    uniq_g = sorted(np.unique(groups))
    valid_g = [g for g in uniq_g if len(np.unique(y[groups != g])) >= 2]
    best_c = 1.0
    if len(valid_g) >= 2:
        losses_by_c = []
        for c in c_grid:
            fold_losses = []
            for g in valid_g:
                itr = groups != g
                ite = groups == g
                clf = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=SEED)
                clf.fit(Z_work[itr], y[itr])
                probs = np.clip(clf.predict_proba(Z_work[ite])[:, 1], 1e-7, 1.0 - 1e-7)
                fold_losses.append(float(-np.mean(y[ite] * np.log(probs) + (1 - y[ite]) * np.log(1 - probs))))
            losses_by_c.append(float(np.mean(fold_losses)))
        best_c = c_grid[int(np.argmin(np.round(losses_by_c, 10)))]

    clf_final = LogisticRegression(C=best_c, max_iter=2000, solver="liblinear", random_state=SEED)
    clf_final.fit(Z_work, y)
    return _IsometricLogReg(clf_final, v_proj), mu, sd, keep, best_c


def run_exp7a(cfg: PathResolver, out_dir: Path) -> dict:
    s_pep = pd.read_csv(cfg.followup("stopfer2022_peptides.tsv.gz"), sep="\t")
    sk5_om = pd.read_csv(cfg.followup("stopfer2022_skmel5_multiomics.tsv.gz"), sep="\t")

    sk5_p = s_pep[s_pep["tab"] == "sd01_8"].copy()
    sk5_p["l2"] = np.log2(sk5_p["intensity"])
    dmso = sk5_p[sk5_p["condition"] == "DMSO"].groupby(["peptide", "gene"], as_index=False).agg(
        pep0=("l2", "mean"), n0=("l2", "count"), sd0=("l2", "std")
    )
    meki = sk5_p[sk5_p["condition"] == "Binimetinib"].groupby(["peptide", "gene"], as_index=False).agg(
        pep1=("l2", "mean"), n1=("l2", "count"), sd1=("l2", "std")
    )
    m_sk5 = dmso.merge(meki, on=["peptide", "gene"], how="inner")
    m_sk5 = m_sk5[(m_sk5["n0"] >= 2) & (m_sk5["n1"] >= 2) & (m_sk5["gene"] != "") & (~m_sk5["gene"].str.contains(";"))].copy()
    m_sk5["dPep"] = m_sk5["pep1"] - m_sk5["pep0"]
    m_sk5 = m_sk5.merge(sk5_om, on="gene", how="left")

    sub_prot = m_sk5.dropna(subset=["prot_l2fc_100nM"]).copy()
    lr_prot = stats.linregress(sub_prot["prot_l2fc_100nM"], sub_prot["dPep"])
    pep_err_v = float(np.nanmedian((sub_prot["sd0"] ** 2) / sub_prot["n0"] + (sub_prot["sd1"] ** 2) / sub_prot["n1"]))
    rho_pep = float(max(0.1, 1.0 - pep_err_v / np.var(sub_prot["dPep"], ddof=1)))
    prot_err_v = float(np.nanmedian(sub_prot["prot_se_100nM"] ** 2))
    rho_prot = float(max(0.1, 1.0 - prot_err_v / np.var(sub_prot["prot_l2fc_100nM"], ddof=1)))
    r2_prot_raw = float(lr_prot.rvalue ** 2)
    r2_prot_disatt = float(min(1.0, r2_prot_raw / (rho_pep * rho_prot)))

    multi_g = sub_prot.groupby("gene").filter(lambda g: len(g) >= 2)
    sib_ranges = multi_g.groupby("gene")["dPep"].apply(lambda s: float(s.max() - s.min()))
    frac_gt2_sk5 = float((sib_ranges > 1.0).mean())

    sub_rna = m_sk5.dropna(subset=["rna_l2fc_100nM"]).copy()
    lr_rna = stats.linregress(sub_rna["rna_l2fc_100nM"], sub_rna["dPep"])

    sub_ub = m_sk5.dropna(subset=["prot_l2fc_100nM", "ub_l2fc_100nM"]).copy()
    X_ub = np.column_stack(
        [
            np.ones(len(sub_ub)),
            (sub_ub["prot_l2fc_100nM"] - sub_ub["prot_l2fc_100nM"].mean()) / sub_ub["prot_l2fc_100nM"].std(),
            (sub_ub["ub_l2fc_100nM"] - sub_ub["ub_l2fc_100nM"].mean()) / sub_ub["ub_l2fc_100nM"].std(),
        ]
    )
    y_ub = (sub_ub["dPep"].values - sub_ub["dPep"].mean()) / sub_ub["dPep"].std()
    beta_ub_ols, _, _, _ = np.linalg.lstsq(X_ub, y_ub, rcond=None)
    resid_ub = y_ub - X_ub @ beta_ub_ols
    genes_u = sub_ub["gene"].values
    uniq_g = np.unique(genes_u)
    XtX_inv = np.linalg.inv(X_ub.T @ X_ub)
    meat = np.zeros((3, 3))
    for g in uniq_g:
        idx = genes_u == g
        score_g = X_ub[idx].T @ resid_ub[idx]
        meat += np.outer(score_g, score_g)
    dof_scale = (len(uniq_g) / (len(uniq_g) - 1)) * ((len(y_ub) - 1) / (len(y_ub) - 3))
    cov_cl = dof_scale * (XtX_inv @ meat @ XtX_inv)
    se_cl = np.sqrt(np.diag(cov_cl))
    t_prot_ub = float(beta_ub_ols[1] / se_cl[1])
    t_ub_ub = float(beta_ub_ols[2] / se_cl[2])
    p_prot_ub = float(2 * stats.t.sf(abs(t_prot_ub), df=len(uniq_g) - 1))
    p_ub_ub = float(2 * stats.t.sf(abs(t_ub_ub), df=len(uniq_g) - 1))
    r2_prot_only = float(stats.linregress(sub_ub["prot_l2fc_100nM"], sub_ub["dPep"]).rvalue ** 2)
    r2_prot_plus_ub = float(1.0 - np.sum(resid_ub ** 2) / np.sum(y_ub ** 2))

    b_pep = pd.read_csv(cfg.followup("bernhardt2025_psilac_peptides.tsv.gz"), sep="\t")
    b_prot = pd.read_csv(cfg.followup("bernhardt2025_mamel63a_ifng_proteome.tsv.gz"), sep="\t")
    b_ifng_rows = []
    for tag, th, pcol in [("Treat_T1", 24, "prot_l2fc_ifng_24h"), ("Treat_T2", 72, "prot_l2fc_ifng_72h")]:
        sub = b_pep[(b_pep["table"] == "Table4") & (b_pep["experiment_tag"] == tag) & (b_pep["ratio_reliable"] == 1)].copy()
        sub = sub.dropna(subset=["l2fc_trt_vs_ref"]).merge(b_prot[["gene", pcol]].dropna(), on="gene", how="inner")
        lr_b = stats.linregress(sub[pcol], sub["l2fc_trt_vs_ref"])
        mg = sub.groupby("gene").filter(lambda g: len(g) >= 2)
        sr = mg.groupby("gene")["l2fc_trt_vs_ref"].apply(lambda s: float(s.max() - s.min()))
        sub["locus"] = sub["hla_allele"].astype(str).str.extract(r"^HLA-([ABC])")[0]
        ab = sub[sub["locus"].isin(["A", "B"])].groupby(["gene", "locus"])["l2fc_trt_vs_ref"].mean().unstack()
        ab = ab.dropna(subset=["A", "B"])
        diff_ba = ab["B"] - ab["A"]
        _, p_wil = stats.wilcoxon(diff_ba) if len(diff_ba) >= 10 else (np.nan, np.nan)
        b_ifng_rows.append(
            {
                "cohort": f"Bernhardt2025_MaMel63a_IFNg_{th}h",
                "n_peptides": int(len(sub)),
                "n_genes": int(sub["gene"].nunique()),
                "slope_dPep_dProt": float(lr_b.slope),
                "r_squared": float(lr_b.rvalue ** 2),
                "frac_multi_genes_gt_2fold": float((sr > 1.0).mean()),
                "n_same_gene_AB_pairs": int(len(diff_ba)),
                "mean_same_gene_B_minus_A": float(diff_ba.mean()),
                "frac_B_gt_A": float((diff_ba > 0).mean()),
                "wilcoxon_p_B_vs_A": float(p_wil),
            }
        )

    df_7a = pd.DataFrame(
        [
            {
                "cohort": "Stopfer2022_SKMEL5_MEKi_100nM_72h",
                "n_peptides": int(len(sub_prot)),
                "n_genes": int(sub_prot["gene"].nunique()),
                "slope_dPep_dProt": float(lr_prot.slope),
                "r_squared": r2_prot_raw,
                "disattenuated_r_squared": r2_prot_disatt,
                "rna_r_squared": float(lr_rna.rvalue ** 2),
                "frac_multi_genes_gt_2fold": frac_gt2_sk5,
                "ub_joint_n_peptides": int(len(sub_ub)),
                "ub_joint_n_genes": int(len(uniq_g)),
                "std_beta_dProt": float(beta_ub_ols[1]),
                "p_dProt_clustered": p_prot_ub,
                "std_beta_dUb": float(beta_ub_ols[2]),
                "p_dUb_clustered": p_ub_ub,
                "r2_dProt_only_on_ub_subset": r2_prot_only,
                "r2_dProt_plus_dUb": r2_prot_plus_ub,
            }
        ]
        + b_ifng_rows
    )
    df_7a.to_csv(out_dir / "exp7a_followup_elasticity_and_ub.tsv", sep="\t", index=False)
    return {
        "skmel5_meki_prot_r2": r2_prot_raw,
        "skmel5_meki_prot_disatt_r2": r2_prot_disatt,
        "skmel5_meki_rna_r2": float(lr_rna.rvalue ** 2),
        "skmel5_meki_sibling_gt2fold": frac_gt2_sk5,
        "skmel5_dynamic_ub": {
            "n_peptides": int(len(sub_ub)),
            "n_genes": int(len(uniq_g)),
            "std_beta_dProt": float(beta_ub_ols[1]),
            "p_dProt": p_prot_ub,
            "std_beta_dUb": float(beta_ub_ols[2]),
            "p_dUb": p_ub_ub,
            "r2_dProt_only": r2_prot_only,
            "r2_dProt_plus_dUb": r2_prot_plus_ub,
        },
        "bernhardt_mamel63a_ifng": b_ifng_rows,
    }


def build_newey_rungs(cfg: PathResolver) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    ndir = cfg.processed("newey2019_replicates")
    samps = pd.read_csv(ndir / "samples.tsv", sep="\t")
    samps = samps[samps["condition"].isin(["IFNg", "TRAM"])].sort_values("sample_id").reset_index(drop=True)
    peps = pd.read_csv(ndir / "peptides_long.tsv.gz", sep="\t")
    peps = peps[peps["sample_id"].isin(samps["sample_id"]) & (peps["intensity"] > 0)].copy()

    s_idx = {sid: i for i, sid in enumerate(samps["sample_id"])}
    nuis = np.zeros((len(samps), 6), dtype=float)
    for sid, grp in peps.groupby("sample_id"):
        r = s_idx[sid]
        nuis[r, 0] = np.log2(grp["intensity"].sum())
        nuis[r, 1] = np.log2(len(grp))
        L = grp["length"]
        nuis[r, 2] = (L == 8).mean()
        nuis[r, 3] = (L == 9).mean()
        nuis[r, 4] = (L == 10).mean()
        nuis[r, 5] = (L >= 11).mean()

    p9 = peps[peps["length"] == 9].copy()
    p9 = p9[p9["peptide"].map(lambda s: len(str(s)) == 9 and set(str(s)) <= set(AMINO_ACIDS))]
    piv = p9.pivot_table(index="sample_id", columns="peptide", values="intensity", aggfunc="sum", fill_value=0.0)
    piv = piv.loc[samps["sample_id"]]
    seq_list = list(piv.columns)
    sig = piv.values
    pi = sig / sig.sum(axis=1, keepdims=True)
    pres = (sig > 0).astype(float)
    pres = pres / pres.sum(axis=1, keepdims=True)

    fmaps = encode_9mer_features(seq_list)
    rng = np.random.default_rng(SEED)
    sub500 = np.zeros((len(samps), 96), dtype=float)
    for r in range(len(samps)):
        det_idx = np.flatnonzero(sig[r] > 0)
        acc = np.zeros(96, dtype=float)
        for _ in range(10):
            pick = rng.choice(det_idx, size=min(500, len(det_idx)), replace=False)
            w = sig[r, pick] / sig[r, pick].sum()
            acc += w @ fmaps["coarse96"][pick]
        sub500[r] = acc / 10.0

    pep_to_gene = p9.groupby("peptide")["gene"].first().reindex(seq_list).fillna("").tolist()
    uniq_genes = sorted({g for g in pep_to_gene if g and ";" not in g})
    g_idx = {g: i for i, g in enumerate(uniq_genes)}
    gene_mat = np.zeros((len(samps), len(uniq_genes)), dtype=float)
    for j, g in enumerate(pep_to_gene):
        if g in g_idx:
            gene_mat[:, g_idx[g]] += pi[:, j]

    rungs = {
        "R0_Quantity_Length": nuis,
        "R1_Coarse96": pi @ fmaps["coarse96"],
        "R1b_Coarse96_Presence": pres @ fmaps["coarse96"],
        "R1c_Coarse96_Sub500": sub500,
        "R2a_AA20": pi @ fmaps["aa20"],
        "R2b_Pos80_Internal": pi @ fmaps["pos80"],
        "R2c_Anchor40": pi @ fmaps["anchor40"],
        "R2_PosAA180": pi @ fmaps["pos180"],
        "R4_Peptide_Identity": pi,
        "R5_SourceGene_Oracle": gene_mat,
    }
    return samps, rungs, p9


def _paired_concordance_nested(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    org_list = sorted(np.unique(groups))
    pairs = {}
    for o in org_list:
        pos = np.flatnonzero((groups == o) & (y == 1))
        neg = np.flatnonzero((groups == o) & (y == 0))
        pairs[o] = np.array([X[i] - X[j] for i, j in itertools.product(pos, neg)])

    def fit_p(tr_orgs, c_val):
        rms = np.sqrt(np.mean([np.mean(pairs[g] ** 2, axis=0) for g in tr_orgs], axis=0))
        keep = np.isfinite(rms) & (rms > 1e-12)
        xs, ys, ws = [], [], []
        for g in tr_orgs:
            dn = pairs[g][:, keep] / rms[keep]
            xs.extend([dn, -dn])
            ys.extend([np.ones(len(dn), int), np.zeros(len(dn), int)])
            w = np.full(len(dn), 1.0 / (2 * len(tr_orgs) * len(dn)))
            ws.extend([w, w])
        X_cat = np.concatenate(xs)
        v_proj = None
        if X_cat.shape[1] > 250:
            u_m, s_v, vt_m = np.linalg.svd(X_cat, full_matrices=False)
            X_cat = u_m * s_v[None, :]
            v_proj = vt_m.T
        clf = LogisticRegression(C=c_val, fit_intercept=False, solver="liblinear", max_iter=2000, tol=1e-7, random_state=SEED)
        clf.fit(X_cat, np.concatenate(ys), sample_weight=np.concatenate(ws))
        return _IsometricLogReg(clf, v_proj), rms, keep

    concs = []
    for held in org_list:
        tr_orgs = [g for g in org_list if g != held]
        c_losses = []
        for c_val in C_GRID:
            in_l = []
            for in_held in tr_orgs:
                in_tr = [g for g in tr_orgs if g != in_held]
                clf, rms, keep = fit_p(in_tr, c_val)
                sc = clf.decision_function(pairs[in_held][:, keep] / rms[keep])
                in_l.append(float(np.mean(np.logaddexp(0.0, -sc))))
            c_losses.append(float(np.mean(in_l)))
        best_c = C_GRID[int(np.argmin(np.round(c_losses, 12)))]
        clf, rms, keep = fit_p(tr_orgs, best_c)
        sc_h = clf.decision_function(pairs[held][:, keep] / rms[keep])
        concs.append(float(np.mean((sc_h > 0).astype(float) + 0.5 * (sc_h == 0))))
    return float(np.mean(concs)), float(np.min(concs))


def eval_newey_single_and_paired(samps: pd.DataFrame, rungs: dict[str, np.ndarray]) -> list[dict]:
    y = (samps["condition"] == "IFNg").astype(int).values
    groups = samps["organoid"].values
    org_list = sorted(np.unique(groups))
    rows = []

    for rname, X in rungs.items():
        y_true_all, y_pred_all, y_score_all = [], [], []
        for held in org_list:
            tr = groups != held
            te = groups == held
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X[tr], y[tr], groups[tr])
            Xte = (X[te][:, keep] - mu[:, keep]) / sd[:, keep]
            y_true_all.extend(y[te])
            y_pred_all.extend(clf.predict(Xte))
            y_score_all.extend(clf.decision_function(Xte))
        ba_single = float(balanced_accuracy_score(y_true_all, y_pred_all))
        auc_single = float(roc_auc_score(y_true_all, y_score_all))
        mean_c, worst_c = _paired_concordance_nested(X, y, groups)

        rows.append(
            {
                "cohort_task": "Newey_CRC_PDO_IFNg_vs_MEKi",
                "representation": rname,
                "single_sample_loo_bal_acc": ba_single,
                "single_sample_loo_auroc": auc_single,
                "paired_shift_loo_concordance": mean_c,
                "paired_shift_worst_org": worst_c,
            }
        )

    X180 = rungs["R2_PosAA180"]
    y_true_all, y_pred_all, y_score_all, concs_rbf = [], [], [], []
    for held in org_list:
        tr = groups != held
        te = groups == held
        mu = X180[tr].mean(axis=0, keepdims=True)
        sd = np.where(X180[tr].std(axis=0, keepdims=True) > 1e-12, X180[tr].std(axis=0, keepdims=True), 1.0)
        Xtr = (X180[tr] - mu) / sd
        Xte = (X180[te] - mu) / sd
        gam = 1.0 / Xtr.shape[1]
        Ktr = rbf_kernel(Xtr, Xtr, gamma=gam)
        Kte = rbf_kernel(Xte, Xtr, gamma=gam)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Ktr, y[tr])
        y_true_all.extend(y[te])
        y_pred_all.extend(clf.predict(Kte))
        y_score_all.extend(clf.decision_function(Kte))

        tr_orgs = [o for o in org_list if o != held]
        d_tr = np.vstack(
            [
                np.array([X180[i] - X180[j] for i, j in itertools.product(np.flatnonzero((groups == o) & (y == 1)), np.flatnonzero((groups == o) & (y == 0)))])
                for o in tr_orgs
            ]
        )
        X_pair_tr = np.vstack([d_tr, -d_tr])
        y_pair_tr = np.concatenate([np.ones(len(d_tr), int), np.zeros(len(d_tr), int)])
        sc_sd = np.where(X_pair_tr.std(axis=0, keepdims=True) > 1e-12, X_pair_tr.std(axis=0, keepdims=True), 1.0)
        X_pair_tr_s = X_pair_tr / sc_sd
        dh = np.array([X180[i] - X180[j] for i, j in itertools.product(np.flatnonzero((groups == held) & (y == 1)), np.flatnonzero((groups == held) & (y == 0)))]) / sc_sd
        Kp_tr = rbf_kernel(X_pair_tr_s, X_pair_tr_s, gamma=1.0 / X180.shape[1])
        Kp_te = rbf_kernel(dh, X_pair_tr_s, gamma=1.0 / X180.shape[1])
        clf_kp = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000, random_state=SEED).fit(Kp_tr, y_pair_tr)
        sc = clf_kp.decision_function(Kp_te)
        concs_rbf.append(float(np.mean((sc > 0).astype(float) + 0.5 * (sc == 0))))

    rows.append(
        {
            "cohort_task": "Newey_CRC_PDO_IFNg_vs_MEKi",
            "representation": "R3_Nonlinear_RBF_PosAA180",
            "single_sample_loo_bal_acc": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "single_sample_loo_auroc": float(roc_auc_score(y_true_all, y_score_all)),
            "paired_shift_loo_concordance": float(np.mean(concs_rbf)),
            "paired_shift_worst_org": float(np.min(concs_rbf)),
        }
    )
    return rows


def eval_velz_ladder(cfg: PathResolver) -> list[dict]:
    vdir = cfg.processed("velz2025")
    samps = pd.read_csv(vdir / "samples.tsv", sep="\t").sort_values("sample_id").reset_index(drop=True)
    rna = pd.read_csv(vdir / "RNA_gene_tpm.tsv.gz", sep="\t").set_index("sample_id").loc[samps["sample_id"]]
    ga = pd.read_csv(vdir / "gene_annotation.tsv", sep="\t")
    sym_to_ensg = ga[ga["symbol_is_unique"] == True].set_index("gene_symbol")["gene_id"].to_dict()
    hm = load_hallmark_sets(cfg)

    def score_set(gset: set[str]) -> np.ndarray:
        cols = [sym_to_ensg[s] for s in gset if s in sym_to_ensg and sym_to_ensg[s] in rna.columns]
        return np.log2(1.0 + rna[cols].values).mean(axis=1)

    y_prolif = 0.5 * (
        score_set(hm.get("E2F_TARGETS", {"MCM2", "MCM5", "PCNA", "TOP2A", "CDK1", "MKI67"}))
        + score_set(hm.get("G2M_CHECKPOINT", {"CCNB1", "CDK1", "TOP2A", "AURKA", "BUB1"}))
    )
    y_ifng = score_set(hm.get("INTERFERON_GAMMA_RESPONSE", {"STAT1", "IRF1", "TAP1", "PSMB8", "B2M", "GBP1"}))
    subgroups = samps["molecular_subgroup"].values

    pep_files = list(vdir.glob("*peptide*.tsv*")) + list(cfg.processed("velz_state").glob("*peptide*.tsv*"))
    p_df = pd.read_csv(pep_files[0], sep="\t")
    seq_col = "peptide" if "peptide" in p_df.columns else "sequence"
    p9 = p_df[p_df[seq_col].astype(str).map(lambda s: len(s) == 9 and set(s) <= set(AMINO_ACIDS))].copy()
    p9["w"] = 1.0
    piv = p9.pivot_table(index="sample_id", columns=seq_col, values="w", aggfunc="sum", fill_value=0.0).loc[samps["sample_id"]]
    seqs = list(piv.columns)
    pi = piv.values / piv.values.sum(axis=1, keepdims=True)
    fmaps = encode_9mer_features(seqs)

    gcol = next((c for c in ("gene", "gene_symbol", "genes", "protein_ids") if c in p9.columns), None)
    gene_mat = np.zeros((len(samps), 1))
    if gcol:
        p2g = p9.groupby(seq_col)[gcol].first().reindex(seqs).fillna("").astype(str).tolist()
        ug = sorted({g.split(";")[0] for g in p2g if g and g != "nan"})
        gi = {g: i for i, g in enumerate(ug)}
        gene_mat = np.zeros((len(samps), len(ug)), dtype=float)
        for j, g in enumerate(p2g):
            g0 = g.split(";")[0]
            if g0 in gi:
                gene_mat[:, gi[g0]] += pi[:, j]

    nuis = np.column_stack([np.log2(piv.values.sum(axis=1)), (piv.values > 0).sum(axis=1)])
    rungs = {
        "R0_Quantity_Depth": nuis,
        "R1_Coarse96": pi @ fmaps["coarse96"],
        "R2a_AA20": pi @ fmaps["aa20"],
        "R2_PosAA180": pi @ fmaps["pos180"],
        "R4_Peptide_Identity": pi,
        "R5_SourceGene_Oracle": gene_mat,
    }

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(samps)), subgroups))
    rows = []
    alphas = np.logspace(-2, 4, 13)

    for rname, X in rungs.items():
        se_mod_p, se_ref_p, se_mod_i, se_ref_i, y_sub_true, y_sub_pred = [], [], [], [], [], []
        for tr, te in folds:
            sub_means_p = {s: y_prolif[tr][subgroups[tr] == s].mean() for s in np.unique(subgroups[tr])}
            sub_means_i = {s: y_ifng[tr][subgroups[tr] == s].mean() for s in np.unique(subgroups[tr])}
            ref_p_tr = np.array([sub_means_p[s] for s in subgroups[tr]])
            ref_p_te = np.array([sub_means_p[s] for s in subgroups[te]])
            ref_i_tr = np.array([sub_means_i[s] for s in subgroups[tr]])
            ref_i_te = np.array([sub_means_i[s] for s in subgroups[te]])

            mu = X[tr].mean(axis=0, keepdims=True)
            sd = np.where(X[tr].std(axis=0, keepdims=True) > 1e-12, X[tr].std(axis=0, keepdims=True), 1.0)
            Xtr = (X[tr] - mu) / sd
            Xte = (X[te] - mu) / sd

            rc_p = RidgeCV(alphas=alphas).fit(Xtr, y_prolif[tr] - ref_p_tr)
            se_mod_p.extend((y_prolif[te] - (ref_p_te + rc_p.predict(Xte))) ** 2)
            se_ref_p.extend((y_prolif[te] - ref_p_te) ** 2)

            rc_i = RidgeCV(alphas=alphas).fit(Xtr, y_ifng[tr] - ref_i_tr)
            se_mod_i.extend((y_ifng[te] - (ref_i_te + rc_i.predict(Xte))) ** 2)
            se_ref_i.extend((y_ifng[te] - ref_i_te) ** 2)

            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xtr, subgroups[tr])
            y_sub_true.extend(subgroups[te])
            y_sub_pred.extend(clf.predict(Xte))

        rows.append(
            {
                "cohort_task": "Velz_N28_CrossPatient_BaselineFree",
                "representation": rname,
                "prolif_rel_mse_vs_subgroup": float(np.mean(se_mod_p) / np.mean(se_ref_p)),
                "ifng_rel_mse_vs_subgroup": float(np.mean(se_mod_i) / np.mean(se_ref_i)),
                "subgroup_k4_bal_acc": float(balanced_accuracy_score(y_sub_true, y_sub_pred)),
            }
        )

    X180 = rungs["R2_PosAA180"]
    se_mod_p, se_ref_p, se_mod_i, se_ref_i, y_sub_true, y_sub_pred = [], [], [], [], [], []
    for tr, te in folds:
        sub_means_p = {s: y_prolif[tr][subgroups[tr] == s].mean() for s in np.unique(subgroups[tr])}
        sub_means_i = {s: y_ifng[tr][subgroups[tr] == s].mean() for s in np.unique(subgroups[tr])}
        ref_p_tr = np.array([sub_means_p[s] for s in subgroups[tr]])
        ref_p_te = np.array([sub_means_p[s] for s in subgroups[te]])
        ref_i_tr = np.array([sub_means_i[s] for s in subgroups[tr]])
        ref_i_te = np.array([sub_means_i[s] for s in subgroups[te]])
        mu = X180[tr].mean(axis=0, keepdims=True)
        sd = np.where(X180[tr].std(axis=0, keepdims=True) > 1e-12, X180[tr].std(axis=0, keepdims=True), 1.0)
        Xtr = (X180[tr] - mu) / sd
        Xte = (X180[te] - mu) / sd
        kr_p = KernelRidge(alpha=10.0, kernel="rbf", gamma=1.0 / Xtr.shape[1]).fit(Xtr, y_prolif[tr] - ref_p_tr)
        se_mod_p.extend((y_prolif[te] - (ref_p_te + kr_p.predict(Xte))) ** 2)
        se_ref_p.extend((y_prolif[te] - ref_p_te) ** 2)
        kr_i = KernelRidge(alpha=10.0, kernel="rbf", gamma=1.0 / Xtr.shape[1]).fit(Xtr, y_ifng[tr] - ref_i_tr)
        se_mod_i.extend((y_ifng[te] - (ref_i_te + kr_i.predict(Xte))) ** 2)
        se_ref_i.extend((y_ifng[te] - ref_i_te) ** 2)
        Ktr = rbf_kernel(Xtr, Xtr, gamma=1.0 / Xtr.shape[1])
        Kte = rbf_kernel(Xte, Xtr, gamma=1.0 / Xtr.shape[1])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Ktr, subgroups[tr])
        y_sub_true.extend(subgroups[te])
        y_sub_pred.extend(clf.predict(Kte))
    rows.append(
        {
            "cohort_task": "Velz_N28_CrossPatient_BaselineFree",
            "representation": "R3_Nonlinear_RBF_PosAA180",
            "prolif_rel_mse_vs_subgroup": float(np.mean(se_mod_p) / np.mean(se_ref_p)),
            "ifng_rel_mse_vs_subgroup": float(np.mean(se_mod_i) / np.mean(se_ref_i)),
            "subgroup_k4_bal_acc": float(balanced_accuracy_score(y_sub_true, y_sub_pred)),
        }
    )
    return rows


def build_paired_shift_catalog(cfg: PathResolver) -> list[dict]:
    entries = []

    def make_entry(study, lineage, contrast_id, drug_class, bio_state, time_h, df_ref, df_trt):
        r9 = df_ref[df_ref["peptide"].map(lambda s: len(str(s)) == 9 and set(str(s)) <= set(AMINO_ACIDS))].copy()
        t9 = df_trt[df_trt["peptide"].map(lambda s: len(str(s)) == 9 and set(str(s)) <= set(AMINO_ACIDS))].copy()
        if "locus" not in r9.columns:
            r9["locus"] = ""
        if "locus" not in t9.columns:
            t9["locus"] = ""
        g_ref = r9.groupby("peptide").agg(w=("intensity", "sum"), gene=("gene", "first"), locus=("locus", "first"))
        g_trt = t9.groupby("peptide").agg(w=("intensity", "sum"), gene=("gene", "first"), locus=("locus", "first"))
        all_peps = sorted(set(g_ref.index) | set(g_trt.index))
        if len(all_peps) < 100:
            return None
        w0 = g_ref["w"].reindex(all_peps).fillna(0.0).values
        w1 = g_trt["w"].reindex(all_peps).fillna(0.0).values
        if w0.sum() <= 0 or w1.sum() <= 0:
            return None
        q0 = w0 / w0.sum()
        q1 = w1 / w1.sum()
        dq = q1 - q0
        genes = [str(g_trt["gene"].get(p, g_ref["gene"].get(p, "")) or "").split(";")[0] for p in all_peps]
        loci = [str(g_trt["locus"].get(p, g_ref["locus"].get(p, "")) or "") for p in all_peps]
        fmaps = encode_9mer_features(all_peps)
        mda_share0 = float(sum(q0[i] for i, g in enumerate(genes) if g in MDA_PANEL))
        mda_share1 = float(sum(q1[i] for i, g in enumerate(genes) if g in MDA_PANEL))

        loc_arr = np.array(loci)
        has_locus = np.isin(loc_arr, ["A", "B", "C"]).sum() >= 50
        d_pos180_withinA = np.zeros(180)
        d_pos180_withinB = np.zeros(180)
        d_locus_share = np.zeros(3)
        if has_locus:
            for li, lch in enumerate(["A", "B", "C"]):
                m_l = loc_arr == lch
                d_locus_share[li] = q1[m_l].sum() - q0[m_l].sum()
            for lch, target_arr in [("A", d_pos180_withinA), ("B", d_pos180_withinB)]:
                m_l = loc_arr == lch
                if w0[m_l].sum() > 0 and w1[m_l].sum() > 0:
                    q0_l = w0[m_l] / w0[m_l].sum()
                    q1_l = w1[m_l] / w1[m_l].sum()
                    target_arr[:] = (q1_l - q0_l) @ fmaps["pos180"][m_l]

        return {
            "study": study,
            "lineage": lineage,
            "contrast_id": contrast_id,
            "drug_class": drug_class,
            "bio_state": bio_state,
            "time_h": time_h,
            "n_9mers_union": len(all_peps),
            "peptides": all_peps,
            "genes": genes,
            "loci": loci,
            "has_locus": bool(has_locus),
            "q0": q0,
            "q1": q1,
            "dq": dq,
            "nuis_delta": np.array(
                [
                    np.log2(w1.sum() / w0.sum()),
                    np.log2(max(1, (w1 > 0).sum()) / max(1, (w0 > 0).sum())),
                    float((dq @ fmaps["krl_count"]).sum()),
                ]
            ),
            "d_coarse96": dq @ fmaps["coarse96"],
            "d_aa20": dq @ fmaps["aa20"],
            "d_pos80": dq @ fmaps["pos80"],
            "d_anchor40": dq @ fmaps["anchor40"],
            "d_pos180": dq @ fmaps["pos180"],
            "d_pos180_withinA": d_pos180_withinA,
            "d_pos180_withinB": d_pos180_withinB,
            "d_locus_share": d_locus_share,
            "d_krl": dq @ fmaps["krl_count"],
            "d_mda_share": mda_share1 - mda_share0,
        }

    b_pep = pd.read_csv(cfg.followup("bernhardt2025_psilac_peptides.tsv.gz"), sep="\t")
    b_pep["locus"] = b_pep["hla_allele"].astype(str).str.extract(r"^HLA-([ABC])")[0].fillna("")
    for (tbl, cline, interv, etag, th), grp in b_pep.groupby(["table", "cell_line", "intervention", "experiment_tag", "time_h"]):
        if tbl == "Table12":
            continue
        valid = grp[(grp["intensity_ref_pulse"] > 0) & grp["l2fc_trt_vs_ref"].notna()].copy()
        if len(valid) < 100:
            continue
        df_r = pd.DataFrame({"peptide": valid["peptide"], "intensity": valid["intensity_ref_pulse"], "gene": valid["gene"], "locus": valid["locus"]})
        df_t = pd.DataFrame(
            {"peptide": valid["peptide"], "intensity": valid["intensity_ref_pulse"] * (2.0 ** valid["l2fc_trt_vs_ref"]), "gene": valid["gene"], "locus": valid["locus"]}
        )
        if etag.startswith("Ctrl"):
            bstate, dclass = "Mock_Control", "DMSO_Mock"
        elif interv == "IFNg":
            bstate, dclass = "IFNg_Inflammatory", "IFNg"
        elif cline == "SKMel28_dr":
            bstate, dclass = "MAPKi_Resistant_NoResponse", "MAPKi"
        else:
            bstate, dclass = "MAPKi_Sensitive_Response", "MAPKi"
        parent_lin = "SKMEL28" if "SKMel28" in cline else cline.upper()
        ent = make_entry("Bernhardt2025", parent_lin, f"B25_{cline}_{interv}_{etag}_{th}h", dclass, bstate, float(th), df_r, df_t)
        if ent is not None:
            ent["cell_subline"] = cline
            ent["is_drug_exposed"] = int(etag.startswith("Treat") and interv in ("Trametinib", "VemuCobi"))
            entries.append(ent)

    s_pep = pd.read_csv(cfg.followup("stopfer2022_peptides.tsv.gz"), sep="\t")
    for (tab, cline, exp_grp), grp in s_pep[s_pep["source_file"] == "sd01"].groupby(["tab", "cell_line", "experiment_group"]):
        df_r = grp[grp["condition"] == "DMSO"].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
        for (cond, dose), tgrp in grp[grp["condition"] != "DMSO"].groupby(["condition", "dose"]):
            df_t = tgrp.groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
            ent = make_entry("Stopfer2022", cline.upper(), f"S22_{tab}_{cline}_{cond}_{dose}", "MAPKi", "MAPKi_Sensitive_Response", 72.0, df_r, df_t)
            if ent is not None:
                ent["cell_subline"] = cline
                ent["is_drug_exposed"] = 1
                entries.append(ent)

    donor_bind_files = {
        "ITO57T": cfg.processed("visibility_binding", "gene_unique", "pdo57t_best.tsv.gz"),
        "ITO58M": cfg.processed("visibility_binding", "gene_unique", "pdo58m_best.tsv.gz"),
        "ITO66T": cfg.processed("visibility_binding", "gene_unique", "pdo66t_best.tsv.gz"),
        "UWB1.289": cfg.processed("visibility_binding", "gene_unique", "chong2018_best.tsv.gz"),
        "A375": cfg.processed("visibility_binding", "a375_best.tsv.gz"),
    }
    donor_pep_to_loc: dict[tuple[str, str], str] = {}
    for lin_key, bf in donor_bind_files.items():
        if bf.exists():
            bdf = pd.read_csv(bf, sep="\t")
            locs = bdf["allele"].astype(str).str.extract(r"HLA-([ABC])")[0].fillna("")
            for p_s, l_s in zip(bdf["peptide"], locs):
                if l_s:
                    donor_pep_to_loc[(lin_key, str(p_s))] = l_s

    pert_configs = [
        ("Bathini2026", "ITO57T", "ITO57T", "IFNG", "IFNg_Inflammatory", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Bathini2026", "ITO58M", "ITO58M", "IFNG", "IFNg_Inflammatory", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Bathini2026", "ITO66T", "ITO66T", "IFNG", "IFNg_Inflammatory", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Chong2018", "UWB.1_289", "UWB1.289", "IFNG", "IFNg_Inflammatory", cfg.processed("visibility_replication", "chong2018_gene_unique")),
        ("Bathini2026", "A375", "A375", "TGFB", "TGFb", cfg.processed("visibility_mhc1tip_gene_unique", "a375")),
    ]
    for study, bg_raw, lin, trt, bstate, pdir in pert_configs:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw) & (samps["modality"] == "peptide")]
        peps = pd.read_csv(pdir / "peptides.tsv.gz", sep="\t")
        c_ids = samps[samps["condition"] == "control"]["sample_id"].tolist()
        t_ids = samps[samps["condition"] == trt]["sample_id"].tolist()
        df_r = peps[peps["sample_id"].isin(c_ids) & (peps["intensity"] > 0)].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
        df_t = peps[peps["sample_id"].isin(t_ids) & (peps["intensity"] > 0)].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
        df_r["locus"] = df_r["peptide"].map(lambda s, _l=lin: donor_pep_to_loc.get((_l, s), ""))
        df_t["locus"] = df_t["peptide"].map(lambda s, _l=lin: donor_pep_to_loc.get((_l, s), ""))
        ent = make_entry(study, lin, f"{study}_{lin}_{trt}", trt, bstate, 48.0, df_r, df_t)
        if ent is not None:
            ent["cell_subline"] = lin
            ent["is_drug_exposed"] = 0
            entries.append(ent)

    n23 = pd.read_csv(cfg.processed("newey2023", "peptides_long.tsv.gz"), sep="\t")
    newey_loc_map: dict[tuple[str, str], str] = {}
    if "predicted_hla" in n23.columns and "organoid" in n23.columns:
        locs_n23 = n23["predicted_hla"].astype(str).str.extract(r"HLA[.*_-]?([ABC])")[0].fillna("")
        for o_s, p_s, l_code in zip(n23["organoid"], n23["peptide"], locs_n23):
            if l_code:
                newey_loc_map[(str(o_s).strip(), str(p_s))] = l_code

    ndir = cfg.processed("newey2019_replicates")
    ns = pd.read_csv(ndir / "samples.tsv", sep="\t")
    np_df = pd.read_csv(ndir / "peptides_long.tsv.gz", sep="\t")
    for org in ["CRC01", "CRC04", "CRC05", "CRC08"]:
        tram_ids = ns[(ns["organoid"] == org) & (ns["condition"] == "TRAM")]["sample_id"].tolist()
        ifng_ids = ns[(ns["organoid"] == org) & (ns["condition"] == "IFNg")]["sample_id"].tolist()
        ut_ids = ns[(ns["organoid"] == org) & (ns["condition"].isin(["UT", "Control", "control"]))]["sample_id"].tolist()
        df_tr = np_df[np_df["sample_id"].isin(tram_ids) & (np_df["intensity"] > 0)].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
        df_if = np_df[np_df["sample_id"].isin(ifng_ids) & (np_df["intensity"] > 0)].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
        df_tr["locus"] = df_tr["peptide"].map(lambda s, _o=org: newey_loc_map.get((_o, s), ""))
        df_if["locus"] = df_if["peptide"].map(lambda s, _o=org: newey_loc_map.get((_o, s), ""))
        if len(ut_ids) > 0:
            df_ut = np_df[np_df["sample_id"].isin(ut_ids) & (np_df["intensity"] > 0)].groupby("peptide", as_index=False).agg(intensity=("intensity", "mean"), gene=("gene", "first"))
            df_ut["locus"] = df_ut["peptide"].map(lambda s, _o=org: newey_loc_map.get((_o, s), ""))
            e1 = make_entry("Newey2019", org, f"Newey_{org}_IFNg_vs_UT", "IFNg", "IFNg_Inflammatory", 48.0, df_ut, df_if)
            e2 = make_entry("Newey2019", org, f"Newey_{org}_TRAM_vs_UT", "MAPKi", "MAPKi_Sensitive_Response", 48.0, df_ut, df_tr)
            for e in (e1, e2):
                if e is not None:
                    e["cell_subline"] = org
                    e["is_drug_exposed"] = int(e["drug_class"] == "MAPKi")
                    entries.append(e)

    return entries


def run_exp7c(cfg: PathResolver, entries: list[dict], out_dir: Path) -> dict:
    b_entries = [
        e
        for e in entries
        if e["study"] == "Bernhardt2025" and e["bio_state"] in ("MAPKi_Sensitive_Response", "MAPKi_Resistant_NoResponse", "Mock_Control")
    ]
    train_b = [e for e in b_entries if e["lineage"] != "SKMEL28"]
    test_sk = [e for e in b_entries if e["lineage"] == "SKMEL28"]
    stopfer_non_sk28 = [e for e in entries if e["study"] == "Stopfer2022" and e["lineage"] != "SKMEL28"]

    all_peps_b = sorted({p for e in b_entries for p in e["peptides"]})
    p_idx_b = {p: i for i, p in enumerate(all_peps_b)}
    all_genes_b = sorted({g for e in b_entries for g in e["genes"] if g})
    g_idx_b = {g: i for i, g in enumerate(all_genes_b)}

    def feat_matrix(elist: list[dict], kind: str) -> np.ndarray:
        if kind == "Drug_Exposure_Shortcut":
            return np.array([[float(e["is_drug_exposed"]), e["time_h"] / 72.0] for e in elist])
        if kind == "R0_Quantity_SILAC_Nuisance":
            return np.vstack([np.concatenate([e["nuis_delta"], e["d_krl"]]) for e in elist])
        if kind == "R1_Coarse96":
            return np.vstack([e["d_coarse96"] for e in elist])
        if kind == "R2a_AA20":
            return np.vstack([e["d_aa20"] for e in elist])
        if kind == "R2b_Pos80_Internal":
            return np.vstack([e["d_pos80"] for e in elist])
        if kind == "R2c_Anchor40":
            return np.vstack([e["d_anchor40"] for e in elist])
        if kind == "R2_PosAA180":
            return np.vstack([e["d_pos180"] for e in elist])
        if kind == "R4_Peptide_Identity":
            M = np.zeros((len(elist), len(all_peps_b)), dtype=float)
            for r, e in enumerate(elist):
                for p, v in zip(e["peptides"], e["dq"]):
                    if p in p_idx_b:
                        M[r, p_idx_b[p]] = v
            return M
        if kind == "R5_SourceGene_Oracle":
            M = np.zeros((len(elist), len(all_genes_b)), dtype=float)
            for r, e in enumerate(elist):
                for g, v in zip(e["genes"], e["dq"]):
                    if g in g_idx_b:
                        M[r, g_idx_b[g]] += v
            return M
        if kind == "R5b_Prespecified_MDA_Axis":
            return np.array([[e["d_mda_share"]] for e in elist])
        raise ValueError(kind)

    y_tr = np.array([1 if e["bio_state"] == "MAPKi_Sensitive_Response" else 0 for e in train_b])
    g_tr = np.array([e["lineage"] for e in train_b])
    y_te = np.array([1 if e["bio_state"] == "MAPKi_Sensitive_Response" else 0 for e in test_sk])
    drug_mask = np.array([e["is_drug_exposed"] == 1 for e in test_sk])

    challenge_rows = []
    sk_pred_table = pd.DataFrame(
        {
            "contrast_id": [e["contrast_id"] for e in test_sk],
            "cell_subline": [e["cell_subline"] for e in test_sk],
            "bio_state": [e["bio_state"] for e in test_sk],
            "time_h": [e["time_h"] for e in test_sk],
            "is_drug_exposed": [e["is_drug_exposed"] for e in test_sk],
            "true_mapk_inhibition": y_te,
            "d_mda_share_pct": [100.0 * e["d_mda_share"] for e in test_sk],
        }
    )

    rep_kinds = [
        "Drug_Exposure_Shortcut",
        "R0_Quantity_SILAC_Nuisance",
        "R1_Coarse96",
        "R2a_AA20",
        "R2b_Pos80_Internal",
        "R2c_Anchor40",
        "R2_PosAA180",
        "R4_Peptide_Identity",
        "R5_SourceGene_Oracle",
        "R5b_Prespecified_MDA_Axis",
    ]
    for kind in rep_kinds:
        Xtr = feat_matrix(train_b, kind)
        Xte = feat_matrix(test_sk, kind)
        if kind == "R5b_Prespecified_MDA_Axis":
            scores_te = Xte[:, 0]
            sel_c = np.nan
        else:
            clf, mu, sd, keep, sel_c = fit_nested_grouped_logreg(Xtr, y_tr, g_tr)
            scores_te = clf.decision_function((Xte[:, keep] - mu[:, keep]) / sd[:, keep])

        sk_pred_table[f"score_{kind}"] = scores_te
        auc_all_sk = float(roc_auc_score(y_te, scores_te))
        wt_scores = scores_te[drug_mask & (y_te == 1)]
        dr_scores = scores_te[drug_mask & (y_te == 0)]
        wt_vs_dr_conc = float(np.mean([1.0 if w > d else (0.5 if w == d else 0.0) for w in wt_scores for d in dr_scores]))

        all_subsets = list(itertools.combinations(range(len(test_sk)), 2))
        obs_rank_sum = sum(stats.rankdata(scores_te)[y_te == 1])
        perm_ge = sum(1 for s_idx in all_subsets if sum(stats.rankdata(scores_te)[list(s_idx)]) >= obs_rank_sum - 1e-12)
        exact_test_p = float(perm_ge / len(all_subsets))

        challenge_rows.append(
            {
                "training_design": "Bernhardt_MaMel63a+UKEMel105b_to_SKMel28_ZeroHLAOverlap",
                "representation": kind,
                "selected_C": sel_c,
                "n_train": len(train_b),
                "n_test_skmel28": len(test_sk),
                "auroc_all_skmel28_8runs": auc_all_sk,
                "exact_combinatorial_p_2of8": exact_test_p,
                "concordance_treated_wt_vs_dr": wt_vs_dr_conc,
                "mean_score_wt_drug": float(np.mean(wt_scores)),
                "mean_score_dr_drug": float(np.mean(dr_scores)),
                "score_gap_wt_minus_dr": float(np.mean(wt_scores) - np.mean(dr_scores)),
            }
        )

    mock_tr_ids = {e["contrast_id"] for e in train_b if e["bio_state"] == "Mock_Control"}
    unseen_idx = [i for i, e in enumerate(b_entries) if e["contrast_id"] not in mock_tr_ids]
    for kind in ["R1_Coarse96", "R2_PosAA180", "R5b_Prespecified_MDA_Axis"]:
        tr_cs = stopfer_non_sk28 + [e for e in train_b if e["bio_state"] == "Mock_Control"]
        y_tr_cs = np.array([1 if e["bio_state"] == "MAPKi_Sensitive_Response" else 0 for e in tr_cs])
        g_tr_cs = np.array([e["lineage"] for e in tr_cs])
        Xtr = feat_matrix(tr_cs, kind)
        Xte_all = feat_matrix(b_entries, kind)
        y_te_all = np.array([1 if e["bio_state"] == "MAPKi_Sensitive_Response" else 0 for e in b_entries])
        if kind == "R5b_Prespecified_MDA_Axis":
            sc_all = Xte_all[:, 0]
            sel_c = np.nan
        else:
            clf, mu, sd, keep, sel_c = fit_nested_grouped_logreg(Xtr, y_tr_cs, g_tr_cs)
            sc_all = clf.decision_function((Xte_all[:, keep] - mu[:, keep]) / sd[:, keep])
        sk_idx = [i for i, e in enumerate(b_entries) if e["lineage"] == "SKMEL28"]
        sc_sk = sc_all[sk_idx]
        wt_sc = sc_sk[drug_mask & (y_te == 1)]
        dr_sc = sc_sk[drug_mask & (y_te == 0)]
        wt_dr_c = float(np.mean([1.0 if w > d else (0.5 if w == d else 0.0) for w in wt_sc for d in dr_sc]))
        challenge_rows.append(
            {
                "training_design": "CrossPlatform_Stopfer2022_TMT_to_Bernhardt2025_pSILAC",
                "representation": kind,
                "selected_C": sel_c,
                "n_train": len(tr_cs),
                "n_test_skmel28": len(test_sk),
                "auroc_all_skmel28_8runs": float(roc_auc_score(y_te, sc_sk)),
                "auroc_unseen_bernhardt_14runs": float(roc_auc_score(y_te_all[unseen_idx], sc_all[unseen_idx])),
                "exact_combinatorial_p_2of8": np.nan,
                "concordance_treated_wt_vs_dr": wt_dr_c,
                "mean_score_wt_drug": float(np.mean(wt_sc)),
                "mean_score_dr_drug": float(np.mean(dr_sc)),
                "score_gap_wt_minus_dr": float(np.mean(wt_sc) - np.mean(dr_sc)),
            }
        )

    pd.DataFrame(challenge_rows).to_csv(out_dir / "exp7c_resistance_and_cross_study_challenge.tsv", sep="\t", index=False)
    sk_pred_table.to_csv(out_dir / "exp7c_skmel28_wt_vs_dr_predictions.tsv", sep="\t", index=False)

    pert_active = [e for e in entries if e["bio_state"] in ("IFNg_Inflammatory", "MAPKi_Sensitive_Response")]
    y_inf = np.array([1 if e["bio_state"] == "IFNg_Inflammatory" else 0 for e in pert_active])
    lineages = np.array([e["lineage"] for e in pert_active])
    studies = np.array([e["study"] for e in pert_active])

    all_peps_p = sorted({p for e in pert_active for p in e["peptides"]})
    p_idx_p = {p: i for i, p in enumerate(all_peps_p)}
    all_genes_p = sorted({g for e in pert_active for g in e["genes"] if g})
    g_idx_p = {g: i for i, g in enumerate(all_genes_p)}

    cross_rows = []
    for kind in [
        "R0_Quantity_Nuisance",
        "R1_Coarse96",
        "R2a_AA20",
        "R2b_Pos80_Internal",
        "R2c_Anchor40",
        "R2_PosAA180",
        "R4_Peptide_Identity",
        "R5_SourceGene_Oracle",
    ]:
        if kind == "R0_Quantity_Nuisance":
            X = np.vstack([e["nuis_delta"][:2] for e in pert_active])
        elif kind == "R1_Coarse96":
            X = np.vstack([e["d_coarse96"] for e in pert_active])
        elif kind == "R2a_AA20":
            X = np.vstack([e["d_aa20"] for e in pert_active])
        elif kind == "R2b_Pos80_Internal":
            X = np.vstack([e["d_pos80"] for e in pert_active])
        elif kind == "R2c_Anchor40":
            X = np.vstack([e["d_anchor40"] for e in pert_active])
        elif kind == "R2_PosAA180":
            X = np.vstack([e["d_pos180"] for e in pert_active])
        elif kind == "R4_Peptide_Identity":
            X = np.zeros((len(pert_active), len(all_peps_p)), dtype=float)
            for r, e in enumerate(pert_active):
                for p, v in zip(e["peptides"], e["dq"]):
                    X[r, p_idx_p[p]] = v
        elif kind == "R5_SourceGene_Oracle":
            X = np.zeros((len(pert_active), len(all_genes_p)), dtype=float)
            for r, e in enumerate(pert_active):
                for g, v in zip(e["genes"], e["dq"]):
                    if g in g_idx_p:
                        X[r, g_idx_p[g]] += v

        sc_lolo, yt_lolo, yp_lolo = [], [], []
        for lin in sorted(np.unique(lineages)):
            tr = lineages != lin
            te = lineages == lin
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X[tr], y_inf[tr], lineages[tr])
            Xte_s = (X[te][:, keep] - mu[:, keep]) / sd[:, keep]
            sc_lolo.extend(clf.decision_function(Xte_s))
            yp_lolo.extend(clf.predict(Xte_s))
            yt_lolo.extend(y_inf[te])

        sc_loso, yt_loso, yp_loso = [], [], []
        for st in sorted(np.unique(studies)):
            test_lins = set(lineages[studies == st])
            tr = (studies != st) & (~np.isin(lineages, list(test_lins)))
            te = studies == st
            if len(np.unique(y_inf[tr])) < 2:
                continue
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X[tr], y_inf[tr], lineages[tr])
            Xte_s = (X[te][:, keep] - mu[:, keep]) / sd[:, keep]
            sc_loso.extend(clf.decision_function(Xte_s))
            yp_loso.extend(clf.predict(Xte_s))
            yt_loso.extend(y_inf[te])

        cross_rows.append(
            {
                "task": "CrossLineage_PairedShift_IFNg_vs_MAPKi",
                "representation": kind,
                "n_contrasts": len(pert_active),
                "n_unique_lineages": int(len(np.unique(lineages))),
                "lolo_auroc": float(roc_auc_score(yt_lolo, sc_lolo)),
                "lolo_bal_acc": float(balanced_accuracy_score(yt_lolo, yp_lolo)),
                "loso_strict_disjoint_lineage_auroc": float(roc_auc_score(yt_loso, sc_loso)),
                "loso_strict_disjoint_lineage_bal_acc": float(balanced_accuracy_score(yt_loso, yp_loso)),
            }
        )

    # Isobaric I=L harmonization check across all 35 contrasts
    il_rows = []
    for kind, fkey, dim in [("R2c_Anchor40_IL", "anchor40", 40), ("R2_PosAA180_IL", "pos180", 180)]:
        X_il = np.zeros((len(pert_active), dim), dtype=float)
        for r, e in enumerate(pert_active):
            peps_il = [p.replace("I", "L") for p in e["peptides"]]
            fm_il = encode_9mer_features(peps_il)
            X_il[r] = e["dq"] @ fm_il[fkey]
        sc_lolo, yt_lolo = [], []
        for lin in sorted(np.unique(lineages)):
            tr, te = lineages != lin, lineages == lin
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X_il[tr], y_inf[tr], lineages[tr])
            sc_lolo.extend(clf.decision_function((X_il[te][:, keep] - mu[:, keep]) / sd[:, keep]))
            yt_lolo.extend(y_inf[te])
        sc_loso, yt_loso = [], []
        for st in sorted(np.unique(studies)):
            tlins = set(lineages[studies == st])
            tr = (studies != st) & (~np.isin(lineages, list(tlins)))
            te = studies == st
            if len(np.unique(y_inf[tr])) < 2:
                continue
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X_il[tr], y_inf[tr], lineages[tr])
            sc_loso.extend(clf.decision_function((X_il[te][:, keep] - mu[:, keep]) / sd[:, keep]))
            yt_loso.extend(y_inf[te])
        il_rows.append(
            {
                "representation": kind,
                "n_contrasts": len(pert_active),
                "lolo_auroc": float(roc_auc_score(yt_lolo, sc_lolo)),
                "loso_strict_disjoint_lineage_auroc": float(roc_auc_score(yt_loso, sc_loso)),
            }
        )
    pd.DataFrame(il_rows).to_csv(out_dir / "exp7c_isobaric_il_sensitivity.tsv", sep="\t", index=False)

    loc_active = [e for e in pert_active if e["has_locus"]]
    y_loc = np.array([1 if e["bio_state"] == "IFNg_Inflammatory" else 0 for e in loc_active])
    lin_loc = np.array([e["lineage"] for e in loc_active])
    locus_decomp_rows = []
    for dname, X_d in [
        ("Between_HLA_Locus_Share_Only_3df", np.vstack([e["d_locus_share"] for e in loc_active])),
        ("Within_HLA_A_PosAA180_Only", np.vstack([e["d_pos180_withinA"] for e in loc_active])),
        ("Within_HLA_B_PosAA180_Only", np.vstack([e["d_pos180_withinB"] for e in loc_active])),
        ("Whole_Repertoire_PosAA180", np.vstack([e["d_pos180"] for e in loc_active])),
    ]:
        sc_d, yt_d = [], []
        for lin in sorted(np.unique(lin_loc)):
            tr = lin_loc != lin
            te = lin_loc == lin
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X_d[tr], y_loc[tr], lin_loc[tr])
            sc_d.extend(clf.decision_function((X_d[te][:, keep] - mu[:, keep]) / sd[:, keep]))
            yt_d.extend(y_loc[te])
        locus_decomp_rows.append(
            {
                "decomposition_component": dname,
                "n_allele_annotated_contrasts": len(loc_active),
                "n_unique_lineages": int(len(np.unique(lin_loc))),
                "lolo_auroc": float(roc_auc_score(yt_d, sc_d)),
            }
        )
    pd.DataFrame(locus_decomp_rows).to_csv(out_dir / "exp7c_within_vs_between_locus_decomposition.tsv", sep="\t", index=False)
    pd.DataFrame(cross_rows).to_csv(out_dir / "exp7c_cross_lineage_study_transfer.tsv", sep="\t", index=False)

    hm = load_hallmark_sets(cfg)
    ifng_set = hm.get("INTERFERON_GAMMA_RESPONSE", {"STAT1", "IRF1", "TAP1", "TAP2", "PSMB8", "PSMB9", "B2M", "GBP1"})
    e2f_set = hm.get("E2F_TARGETS", {"MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA", "TOP2A", "CDK1", "MKI67"})

    prog_records = []
    nrna = pd.read_csv(cfg.processed("newey2023", "independent_rna_allgenes.tsv.gz"), sep="\t")
    sym_col = next(c for c in nrna.columns if "symbol" in c.lower() or c == "gene")
    resp_col = next(c for c in nrna.columns if "response" in c.lower() or "log2" in c.lower())
    for org in ["CRC01", "CRC04", "CRC05", "CRC08"]:
        sub_r = nrna[nrna["organoid"] == org]
        ent_match = [e for e in entries if e["contrast_id"] == f"Newey_{org}_IFNg_vs_UT"]
        if ent_match and len(sub_r) > 0:
            prog_records.append(
                {
                    "contrast_id": ent_match[0]["contrast_id"],
                    "lineage": org,
                    "study": "Newey2019_RNA",
                    "dz_ifng": float(sub_r[sub_r[sym_col].isin(ifng_set)][resp_col].mean()),
                    "dz_e2f": float(sub_r[sub_r[sym_col].isin(e2f_set)][resp_col].mean()),
                    "d_coarse96": ent_match[0]["d_coarse96"],
                    "d_pos180": ent_match[0]["d_pos180"],
                    "d_anchor40": ent_match[0]["d_anchor40"],
                    "nuis": ent_match[0]["nuis_delta"][:2],
                }
            )

    for study, bg_raw, lin, trt, pdir in [
        ("Bathini2026", "ITO57T", "ITO57T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Bathini2026", "ITO58M", "ITO58M", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Bathini2026", "ITO66T", "ITO66T", "IFNG", cfg.processed("visibility_mhc1tip_gene_unique", "pdo")),
        ("Chong2018", "UWB.1_289", "UWB1.289", "IFNG", cfg.processed("visibility_replication", "chong2018_gene_unique")),
        ("Bathini2026", "A375", "A375", "TGFB", cfg.processed("visibility_mhc1tip_gene_unique", "a375")),
    ]:
        samps = pd.read_csv(pdir / "samples.tsv", sep="\t")
        samps = samps[(samps["included"] == True) & (samps["background"] == bg_raw) & (samps["modality"] == "protein")]
        prots = pd.read_csv(pdir / "proteins.tsv.gz", sep="\t")
        c_p = prots[prots["sample_id"].isin(samps[samps["condition"] == "control"]["sample_id"])].groupby("gene")["log2_abundance"].mean()
        t_p = prots[prots["sample_id"].isin(samps[samps["condition"] == trt]["sample_id"])].groupby("gene")["log2_abundance"].mean()
        dp = (t_p - c_p).dropna()
        ent_match = [e for e in entries if e["contrast_id"] == f"{study}_{lin}_{trt}"]
        if ent_match:
            prog_records.append(
                {
                    "contrast_id": ent_match[0]["contrast_id"],
                    "lineage": lin,
                    "study": study,
                    "dz_ifng": float(dp[dp.index.isin(ifng_set)].mean()),
                    "dz_e2f": float(dp[dp.index.isin(e2f_set)].mean()),
                    "d_coarse96": ent_match[0]["d_coarse96"],
                    "d_pos180": ent_match[0]["d_pos180"],
                    "d_anchor40": ent_match[0]["d_anchor40"],
                    "nuis": ent_match[0]["nuis_delta"][:2],
                }
            )

    sk5_om = pd.read_csv(cfg.followup("stopfer2022_skmel5_multiomics.tsv.gz"), sep="\t")
    for tab, col_r, dose_lbl in [("sd01_8", "rna_l2fc_100nM", "100nM"), ("sd01_10", "rna_l2fc_1uM", "1uM")]:
        ent_match = [e for e in entries if e["contrast_id"] == f"S22_{tab}_SKMEL5_Binimetinib_{dose_lbl}"]
        if ent_match:
            sr = sk5_om.dropna(subset=[col_r])
            prog_records.append(
                {
                    "contrast_id": ent_match[0]["contrast_id"],
                    "lineage": "SKMEL5",
                    "study": "Stopfer2022",
                    "dz_ifng": float(sr[sr["gene"].isin(ifng_set)][col_r].mean()),
                    "dz_e2f": float(sr[sr["gene"].isin(e2f_set)][col_r].mean()),
                    "d_coarse96": ent_match[0]["d_coarse96"],
                    "d_pos180": ent_match[0]["d_pos180"],
                    "d_anchor40": ent_match[0]["d_anchor40"],
                    "nuis": ent_match[0]["nuis_delta"][:2],
                }
            )
    b_prot = pd.read_csv(cfg.followup("bernhardt2025_mamel63a_ifng_proteome.tsv.gz"), sep="\t")
    for etag, th, pcol in [("Treat_T1", 24, "prot_l2fc_ifng_24h"), ("Treat_T2", 72, "prot_l2fc_ifng_72h")]:
        ent_match = [e for e in entries if e["contrast_id"] == f"B25_MaMel63a_IFNg_{etag}_{th}h"]
        if ent_match:
            bp = b_prot.dropna(subset=[pcol])
            prog_records.append(
                {
                    "contrast_id": ent_match[0]["contrast_id"],
                    "lineage": "MAMEL63A",
                    "study": "Bernhardt2025",
                    "dz_ifng": float(bp[bp["gene"].isin(ifng_set)][pcol].mean()),
                    "dz_e2f": float(bp[bp["gene"].isin(e2f_set)][pcol].mean()),
                    "d_coarse96": ent_match[0]["d_coarse96"],
                    "d_pos180": ent_match[0]["d_pos180"],
                    "d_anchor40": ent_match[0]["d_anchor40"],
                    "nuis": ent_match[0]["nuis_delta"][:2],
                }
            )

    cont_rows = []
    for target_col, target_name in [("dz_ifng", "Hallmark_IFNg_Response"), ("dz_e2f", "Hallmark_E2F_Targets")]:
        valid_p = [r for r in prog_records if np.isfinite(r[target_col])]
        y_val = np.array([r[target_col] for r in valid_p])
        lins_p = np.array([r["lineage"] for r in valid_p])
        for rep_key, rlabel in [("nuis", "R0_Quantity_Depth"), ("d_coarse96", "R1_Coarse96"), ("d_anchor40", "R2c_Anchor40"), ("d_pos180", "R2_PosAA180")]:
            Xp = np.vstack([r[rep_key] for r in valid_p])
            preds, refs = np.zeros(len(valid_p)), np.zeros(len(valid_p))
            for lin in np.unique(lins_p):
                tr = lins_p != lin
                te = lins_p == lin
                refs[te] = y_val[tr].mean()
                mu = Xp[tr].mean(axis=0, keepdims=True)
                sd = np.where(Xp[tr].std(axis=0, keepdims=True) > 1e-12, Xp[tr].std(axis=0, keepdims=True), 1.0)
                rc = RidgeCV(alphas=np.logspace(-1, 4, 15)).fit((Xp[tr] - mu) / sd, y_val[tr] - y_val[tr].mean())
                preds[te] = y_val[tr].mean() + rc.predict((Xp[te] - mu) / sd)
            r_corr, p_corr = stats.pearsonr(preds, y_val)
            rel_mse = float(np.mean((y_val - preds) ** 2) / np.mean((y_val - refs) ** 2))
            cont_rows.append(
                {
                    "program_axis": target_name,
                    "representation": rlabel,
                    "n_contrasts": len(valid_p),
                    "n_unique_lineages": int(len(np.unique(lins_p))),
                    "lolo_pearson_r": float(r_corr),
                    "lolo_pearson_p": float(p_corr),
                    "lolo_relative_mse_vs_mean": rel_mse,
                }
            )
    pd.DataFrame(cont_rows).to_csv(out_dir / "exp7c_continuous_program_shift_recovery.tsv", sep="\t", index=False)

    return {
        "resistance_challenge": challenge_rows,
        "cross_lineage_ifng_vs_mapki": cross_rows,
        "isobaric_il_harmonization": il_rows,
        "within_vs_between_locus": locus_decomp_rows,
        "continuous_program_recovery": cont_rows,
    }


def run_exp7d(entries: list[dict], samps_newey: pd.DataFrame, p9_newey: pd.DataFrame, out_dir: Path) -> dict:
    """Distributed-mixture vs few-marker redundancy tests."""
    rng = np.random.default_rng(SEED)

    p9_clean = p9_newey[p9_newey["gene"].notna() & (p9_newey["gene"] != "") & (~p9_newey["gene"].astype(str).str.contains(";"))].copy()
    newey_genes = sorted(p9_clean["gene"].unique())
    perm_ng = list(rng.permutation(newey_genes))
    ng_bucket = {g: (i % 5) for i, g in enumerate(perm_ng)}
    p9_clean["gbucket"] = p9_clean["gene"].map(ng_bucket)

    y_new = (samps_newey["condition"] == "IFNg").astype(int).values
    g_new = samps_newey["organoid"].values
    newey_split_c96 = []
    newey_split_aa20 = []
    for k in range(5):
        sub_k = p9_clean[p9_clean["gbucket"] == k]
        piv_k = sub_k.pivot_table(index="sample_id", columns="peptide", values="intensity", aggfunc="sum", fill_value=0.0).reindex(
            samps_newey["sample_id"], fill_value=0.0
        )
        pi_k = piv_k.values / piv_k.values.sum(axis=1, keepdims=True)
        fm_k = encode_9mer_features(list(piv_k.columns))
        c_c96, _ = _paired_concordance_nested(pi_k @ fm_k["coarse96"], y_new, g_new)
        c_aa20, _ = _paired_concordance_nested(pi_k @ fm_k["aa20"], y_new, g_new)
        newey_split_c96.append(c_c96)
        newey_split_aa20.append(c_aa20)

    pert_active = [e for e in entries if e["bio_state"] in ("IFNg_Inflammatory", "MAPKi_Sensitive_Response")]
    y_inf = np.array([1 if e["bio_state"] == "IFNg_Inflammatory" else 0 for e in pert_active])
    lineages = np.array([e["lineage"] for e in pert_active])
    e_fmaps = [encode_9mer_features(e["peptides"]) for e in pert_active]
    all_genes = sorted({g for e in pert_active for g in e["genes"] if g})

    perm_genes = list(rng.permutation(all_genes))
    k_folds = 5
    gene_buckets = {g: (i % k_folds) for i, g in enumerate(perm_genes)}

    sub_pos180 = np.zeros((k_folds, len(pert_active), 180), dtype=float)
    sub_anc40 = np.zeros((k_folds, len(pert_active), 40), dtype=float)
    for k in range(k_folds):
        for r, (e, fm) in enumerate(zip(pert_active, e_fmaps)):
            mask = np.array([gene_buckets.get(g, -1) == k for g in e["genes"]])
            w0, w1 = e["q0"][mask], e["q1"][mask]
            q0_k = w0 / w0.sum() if w0.sum() > 0 else np.zeros_like(w0)
            q1_k = w1 / w1.sum() if w1.sum() > 0 else np.zeros_like(w1)
            dq_k = q1_k - q0_k
            sub_pos180[k, r] = dq_k @ fm["pos180"][mask]
            sub_anc40[k, r] = dq_k @ fm["anchor40"][mask]

    within_180, cross_180, scores_180 = [], [], np.zeros((k_folds, len(pert_active)))
    within_a40, cross_a40, scores_a40 = [], [], np.zeros((k_folds, len(pert_active)))
    for k_tr in range(k_folds):
        for k_te in range(k_folds):
            sc_p, sc_a = np.zeros(len(pert_active)), np.zeros(len(pert_active))
            for lin in np.unique(lineages):
                tr = lineages != lin
                te = lineages == lin
                for arr, out_sc in [(sub_pos180, sc_p), (sub_anc40, sc_a)]:
                    clf, mu, sd, keep, _ = fit_nested_grouped_logreg(arr[k_tr, tr], y_inf[tr], lineages[tr])
                    out_sc[te] = clf.decision_function((arr[k_te, te][:, keep] - mu[:, keep]) / sd[:, keep])
            auc_p = float(roc_auc_score(y_inf, sc_p))
            auc_a = float(roc_auc_score(y_inf, sc_a))
            if k_tr == k_te:
                within_180.append(auc_p)
                within_a40.append(auc_a)
                scores_180[k_tr] = sc_p
                scores_a40[k_tr] = sc_a
            else:
                cross_180.append(auc_p)
                cross_a40.append(auc_a)

    pair_r_180 = [float(stats.pearsonr(scores_180[a], scores_180[b])[0]) for a, b in itertools.combinations(range(k_folds), 2)]
    pair_r_a40 = [float(stats.pearsonr(scores_a40[a], scores_a40[b])[0]) for a, b in itertools.combinations(range(k_folds), 2)]

    ablation_rows = []
    all_peps_p = sorted({p for e in pert_active for p in e["peptides"]})
    p_idx_p = {p: i for i, p in enumerate(all_peps_p)}
    P_mat = np.zeros((len(pert_active), len(all_peps_p)), dtype=float)
    for r, e in enumerate(pert_active):
        for p, v in zip(e["peptides"], e["dq"]):
            P_mat[r, p_idx_p[p]] = v

    for top_k in [1, 3, 5, 10, 25]:
        sc_mk = np.zeros(len(pert_active))
        zero_overlap_folds = 0
        for lin in np.unique(lineages):
            tr = lineages != lin
            te = lineages == lin
            diff_tr = np.abs(P_mat[tr][y_inf[tr] == 1].mean(axis=0) - P_mat[tr][y_inf[tr] == 0].mean(axis=0))
            top_cols = np.argsort(diff_tr)[::-1][:top_k]
            Xtr = P_mat[tr][:, top_cols]
            Xte = P_mat[te][:, top_cols]
            if np.all(Xte == 0):
                zero_overlap_folds += int(te.sum())
            clf, mu, sd, keep, _ = fit_nested_grouped_logreg(Xtr, y_inf[tr], lineages[tr])
            sc_mk[te] = clf.decision_function((Xte[:, keep] - mu[:, keep]) / sd[:, keep])
        auc_m = float(roc_auc_score(y_inf, sc_mk))
        ablation_rows.append(
            {
                "mode": f"Top_{top_k}_Peptide_Markers_Only",
                "frac_genes_removed": 0.0,
                "lolo_auroc_pos180": auc_m,
                "lolo_auroc_anchor40": auc_m,
                "zero_feature_test_contrasts": zero_overlap_folds,
            }
        )

    g_idx_p = {g: i for i, g in enumerate(all_genes)}
    G_mat = np.zeros((len(pert_active), len(all_genes)), dtype=float)
    for r, e in enumerate(pert_active):
        for g, v in zip(e["genes"], e["dq"]):
            if g in g_idx_p:
                G_mat[r, g_idx_p[g]] += v

    for rem_frac in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
        sc_180, sc_a40 = np.zeros(len(pert_active)), np.zeros(len(pert_active))
        for lin in np.unique(lineages):
            tr = lineages != lin
            te = lineages == lin
            g_score = np.abs(G_mat[tr][y_inf[tr] == 1].mean(axis=0) - G_mat[tr][y_inf[tr] == 0].mean(axis=0))
            n_rem = int(round(rem_frac * len(all_genes)))
            banned_genes = set(np.array(all_genes)[np.argsort(g_score)[::-1][:n_rem]]) if n_rem > 0 else set()

            X_p, X_a = np.zeros((len(pert_active), 180)), np.zeros((len(pert_active), 40))
            for r, (e, fm) in enumerate(zip(pert_active, e_fmaps)):
                keep_m = np.array([(g not in banned_genes) for g in e["genes"]])
                w0, w1 = e["q0"][keep_m], e["q1"][keep_m]
                q0_k = w0 / w0.sum() if w0.sum() > 0 else np.zeros_like(w0)
                q1_k = w1 / w1.sum() if w1.sum() > 0 else np.zeros_like(w1)
                dq_k = q1_k - q0_k
                X_p[r] = dq_k @ fm["pos180"][keep_m]
                X_a[r] = dq_k @ fm["anchor40"][keep_m]

            for X_curr, sc_out in [(X_p, sc_180), (X_a, sc_a40)]:
                clf, mu, sd, keep, _ = fit_nested_grouped_logreg(X_curr[tr], y_inf[tr], lineages[tr])
                sc_out[te] = clf.decision_function((X_curr[te][:, keep] - mu[:, keep]) / sd[:, keep])

        ablation_rows.append(
            {
                "mode": "Collective_Distribution_After_TopGene_Ablation",
                "frac_genes_removed": rem_frac,
                "lolo_auroc_pos180": float(roc_auc_score(y_inf, sc_180)),
                "lolo_auroc_anchor40": float(roc_auc_score(y_inf, sc_a40)),
                "zero_feature_test_contrasts": 0,
            }
        )

    pd.DataFrame(ablation_rows).to_csv(out_dir / "exp7d_marker_ablation_vs_panels.tsv", sep="\t", index=False)

    budget_rows = []
    for M in [25, 50, 100, 250, 500, 1000, 2000]:
        aucs_a40, aucs_180 = [], []
        for _ in range(20):
            X_a = np.zeros((len(pert_active), 40))
            X_p = np.zeros((len(pert_active), 180))
            for r, (e, fm) in enumerate(zip(pert_active, e_fmaps)):
                idx0 = np.flatnonzero(e["q0"] > 0)
                idx1 = np.flatnonzero(e["q1"] > 0)
                p0 = rng.choice(idx0, size=min(M, len(idx0)), replace=False)
                p1 = rng.choice(idx1, size=min(M, len(idx1)), replace=False)
                w0 = e["q0"][p0] / e["q0"][p0].sum()
                w1 = e["q1"][p1] / e["q1"][p1].sum()
                X_a[r] = (w1 @ fm["anchor40"][p1]) - (w0 @ fm["anchor40"][p0])
                X_p[r] = (w1 @ fm["pos180"][p1]) - (w0 @ fm["pos180"][p0])
            for X_curr, bucket in [(X_a, aucs_a40), (X_p, aucs_180)]:
                sc = np.zeros(len(pert_active))
                for lin in np.unique(lineages):
                    tr = lineages != lin
                    te = lineages == lin
                    mu = X_curr[tr].mean(axis=0, keepdims=True)
                    sd = np.where(X_curr[tr].std(axis=0, keepdims=True) > 1e-12, X_curr[tr].std(axis=0, keepdims=True), 1.0)
                    clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear", random_state=SEED).fit((X_curr[tr] - mu) / sd, y_inf[tr])
                    sc[te] = clf.decision_function((X_curr[te] - mu) / sd)
                bucket.append(float(roc_auc_score(y_inf, sc)))
        budget_rows.append(
            {
                "peptide_budget_M": M,
                "mean_auroc_anchor40": float(np.mean(aucs_a40)),
                "sd_auroc_anchor40": float(np.std(aucs_a40)),
                "mean_auroc_pos180": float(np.mean(aucs_180)),
                "sd_auroc_pos180": float(np.std(aucs_180)),
            }
        )
    pd.DataFrame(budget_rows).to_csv(out_dir / "exp7d_peptide_budget_curve.tsv", sep="\t", index=False)

    return {
        "newey_5fold_disjoint_gene_splits": {
            "coarse96_concordance_per_20pct_gene_split": newey_split_c96,
            "coarse96_mean_concordance": float(np.mean(newey_split_c96)),
            "aa20_concordance_per_20pct_gene_split": newey_split_aa20,
            "aa20_mean_concordance": float(np.mean(newey_split_aa20)),
        },
        "cross_lineage_5fold_disjoint_gene_splits": {
            "within_split_mean_auroc_pos180": float(np.mean(within_180)),
            "cross_split_disjoint_genes_mean_auroc_pos180": float(np.mean(cross_180)),
            "mean_pairwise_score_r_pos180": float(np.mean(pair_r_180)),
            "within_split_mean_auroc_anchor40": float(np.mean(within_a40)),
            "cross_split_disjoint_genes_mean_auroc_anchor40": float(np.mean(cross_a40)),
            "mean_pairwise_score_r_anchor40": float(np.mean(pair_r_a40)),
        },
        "ablation_summary": ablation_rows,
        "budget_curve": budget_rows,
    }


def run_exp7e(cfg: PathResolver, entries: list[dict], out_dir: Path) -> dict:
    b_rows = []
    for e in entries:
        if e["study"] == "Bernhardt2025":
            b_rows.append(
                {
                    "cohort": "Bernhardt2025_pSILAC",
                    "contrast_id": e["contrast_id"],
                    "cell_subline": e["cell_subline"],
                    "bio_state": e["bio_state"],
                    "time_h": e["time_h"],
                    "l2_shift_coarse96": float(np.linalg.norm(e["d_coarse96"])),
                    "l2_shift_pos180": float(np.linalg.norm(e["d_pos180"])),
                    "d_mda_share_pct": float(100.0 * e["d_mda_share"]),
                }
            )

    s_pep = pd.read_csv(cfg.followup("stopfer2022_peptides.tsv.gz"), sep="\t")
    clx = s_pep[s_pep["source_file"] == "sd02"].copy()
    for (tab, cline), grp in clx.groupby(["tab", "cell_line"]):
        c_grp = grp[grp["condition"] == "Control"]
        r9 = c_grp[c_grp["length"] == 9].groupby("peptide").agg(w=("intensity", "mean"), gene=("gene", "first"))
        for (cond, td), t_grp in grp[grp["condition"] != "Control"].groupby(["condition", "time_days"]):
            t9 = t_grp[t_grp["length"] == 9].groupby("peptide").agg(w=("intensity", "mean"), gene=("gene", "first"))
            peps = sorted(set(r9.index) | set(t9.index))
            if len(peps) < 100:
                continue
            w0 = r9["w"].reindex(peps).fillna(0.0).values
            w1 = t9["w"].reindex(peps).fillna(0.0).values
            q0, q1 = w0 / w0.sum(), w1 / w1.sum()
            dq = q1 - q0
            fm = encode_9mer_features(peps)
            genes = [str(t9["gene"].get(p, r9["gene"].get(p, "")) or "").split(";")[0] for p in peps]
            d_mda = sum(q1[i] - q0[i] for i, g in enumerate(genes) if g in MDA_PANEL)
            b_rows.append(
                {
                    "cohort": "Stopfer2022_Xenograft_TimeCourse",
                    "contrast_id": f"{tab}_{cond}_{int(td)}d",
                    "cell_subline": cline,
                    "bio_state": f"Xenograft_{cond}",
                    "time_h": float(td * 24.0),
                    "l2_shift_coarse96": float(np.linalg.norm(dq @ fm["coarse96"])),
                    "l2_shift_pos180": float(np.linalg.norm(dq @ fm["pos180"])),
                    "d_mda_share_pct": float(100.0 * d_mda),
                }
            )

    df_t = pd.DataFrame(b_rows).sort_values(["cohort", "cell_subline", "bio_state", "time_h"])
    df_t.to_csv(out_dir / "exp7e_temporal_trajectories.tsv", sep="\t", index=False)

    bern_sens_24 = df_t[(df_t["cohort"] == "Bernhardt2025_pSILAC") & (df_t["bio_state"] == "MAPKi_Sensitive_Response") & (df_t["time_h"] == 24)]["l2_shift_pos180"].mean()
    bern_sens_late = df_t[(df_t["cohort"] == "Bernhardt2025_pSILAC") & (df_t["bio_state"] == "MAPKi_Sensitive_Response") & (df_t["time_h"] >= 48)]["l2_shift_pos180"].mean()
    bern_res_late = df_t[(df_t["cohort"] == "Bernhardt2025_pSILAC") & (df_t["bio_state"] == "MAPKi_Resistant_NoResponse") & (df_t["time_h"] >= 48)]["l2_shift_pos180"].mean()
    bern_mock_late = df_t[(df_t["cohort"] == "Bernhardt2025_pSILAC") & (df_t["bio_state"] == "Mock_Control") & (df_t["time_h"] >= 48)]["l2_shift_pos180"].mean()
    clx_mda_1d = df_t[(df_t["cohort"] == "Stopfer2022_Xenograft_TimeCourse") & (df_t["time_h"] == 24)]["d_mda_share_pct"].mean()
    clx_mda_3d = df_t[(df_t["cohort"] == "Stopfer2022_Xenograft_TimeCourse") & (df_t["time_h"] == 72)]["d_mda_share_pct"].mean()

    return {
        "bernhardt_sensitive_pos180_shift_24h": float(bern_sens_24),
        "bernhardt_sensitive_pos180_shift_48_72h": float(bern_sens_late),
        "bernhardt_resistant_pos180_shift_72h": float(bern_res_late),
        "bernhardt_mock_pos180_shift_72h": float(bern_mock_late),
        "stopfer_xenograft_mda_share_gain_24h_pct": float(clx_mda_1d),
        "stopfer_xenograft_mda_share_gain_72h_pct": float(clx_mda_3d),
    }


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp7_distributional_sensing")
    out_dir.mkdir(parents=True, exist_ok=True)

    res_7a = run_exp7a(cfg, out_dir)
    samps_newey, rungs_newey, p9_newey = build_newey_rungs(cfg)
    newey_ladder = eval_newey_single_and_paired(samps_newey, rungs_newey)
    velz_ladder = eval_velz_ladder(cfg)
    pd.DataFrame(newey_ladder).to_csv(out_dir / "exp7b_newey_representation_ladder.tsv", sep="\t", index=False)
    pd.DataFrame(velz_ladder).to_csv(out_dir / "exp7b_velz_representation_ladder.tsv", sep="\t", index=False)

    entries = build_paired_shift_catalog(cfg)
    res_7c = run_exp7c(cfg, entries, out_dir)
    res_7d = run_exp7d(entries, samps_newey, p9_newey, out_dir)
    res_7e = run_exp7e(cfg, entries, out_dir)

    summary = {
        "n_paired_shift_contrasts_catalog": len(entries),
        "exp7a_followup_elasticity_and_ub": res_7a,
        "exp7b_newey_ladder": newey_ladder,
        "exp7b_velz_ladder": velz_ladder,
        "exp7c_transfer_and_resistance": res_7c,
        "exp7d_distributed_redundancy": res_7d,
        "exp7e_temporal": res_7e,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
