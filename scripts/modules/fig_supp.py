"""Generate publication-grade Supplementary Figures S1-S5 for supplement.tex.

Style invariants match make_paper_figures.py:
  - No subplot titles (ax.set_title forbidden).
  - Bold panel tags 'A.', 'B.', 'C.', 'D.' outside axes.
  - Native widths matched to 0.98\\textwidth (6.66 in) so rendered PDF text is >= 7.0 pt everywhere.
  - Okabe-Ito semantic palette + fonttype 42 vector PDFs + 300-DPI PNG previews.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO = Path(__file__).resolve().parents[2]
RES6 = REPO / "results" / "salvage" / "exp6_referee_falsifications"
RES7 = REPO / "results" / "salvage" / "exp7_distributional_sensing"
RES8 = REPO / "results" / "salvage" / "exp8_immunity_bridge"
RES9 = REPO / "results" / "salvage" / "exp9_post_meeting"
FIG_DIR = REPO / "manuscript" / "figures_paper"

C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_PURP = "#CC79A7"
C_GOLD = "#E69F00"
C_SKY = "#56B4E9"
C_GRAY = "#666666"
C_LIGHT = "#E5E7EB"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.2,
        "axes.labelsize": 8.6,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.05) -> None:
    ax.text(x, y, f"{label}.", transform=ax.transAxes, fontsize=10.8, fontweight="bold", va="bottom", ha="left", color="#111111")


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_fig_s1() -> None:
    """Fig S1: Split-Half Replicate Reliability Sensitivity & Multiplicity-Matched Sibling Null."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.45, 2.95))
    fig.subplots_adjust(wspace=0.38)
    df = pd.read_csv(RES6 / "disattenuation_and_sibling_null.tsv", sep="\t")
    x = np.arange(len(df))
    xlabs = [f"{r['background']}" + (" (TGFβ)" if r["perturbation"] != "IFNG" else "") for _, r in df.iterrows()]

    add_panel_label(ax_a, "A")
    w = 0.26
    ax_a.bar(x - w, 100.0 * df["r2_raw"], width=w, color=C_VERM, label="Observed OLS $R^2$")
    ax_a.bar(x, 100.0 * df["r2_fully_disattenuated"], width=w, color=C_GOLD, label="Disattenuated $R^2$ (rep. SE)")
    ax_a.bar(x + w, 100.0 * df["dPep_reliability_ceiling"], width=w, color=C_SKY, label="$\\Delta\\log_2$ Pep reliability")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabs, fontsize=7.2, rotation=24, ha="right")
    ax_a.set_ylim(0, 122)
    ax_a.set_ylabel("Variance share / reliability (%)")
    ax_a.legend(frameon=False, loc="upper right", fontsize=7.1)
    despine(ax_a)

    add_panel_label(ax_b, "B")
    w2 = 0.35
    ax_b.bar(x - w2 / 2, 100.0 * df["obs_frac_gt2x_split"], width=w2, color=C_BLUE, label="Observed $>2\\times$ split")
    ax_b.bar(x + w2 / 2, 100.0 * df["multiplicity_noise_null_gt2x"], width=w2, color=C_GRAY, alpha=0.75, label="Multiplicity noise null")
    yerr_hi = 100.0 * (df["multiplicity_noise_null_p95"] - df["multiplicity_noise_null_gt2x"])
    ax_b.errorbar(x + w2 / 2, 100.0 * df["multiplicity_noise_null_gt2x"], yerr=[np.zeros(len(df)), yerr_hi], fmt="none", ecolor="#111111", lw=0.9, capsize=2.2)
    for xi, (_, r) in enumerate(df.iterrows()):
        ax_b.text(xi - w2 / 2, 100.0 * r["obs_frac_gt2x_split"] + 1.5, f"{r['fold_over_noise_null']:.1f}×", ha="center", fontsize=7.2, fontweight="bold", color=C_BLUE)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xlabs, fontsize=7.2, rotation=24, ha="right")
    ax_b.set_ylim(0, 78)
    ax_b.set_ylabel("Multi-ligand genes $>2\\times$ (%)")
    ax_b.legend(frameon=False, loc="upper right", fontsize=7.1)
    despine(ax_b)

    save_fig(fig, "figS1_disattenuation_and_noise_null")


def make_fig_s2() -> None:
    """Fig S2: Within-Exact-HLA-Allele ERAP1 Trimming (with Exact 10! Patient Perm P) & Jackknife in MONTE."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.45, 2.95))
    fig.subplots_adjust(wspace=0.40)

    al = pd.read_csv(RES6 / "monte_within_exact_allele_erap1.tsv", sep="\t")
    jk = pd.read_csv(RES6 / "monte_jackknife_sensitivity.tsv", sep="\t")
    s8 = json.loads((RES8 / "summary.json").read_text())
    p_10fac = s8["monte_erap1_exact_10factorial_permutation"]["exact_two_sided_patient_permutation_p"]

    add_panel_label(ax_a, "A")
    ax_a.scatter(al["erap1_within_allele"], al["len_within_allele"], s=34, color=C_BLUE, alpha=0.86, edgecolors="white", lw=0.4)
    xg = np.linspace(al["erap1_within_allele"].min() - 0.2, al["erap1_within_allele"].max() + 0.2, 100)
    sl, ic = np.polyfit(al["erap1_within_allele"], al["len_within_allele"], 1)
    ax_a.plot(xg, sl * xg + ic, ls="--", lw=1.1, color="#111111")
    ax_a.axhline(0.0, ls=":", lw=0.7, color=C_GRAY)
    ax_a.axvline(0.0, ls=":", lw=0.7, color=C_GRAY)
    ax_a.text(
        0.04,
        0.05,
        f"16 alleles ($N=35$)\n$r = -0.75$\n10! patient perm $P = {p_10fac:.4f}$",
        transform=ax_a.transAxes,
        va="bottom",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_a.set_xlabel("Within-allele ERAP1 ($\\Delta\\log_2$ TMT)")
    ax_a.set_ylabel("Within-allele length ($\\Delta$ AA)")
    despine(ax_a)

    add_panel_label(ax_b, "B")
    y_idx = np.arange(len(jk))
    short_pts = [p.replace("C3N-", "").replace("C3L-", "") for p in jk["omitted_patient"]]
    ax_b.scatter(jk["r_appm_breadth_noIP"], y_idx + 0.14, color=C_GREEN, s=30, label="APPM vs. breadth ($r$)")
    ax_b.scatter(jk["r_erap1_len_noIP"], y_idx - 0.14, color=C_BLUE, marker="s", s=28, label="ERAP1 vs. length ($r$)")
    ax_b.axvline(0.0, ls="--", lw=0.8, color=C_GRAY)
    ax_b.set_yticks(y_idx)
    ax_b.set_yticklabels([f"Drop {s}" for s in short_pts], fontsize=7.3)
    ax_b.set_xlim(-0.92, 0.96)
    ax_b.set_ylim(-0.6, 11.4)
    ax_b.set_xlabel("Leave-one-tumor-out $r$ ($N=9$)")
    ax_b.legend(frameon=False, loc="upper center", fontsize=7.1)
    despine(ax_b)

    save_fig(fig, "figS2_monte_allele_and_jackknife")


def make_fig_s3() -> None:
    """Fig S3: Linear-Scale Heavy-Chain Intensity-Gain Ratio & Newey External Validation."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.45, 2.95))
    fig.subplots_adjust(wspace=0.40)

    mol = pd.read_csv(RES6 / "pdo_molar_heavy_chain_stoichiometry.tsv", sep="\t")
    new_df = pd.read_csv(RES6 / "newey_external_hlab_sibling_validation.tsv", sep="\t")

    add_panel_label(ax_a, "A")
    cols_m = [C_GOLD, C_SKY, C_GREEN]
    for idx, (_, r) in enumerate(mol.iterrows()):
        ax_a.scatter([r["delta_molar_B_over_A_ratio"]], [r["same_gene_dPep_B_minus_A"]], s=72, color=cols_m[idx], edgecolors="#111111", lw=0.8, zorder=4)
        off = (6, -6) if r["background"] != "ITO58M" else (-86, -4)
        ax_a.annotate(
            f"{r['background']}\n(A/B={r['baseline_A_over_B_fold']:.1f}×)",
            (r["delta_molar_B_over_A_ratio"], r["same_gene_dPep_B_minus_A"]),
            xytext=off,
            textcoords="offset points",
            fontsize=7.3,
            fontweight="bold",
        )
    ax_a.plot(mol["delta_molar_B_over_A_ratio"], mol["same_gene_dPep_B_minus_A"], ls="--", lw=1.0, color="#334155")
    ax_a.axhline(0.0, ls=":", lw=0.8, color=C_GRAY)
    ax_a.axvline(1.0, ls=":", lw=0.8, color=C_GREEN)
    ax_a.set_xlim(0.0, 1.40)
    ax_a.set_ylim(-0.44, 1.20)
    ax_a.set_xlabel("Linear-scale heavy-chain gain ratio $\\Delta M_{\\mathrm{B}} / \\Delta M_{\\mathrm{A}}$")
    ax_a.set_ylabel("Same-protein $\\Delta\\log_2\\mathrm{Pep}(\\mathrm{B}-\\mathrm{A})$")
    despine(ax_a)

    add_panel_label(ax_b, "B")
    x3 = np.arange(len(new_df))
    w = 0.34
    ax_b.bar(x3 - w / 2, new_df["mean_gene_dPep_B_minus_A"], width=w, color=C_GREEN, label="Unadjusted $\\Delta\\mathrm{Pep}(\\mathrm{B}-\\mathrm{A})$")
    ax_b.bar(x3 + w / 2, new_df["pep0_controlled_ancova_B_minus_A"], width=w, color=C_BLUE, label="$\\mathrm{Pep}_0$-adjusted ANCOVA")
    p_labels = ["$P < 10^{-100}$", "$P = 0.0030$", "$P = 5.2\\times 10^{-10}$"]
    for xi, ((_, r), p_str) in enumerate(zip(new_df.iterrows(), p_labels)):
        ax_b.text(xi, max(r["mean_gene_dPep_B_minus_A"], r["pep0_controlled_ancova_B_minus_A"]) + 0.04, p_str, ha="center", fontsize=7.2)
    ax_b.set_xticks(x3)
    ax_b.set_xticklabels([f"{r['background']}\n({int(r['n_genes_both_A_and_B']):,}g)" for _, r in new_df.iterrows()], fontsize=7.4)
    ax_b.set_ylim(0, 1.42)
    ax_b.set_ylabel("Same-protein B − A shift ($\\log_2$)")
    ax_b.legend(frameon=False, loc="upper right", fontsize=7.1)
    despine(ax_b)

    save_fig(fig, "figS3_stoichiometry_and_newey_validation")


def make_fig_s4() -> None:
    """Fig S4: Exact UniProt FT_SIGNAL Cleavage-Span Audit, Length Odds Ratios & Within-Pep0-Tercile Persistence (with Affinity)."""
    fig = plt.figure(figsize=(6.45, 5.55))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.40)

    span = pd.read_csv(RES8 / "hap1_exact_signal_span_cleavage_audit.tsv", sep="\t")
    lor = pd.read_csv(RES6 / "hap1_batch_invariant_length_odds_ratios.tsv", sep="\t")
    ter = pd.read_csv(RES6 / "persistence_within_pep0_terciles.tsv", sep="\t")

    # Panel A: Exact UniProt FT_SIGNAL Cleavage-Span Retention across TAP1, TAP2, ERAP1
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    kos = ["TAP1", "TAP2", "ERAP1"]
    cats = [
        ("1_Strictly_Inside_Cleaved_FT_SIGNAL", "Inside signal span ($n=102$)", C_GREEN),
        ("3_Mature_Chain_Downstream_of_FT_SIGNAL", "Mature chain ($n=747$)", C_SKY),
        ("4_Unannotated_Gene_Nterm_le35", "Unann. $\\leq 35$ ($n=894$)", C_GOLD),
        ("5_Unannotated_Gene_Internal_gt35", "Unann. $>35$ ($n=13{,}379$)", C_GRAY),
    ]
    xk = np.arange(len(kos))
    w = 0.19
    for qi, (ckey, clab, col) in enumerate(cats):
        vals = [100.0 * span[(span["knockout"] == k) & (span["span_category"] == ckey)]["retention_n2_of_3"].iloc[0] for k in kos]
        ax_a.bar(xk + (qi - 1.5) * w, vals, width=w, color=col, label=clab)
    ax_a.set_xticks(xk)
    ax_a.set_xticklabels(["TAP1 KO*", "TAP2 KO*", "ERAP1 KO"], fontsize=7.5)
    ax_a.set_ylim(0, 136)
    ax_a.set_ylabel("Strict KO retention ($\\geq 2/3$, %)")
    ax_a.legend(frameon=False, loc="upper right", fontsize=7.0)
    despine(ax_a)

    # Panel B: Survivor Median dPep across exact FT_SIGNAL span categories
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    for qi, (ckey, clab, col) in enumerate(cats):
        vals = [span[(span["knockout"] == k) & (span["span_category"] == ckey)]["surv_median_dPep_log2"].iloc[0] for k in kos]
        ax_b.bar(xk + (qi - 1.5) * w, vals, width=w, color=col)
    ax_b.axhline(0.0, ls="--", lw=0.8, color="#111111")
    ax_b.annotate(
        "Inside signal span:\n+6.56 vs. −1.74\n(unann. ≤35 AA)",
        xy=(1 - 1.5 * w, 6.56),
        xytext=(1.02, 3.8),
        fontsize=7.2,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#111111"),
        bbox=dict(boxstyle="round,pad=0.18", fc="#ECFDF5", ec=C_GREEN, lw=0.6),
    )
    ax_b.set_xticks(xk)
    ax_b.set_xticklabels(["TAP1 KO*", "TAP2 KO*", "ERAP1 KO"], fontsize=7.5)
    ax_b.set_ylim(-3.4, 8.0)
    ax_b.set_ylabel("Survivor median $\\Delta\\log_2$ intensity")
    despine(ax_b)

    # Panel C: Batch-invariant within-sample (10-11mer / 8-9mer) odds ratio vs WT across all 11 KOs
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    lor_s = lor.sort_values("within_repertoire_10_11mer_odds_ratio_vs_wt", ascending=True)
    yl = np.arange(len(lor_s))
    cols_l = [C_BLUE if k == "ERAP1" else (C_VERM if k in ["TAPBP", "PDIA3", "GANAB", "TAP1", "TAP2"] else C_GRAY) for k in lor_s["knockout"]]
    ax_c.barh(yl, lor_s["within_repertoire_10_11mer_odds_ratio_vs_wt"], height=0.58, color=cols_l, alpha=0.88)
    ax_c.axvline(1.0, ls="--", lw=0.8, color="#111111")
    ax_c.set_yticks(yl)
    ax_c.set_yticklabels(lor_s["knockout"], fontsize=7.3)
    ax_c.set_xlabel("10–11-mer / 8–9-mer odds ratio vs. WT")
    despine(ax_c)

    # Panel D: 0-1 AUROC axis Within-Pep0-Tercile Durability (including Predicted HLA affinity!)
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    t_sum = ter[ter["endpoint"] == "durable"].groupby("pep0_tercile")[["auroc_mech", "auroc_pep0", "auroc_aff", "auroc_prot0"]].mean().loc[["Low_Pep0", "Mid_Pep0", "High_Pep0"]]
    xt = np.arange(3)
    wt4 = 0.19
    ax_d.bar(xt - 1.5 * wt4, t_sum["auroc_mech"], width=wt4, color=C_BLUE, label="Floor-imputed ridge")
    ax_d.bar(xt - 0.5 * wt4, t_sum["auroc_pep0"], width=wt4, color=C_GREEN, label="Baseline HLA-I MS")
    ax_d.bar(xt + 0.5 * wt4, t_sum["auroc_aff"], width=wt4, color=C_GOLD, label="Predicted HLA affinity")
    ax_d.bar(xt + 1.5 * wt4, t_sum["auroc_prot0"], width=wt4, color=C_VERM, label="Intracellular proteome")
    ax_d.axhline(0.5, ls="--", lw=0.8, color=C_GRAY)
    ax_d.set_xticks(xt)
    ax_d.set_xticklabels(["Low $\\mathrm{Pep}_0$", "Mid $\\mathrm{Pep}_0$", "High $\\mathrm{Pep}_0$"], fontsize=7.5)
    ax_d.set_ylim(0.0, 1.12)
    ax_d.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_d.set_ylabel("Mean LOBO durability AUROC")
    ax_d.legend(frameon=False, loc="upper right", fontsize=7.0)
    despine(ax_d)

    save_fig(fig, "figS4_sec61_2x2_and_tercile_persistence")


def make_fig_s5() -> None:
    """Fig S5: Distributed Gene Ablation, Peptide-Budget Scaling (Nested vs Fixed C), Within-Locus Breakdown & Temporal Dynamics."""
    fig = plt.figure(figsize=(6.45, 5.55))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.40)

    # Panel A: Top-Gene Ablation & 5-Fold Disjoint Gene Splits (0-1 AUROC axis)
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    abl = pd.read_csv(RES7 / "exp7d_marker_ablation_vs_panels.tsv", sep="\t")
    coll = abl[abl["mode"] == "Collective_Distribution_After_TopGene_Ablation"].copy()
    x_rem = 100.0 * coll["frac_genes_removed"].values
    ax_a.plot(x_rem, coll["lolo_auroc_pos180"], "-o", color=C_BLUE, lw=1.5, ms=4.5, label="Full 9-mer (P1–P9)")
    ax_a.plot(x_rem, coll["lolo_auroc_anchor40"], "-s", color=C_GREEN, lw=1.4, ms=4.2, label="HLA anchors (P2, P9)")
    top1_auc = float(abl[abl["mode"] == "Top_1_Peptide_Markers_Only"]["lolo_auroc_pos180"].iloc[0])
    top10_auc = float(abl[abl["mode"] == "Top_10_Peptide_Markers_Only"]["lolo_auroc_pos180"].iloc[0])
    ax_a.axhline(top10_auc, ls="-.", lw=1.0, color=C_GOLD, label=f"Top-10 peptides ({top10_auc:.2f})")
    ax_a.axhline(top1_auc, ls="--", lw=1.0, color=C_VERM, label=f"Top-1 peptide ({top1_auc:.2f})")
    ax_a.set_xlim(-2, 53)
    ax_a.set_ylim(0.0, 1.15)
    ax_a.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_xlabel("Top training genes ablated (%)")
    ax_a.set_ylabel("LOLO AUROC")
    ax_a.legend(frameon=False, loc="lower left", fontsize=7.0)
    despine(ax_a)

    # Panel B: Peptide Budget M Curve Comparing Nested-CV C vs Fixed C=0.1 (0-1 AUROC axis)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    bud = pd.read_csv(RES8 / "peptide_budget_curve_nested_vs_fixed_C.csv")
    m_vals = bud["peptide_budget_M"].values
    ax_b.plot(m_vals, bud["mean_lolo_auroc_nested_C"], "-o", color=C_BLUE, lw=1.5, ms=4.5, label="Full 9-mer (nested $C$)")
    ax_b.plot(m_vals, bud["mean_lolo_auroc_fixed_C01"], "--s", color=C_SKY, lw=1.4, ms=4.2, label="Full 9-mer (fixed $C=0.1$)")
    ax_b.axhline(0.50, ls=":", lw=0.8, color=C_GRAY)
    ax_b.set_xscale("log")
    ax_b.set_xticks([25, 50, 100, 250, 500, 1000, 2000])
    ax_b.set_xticklabels(["25", "50", "100", "250", "500", "1k", "2k"], fontsize=7.3)
    ax_b.minorticks_off()
    ax_b.set_ylim(0.0, 1.12)
    ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_b.set_xlabel("Subsampled 9-mer budget ($M$)")
    ax_b.set_ylabel("LOLO AUROC")
    ax_b.legend(frameon=False, loc="lower right", fontsize=7.1)
    despine(ax_b)

    # Panel C: Within-HLA-A vs Within-HLA-B vs Locus Reallocation (20 Author-Annotated Contrasts)
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    loc_df = pd.read_csv(RES7 / "exp7c_within_vs_between_locus_decomposition.tsv", sep="\t")
    labels_c = ["Locus share (3 df)", "Within HLA-A", "Within HLA-B", "Full 9-mer (P1–P9)"]
    ya = np.arange(len(labels_c))[::-1]
    vals_c = loc_df["lolo_auroc"].values
    cols_c = [C_GOLD, C_SKY, C_VERM, C_BLUE]
    ax_c.barh(ya, vals_c, height=0.55, color=cols_c, alpha=0.88)
    ax_c.axvline(0.50, ls="--", lw=0.8, color=C_GRAY)
    for yv, vv in zip(ya, vals_c):
        ax_c.text(vv + 0.02, yv, f"{vv:.2f}", va="center", fontsize=7.4, fontweight="bold" if vv > 0.65 else "normal")
    ax_c.set_yticks(ya)
    ax_c.set_yticklabels(labels_c, fontsize=7.5)
    ax_c.set_xlim(0.0, 1.12)
    ax_c.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_c.set_xlabel("LOLO AUROC (20 author-typed)")
    despine(ax_c)

    # Panel D: Temporal Trajectories (Bernhardt pSILAC) + Aranha 2026 PXD085506 Matched ISG Induction
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    traj = pd.read_csv(RES7 / "exp7e_temporal_trajectories.tsv", sep="\t")
    bp = traj[traj["cohort"] == "Bernhardt2025_pSILAC"]
    cats_t = [
        ("MAPKi_Sensitive_Response", "Sensitive + MAPKi", C_GREEN, "-o"),
        ("MAPKi_Resistant_NoResponse", "Resistant + MAPKi", C_VERM, "--s"),
        ("Mock_Control", "Mock DMSO", C_GRAY, ":^"),
    ]
    for bstate, lbl, col, fmt in cats_t:
        g24 = bp[(bp["bio_state"] == bstate) & (bp["time_h"] == 24)]["l2_shift_pos180"].mean()
        g72 = bp[(bp["bio_state"] == bstate) & (bp["time_h"] >= 48)]["l2_shift_pos180"].mean()
        ax_d.plot([24, 72], [g24, g72], fmt, color=col, lw=1.5, ms=5.0, label=lbl)
    p85 = pd.read_csv(RES8 / "external_aranha2026_pxd085506_proteome_state.csv")
    ax_d.text(
        0.04,
        0.95,
        f"Aranha HAP1 Hallmark IFN-$\\gamma$:\n+{p85['hallmark_ifng_mean_l2fc'].min():.2f} to +{p85['hallmark_ifng_mean_l2fc'].max():.2f} $\\log_2$FC\nacross all 4 genotypes",
        transform=ax_d.transAxes,
        va="top",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_d.set_xticks([24, 72])
    ax_d.set_xticklabels(["Early (24 h)", "Late (48–72 h)"], fontsize=7.5)
    ax_d.set_xlim(14, 82)
    ax_d.set_ylim(-0.02, 0.36)
    ax_d.set_ylabel("Shift norm $\\|\\Delta q\\|_2$")
    ax_d.legend(frameon=False, loc="lower left", fontsize=7.0)
    despine(ax_d)

    save_fig(fig, "figS5_followup_temporal_and_locus")


CHEM_AA_ORDER = [
    ("L", "L/I", C_BLUE), ("V", "V", C_BLUE), ("M", "M", C_BLUE), ("A", "A", C_BLUE),
    ("F", "F", C_BLUE), ("Y", "Y", C_BLUE), ("W", "W", C_BLUE),
    ("P", "P", C_GOLD),
    ("G", "G", C_GREEN), ("S", "S", C_GREEN), ("T", "T", C_GREEN),
    ("N", "N", C_GREEN), ("Q", "Q", C_GREEN), ("C", "C", C_GREEN),
    ("R", "R", C_VERM), ("K", "K", C_VERM), ("H", "H", C_VERM),
    ("E", "E", C_PURP), ("D", "D", C_PURP),
]
CHEM_DIVIDERS = [6.5, 7.5, 13.5, 16.5]


def _draw_supp_heatmap(ax: plt.Axes, df_sub: pd.DataFrame, val_col: str, vmax: float, thresh: float, cbar_lab: str) -> None:
    aa_keys = [x[0] for x in CHEM_AA_ORDER]
    aa_disp = [x[1] for x in CHEM_AA_ORDER]
    piv = df_sub.pivot(index="aa", columns="position", values=val_col).reindex(index=aa_keys, columns=range(1, 10))
    mat = piv.values
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for y_div in CHEM_DIVIDERS:
        ax.axhline(y_div, color="#334155", lw=0.7, alpha=0.75)
    for ri, (_, _, c_hex) in enumerate(CHEM_AA_ORDER):
        ax.add_patch(mpatches.Rectangle((-0.78, ri - 0.5), 0.28, 1.0, fc=c_hex, ec="none", clip_on=False, zorder=4))
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if np.isfinite(v) and abs(v) >= thresh:
                ax.text(ci, ri, f"{v:+.1f}", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white" if abs(v) > 0.56 * vmax else "#111111")
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"P{k}" for k in range(1, 10)], fontsize=7.0)
    ax.set_yticks(range(len(aa_disp)))
    ax.set_yticklabels(aa_disp, fontsize=6.7)
    ax.tick_params(axis="y", pad=5.5, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.044, pad=0.025)
    cbar.ax.tick_params(labelsize=6.6)
    cbar.set_label(cbar_lab, fontsize=7.0)


def make_fig_s6() -> None:
    """Fig S6: Pooled MAPKi vs Within-Locus (HLA-A, HLA-B, HLA-C) IFN-gamma 19x9 Chemistry-Grouped Heatmaps."""
    fig = plt.figure(figsize=(6.65, 6.20))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.42)

    logo_all = pd.read_csv(RES9 / "pssm_shift_and_decoder_logo_matrices.csv")
    logo_il = logo_all[logo_all["sequence_coding"] == "Isobaric_IL_Harmonized"].copy()
    wl_all = pd.read_csv(RES9 / "pssm_within_locus_ifng_shifts.csv")
    wl_il = wl_all[wl_all["sequence_coding"] == "Isobaric_IL_Harmonized"].copy()

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A", x=-0.18, y=1.03)
    _draw_supp_heatmap(ax_a, logo_il, "mean_dq_mapki_pp", vmax=2.0, thresh=0.95, cbar_lab="MAPKi $\\Delta q$ (pp)")

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B", x=-0.18, y=1.03)
    _draw_supp_heatmap(ax_b, wl_il[wl_il["locus"] == "HLA-A"], "mean_within_locus_ifng_pp", vmax=4.0, thresh=1.80, cbar_lab="Within HLA-A $\\Delta\\mu_{\\mathrm{A}}$ (pp)")

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C", x=-0.18, y=1.03)
    _draw_supp_heatmap(ax_c, wl_il[wl_il["locus"] == "HLA-B"], "mean_within_locus_ifng_pp", vmax=6.2, thresh=2.80, cbar_lab="Within HLA-B $\\Delta\\mu_{\\mathrm{B}}$ (pp)")

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D", x=-0.18, y=1.03)
    _draw_supp_heatmap(ax_d, wl_il[wl_il["locus"] == "HLA-C"], "mean_within_locus_ifng_pp", vmax=5.5, thresh=2.50, cbar_lab="Within HLA-C $\\Delta\\mu_{\\mathrm{C}}$ (pp)")

    chem_leg = [
        mpatches.Patch(fc=C_BLUE, label="Hydrophobic"),
        mpatches.Patch(fc=C_GOLD, label="Proline"),
        mpatches.Patch(fc=C_GREEN, label="Polar"),
        mpatches.Patch(fc=C_VERM, label="Basic"),
        mpatches.Patch(fc=C_PURP, label="Acidic"),
    ]
    fig.legend(handles=chem_leg, frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.02), ncol=5, fontsize=6.8, handlelength=0.9, columnspacing=0.8)

    save_fig(fig, "figS6_within_locus_pssm_logos")


def run(res_root: Path, fig_dir: Path) -> list[str]:
    global RES, RES6, RES7, RES8, RES9, FIG_DIR
    RES = Path(res_root)
    RES6 = RES / "falsification_controls" if (RES / "falsification_controls").exists() else RES / "exp6_referee_falsifications"
    RES7 = RES / "exp7_distributional_sensing"
    RES8 = RES / "exp8_immunity_bridge"
    RES9 = RES / "exp9_post_meeting"
    FIG_DIR = Path(fig_dir)
    make_fig_s1()
    make_fig_s2()
    make_fig_s3()
    make_fig_s4()
    make_fig_s5()
    make_fig_s6()
    return sorted([p.name for p in FIG_DIR.glob("figS*.pdf")])
