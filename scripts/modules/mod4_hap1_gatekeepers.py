"""Module 4: Isogenic CRISPR Dissection of 11 Antigen-Processing Gatekeepers
(HAP1 Atlas) and Co-Translational Sec61 Topological Bypass Under TAP Loss.
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


def load_canonical_sequences(cfg: PathResolver) -> dict[str, str]:
    fasta_path = cfg.raw(
        "visibility_hap1",
        "search",
        "2024-02-24-decoys-contam-uniprotkb_Human_AND_reviewed_true_AND_m_2024_02_24_wiRT.fasta.fas",
    )
    gene_seqs: dict[str, str] = {}
    cur_gene, cur_seq = None, []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_gene and cur_seq:
                    s = "".join(cur_seq)
                    if cur_gene not in gene_seqs or len(s) > len(gene_seqs[cur_gene]):
                        gene_seqs[cur_gene] = s
                cur_gene, cur_seq = None, []
                if not line.startswith(">rev_") and not line.startswith(">contam_"):
                    for tok in line[1:].split():
                        if tok.startswith("GN="):
                            cur_gene = tok[3:].strip()
                            break
            else:
                cur_seq.append(line.strip())
    if cur_gene and cur_seq:
        s = "".join(cur_seq)
        if cur_gene not in gene_seqs or len(s) > len(gene_seqs[cur_gene]):
            gene_seqs[cur_gene] = s
    return gene_seqs


def load_hallmark_sets(cfg: PathResolver) -> dict[str, set[str]]:
    gmt = cfg.raw("visibility_mhc1tip", "MB_39_TGFb_mhc1tip", "h.all.v2023.2.Hs.symbols.gmt")
    sets: dict[str, set[str]] = {}
    with open(gmt) as f:
        for line in f:
            parts = line.strip().split("\t")
            sets[parts[0].replace("HALLMARK_", "")] = set(parts[2:])
    return sets


def run(cfg: PathResolver | None = None) -> dict:
    cfg = cfg or resolve_config()
    out_dir = cfg.results("exp4_hap1_gatekeepers")
    out_dir.mkdir(parents=True, exist_ok=True)

    hdir = cfg.processed("visibility_hap1_gene_unique")
    samps = pd.read_csv(hdir / "samples.tsv", sep="\t")
    samps = samps[samps["included"] == True]
    peps = pd.read_csv(hdir / "peptides.tsv.gz", sep="\t")
    prots = pd.read_csv(hdir / "proteins.tsv.gz", sep="\t")
    bdf = pd.read_csv(cfg.processed("visibility_binding", "hap1_best.tsv.gz"), sep="\t").drop_duplicates("peptide")
    topo = pd.read_csv(cfg.annotations("gene_topology_summary.tsv.gz"), sep="\t").set_index("gene")
    gene_seqs = load_canonical_sequences(cfg)
    hallmark = load_hallmark_sets(cfg)

    peps_pos = peps[
        peps["normalized_intensity_positive_only"].notna() & (peps["normalized_intensity_positive_only"] > 0)
    ].copy()
    peps_pos["log2_pep"] = np.log2(peps_pos["normalized_intensity_positive_only"])

    wt_pep_s = samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["sample_id"].tolist()
    wt_prot_s = samps[(samps["modality"] == "protein") & (samps["condition"] == "control")]["sample_id"].tolist()
    wt_dates = set(samps[(samps["modality"] == "peptide") & (samps["condition"] == "control")]["run_dates"].astype(str))

    wt_any_peps = set(peps_pos[peps_pos["sample_id"].isin(wt_pep_s)]["peptide"].unique())
    wt_p = (
        peps_pos[peps_pos["sample_id"].isin(wt_pep_s)]
        .groupby(["peptide", "gene"])
        .agg(pep0=("log2_pep", "mean"), n0=("log2_pep", "count"))
        .reset_index()
    )
    wt_p = wt_p[wt_p["n0"] >= 2].merge(
        bdf[["peptide", "allele", "affinity_nM", "binding_rank_percent"]], on="peptide", how="left"
    )
    wt_p["length"] = wt_p["peptide"].str.len()

    def get_start_aa(row: pd.Series) -> int:
        s = gene_seqs.get(str(row["gene"]), "")
        if not s:
            return -1
        idx = s.find(str(row["peptide"]))
        return idx + 1 if idx >= 0 else -1

    sig_genes = set(topo.index[topo["has_signal_peptide"].astype(bool)])
    sec_genes = set(topo.index[topo["is_secretory_er_membrane"].astype(bool)])
    wt_p["start_aa"] = wt_p.apply(get_start_aa, axis=1)
    wt_p["is_nterm_leader_35aa"] = (wt_p["start_aa"] >= 1) & (wt_p["start_aa"] <= 35)
    wt_p["has_uniprot_signal"] = wt_p["gene"].isin(sig_genes)
    wt_p["is_secretory_er"] = wt_p["gene"].isin(sec_genes)

    wt_pr = (
        prots[prots["sample_id"].isin(wt_prot_s) & np.isfinite(prots["log2_abundance"])]
        .groupby("gene")
        .agg(prot0=("log2_abundance", "mean"), pn0=("log2_abundance", "count"))
    )
    wt_pr = wt_pr[wt_pr["pn0"] >= 2]

    kos = sorted([c for c in samps["condition"].unique() if c != "control"])
    ko_rows = []
    leader_rows = []

    for ko in kos:
        ko_pep_s = samps[(samps["modality"] == "peptide") & (samps["condition"] == ko)]["sample_id"].tolist()
        ko_prot_s = samps[(samps["modality"] == "protein") & (samps["condition"] == ko)]["sample_id"].tolist()
        ko_dates = set(samps[(samps["modality"] == "peptide") & (samps["condition"] == ko)]["run_dates"].astype(str))
        same_date = bool(ko_dates == wt_dates)

        kp = (
            peps_pos[peps_pos["sample_id"].isin(ko_pep_s)]
            .groupby("peptide")
            .agg(pep1=("log2_pep", "mean"), n1=("log2_pep", "count"))
            .reset_index()
        )
        m = wt_p.merge(kp, on="peptide", how="left")
        m["n1"] = m["n1"].fillna(0).astype(int)
        m["retained"] = m["n1"] >= 2
        m["dPep"] = m["pep1"] - m["pep0"]

        kpr = (
            prots[prots["sample_id"].isin(ko_prot_s) & np.isfinite(prots["log2_abundance"])]
            .groupby("gene")
            .agg(prot1=("log2_abundance", "mean"), pn1=("log2_abundance", "count"))
        )
        kpr = kpr[kpr["pn1"] >= 2]
        prm = wt_pr.join(kpr, how="inner")
        prm["dProt"] = prm["prot1"] - prm["prot0"]
        m = m.merge(prm[["dProt"]].reset_index(), on="gene", how="left")
        m["u_resid"] = m["dPep"] - m["dProt"]

        kp_denovo = kp[(kp["n1"] >= 2) & (~kp["peptide"].isin(wt_any_peps))].copy()
        kp_denovo["length"] = kp_denovo["peptide"].str.len()

        ret_len = m.groupby("length")["retained"].mean().to_dict()
        ret_all = m.groupby("allele")["retained"].mean().to_dict()
        surv = m[m["retained"]].copy()
        dn_10_11 = float(kp_denovo["length"].isin([10, 11]).mean()) if len(kp_denovo) > 0 else np.nan

        ko_rows.append(
            {
                "knockout": ko.replace("_KO", ""),
                "same_acquisition_date_as_wt": same_date,
                "n_baseline_peptides": int(len(m)),
                "n_surviving_peptides": int(len(surv)),
                "overall_retention": float(m["retained"].mean()),
                "n_denovo_peptides": int(len(kp_denovo)),
                "denovo_frac_10_11mer": dn_10_11,
                "ret_8mer": float(ret_len.get(8, np.nan)),
                "ret_9mer": float(ret_len.get(9, np.nan)),
                "ret_10mer": float(ret_len.get(10, np.nan)),
                "ret_11mer": float(ret_len.get(11, np.nan)),
                "ret_10_11_minus_8_9": float(
                    m[m["length"].isin([10, 11])]["retained"].mean() - m[m["length"].isin([8, 9])]["retained"].mean()
                ),
                "ret_HLA_A0201": float(ret_all.get("HLA-A*02:01", np.nan)),
                "ret_HLA_B4001": float(ret_all.get("HLA-B*40:01", np.nan)),
                "ret_HLA_C0304": float(ret_all.get("HLA-C*03:04", np.nan)),
            }
        )

        ldr = m["is_nterm_leader_35aa"]
        ct = pd.crosstab(ldr, m["retained"])
        odds, fisher_p = stats.fisher_exact(ct.values) if ct.shape == (2, 2) else (np.nan, np.nan)
        surv_ldr = surv[surv["is_nterm_leader_35aa"]]["dPep"].dropna()
        surv_int = surv[~surv["is_nterm_leader_35aa"]]["dPep"].dropna()
        mwu_p = float(stats.mannwhitneyu(surv_ldr, surv_int).pvalue) if (len(surv_ldr) >= 3 and len(surv_int) >= 3) else np.nan
        leader_rows.append(
            {
                "knockout": ko.replace("_KO", ""),
                "same_date": same_date,
                "ret_leader_1_35aa": float(m.loc[ldr, "retained"].mean()),
                "ret_internal_gt35aa": float(m.loc[~ldr, "retained"].mean()),
                "ret_odds_ratio": float(odds),
                "ret_fisher_p": float(fisher_p),
                "surv_median_dPep_leader": float(surv_ldr.median()) if len(surv_ldr) else np.nan,
                "surv_median_dPep_internal": float(surv_int.median()) if len(surv_int) else np.nan,
                "surv_leader_minus_internal_log2": float(surv_ldr.median() - surv_int.median()) if (len(surv_ldr) and len(surv_int)) else np.nan,
                "surv_mwu_p": mwu_p,
            }
        )

        if ko in ("TAP1_KO", "TAP2_KO"):
            hm_rows = []
            for hname, hgenes in hallmark.items():
                sub_h = m[m["gene"].isin(hgenes)]
                if sub_h["gene"].nunique() < 10:
                    continue
                g_ret = sub_h.groupby("gene")["retained"].mean()
                sub_surv = sub_h[sub_h["retained"] & sub_h["u_resid"].notna()]
                g_ures = sub_surv.groupby("gene")["u_resid"].mean() if len(sub_surv) else pd.Series(dtype=float)
                g_sec = sub_h.drop_duplicates("gene")["is_secretory_er"].mean()
                g_sig = sub_h.drop_duplicates("gene")["has_uniprot_signal"].mean()
                g_ldr = sub_h.groupby("gene")["is_nterm_leader_35aa"].mean().mean()
                hm_rows.append(
                    {
                        "knockout": ko.replace("_KO", ""),
                        "hallmark_program": hname,
                        "n_baseline_genes": int(sub_h["gene"].nunique()),
                        "n_surviving_genes": int(len(g_ures)),
                        "frac_secretory_er_genes": float(g_sec),
                        "frac_signal_peptide_genes": float(g_sig),
                        "mean_leader_peptide_frac": float(g_ldr),
                        "gene_mean_retention": float(g_ret.mean()),
                        "gene_mean_proteome_adj_u": float(g_ures.mean()) if len(g_ures) >= 5 else np.nan,
                    }
                )
            pd.DataFrame(hm_rows).to_csv(out_dir / f"hallmark50_topology_in_{ko.lower()}.tsv", sep="\t", index=False)

    ko_df = pd.DataFrame(ko_rows)
    ldr_df = pd.DataFrame(leader_rows)
    ko_df.to_csv(out_dir / "hap1_11ko_two_margin_summary.tsv", sep="\t", index=False)
    ldr_df.to_csv(out_dir / "hap1_leader_signal_peptide_sparing.tsv", sep="\t", index=False)

    pair_audit_rows = []
    for ko_tag in ["TAP1_KO", "TAP2_KO"]:
        for prog in ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_E2F_TARGETS", "HALLMARK_INTERFERON_GAMMA_RESPONSE"]:
            po_path = cfg.upstream(
                "results", "visibility_final_orthogonal", f"hap1_{ko_tag}", f"HAP1_{ko_tag}_main", f"{prog}_pair_outcomes.tsv"
            )
            po = pd.read_csv(po_path, sep="\t")
            po["t_sec"] = po["target_gene"].isin(sec_genes)
            po["c_sec"] = po["control_gene"].isin(sec_genes)
            po["t_sig"] = po["target_gene"].isin(sig_genes)
            po["c_sig"] = po["control_gene"].isin(sig_genes)
            surv = po[po["both_survive"] == True]
            mis = surv[surv["t_sec"] & (~surv["c_sec"])]
            mat = surv[surv["t_sec"] == surv["c_sec"]]
            pair_audit_rows.append(
                {
                    "knockout": ko_tag.replace("_KO", ""),
                    "program": prog.replace("HALLMARK_", ""),
                    "n_baseline_pairs": int(len(po)),
                    "target_sec_frac": float(po["t_sec"].mean()),
                    "control_sec_frac": float(po["c_sec"].mean()),
                    "target_sigpep_frac": float(po["t_sig"].mean()),
                    "control_sigpep_frac": float(po["c_sig"].mean()),
                    "n_surviving_pairs": int(len(surv)),
                    "all_survivors_mean_pair_u": float(surv["pair_u"].mean()) if len(surv) else np.nan,
                    "topology_mismatched_n": int(len(mis)),
                    "topology_mismatched_mean_pair_u": float(mis["pair_u"].mean()) if len(mis) else np.nan,
                    "topology_matched_n": int(len(mat)),
                    "topology_matched_mean_pair_u": float(mat["pair_u"].mean()) if len(mat) else np.nan,
                }
            )
    pd.DataFrame(pair_audit_rows).to_csv(out_dir / "tap_ko_topology_matched_reanalysis.tsv", sep="\t", index=False)

    tap1_hm = pd.read_csv(out_dir / "hallmark50_topology_in_tap1_ko.tsv", sep="\t")
    r_ret, p_ret = stats.pearsonr(tap1_hm["frac_signal_peptide_genes"], tap1_hm["gene_mean_retention"])
    val_u = tap1_hm.dropna(subset=["gene_mean_proteome_adj_u"])
    r_u, p_u = stats.pearsonr(val_u["frac_secretory_er_genes"], val_u["gene_mean_proteome_adj_u"])

    summary = {
        "erap1_denovo_10_11mer_frac": float(ko_df.loc[ko_df["knockout"] == "ERAP1", "denovo_frac_10_11mer"].iloc[0]),
        "tap2_leader_minus_internal_log2": float(
            ldr_df.loc[ldr_df["knockout"] == "TAP2", "surv_leader_minus_internal_log2"].iloc[0]
        ),
        "tap2_leader_fisher_p": float(ldr_df.loc[ldr_df["knockout"] == "TAP2", "ret_fisher_p"].iloc[0]),
        "tap1_hallmark50_r_sigpep_vs_retention": float(r_ret),
        "tap1_hallmark50_p_sigpep_vs_retention": float(p_ret),
        "tap1_hallmark50_r_sec_vs_u": float(r_u),
        "tap1_hallmark50_p_sec_vs_u": float(p_u),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
