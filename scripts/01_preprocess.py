#!/usr/bin/env python3
"""Stage curated UniProt human topology, strict MONTE HLA-II ligands, and melanoma follow-up cohorts.

Usage:
    python3 scripts/01_preprocess.py --data-dir ./data --processed-dir ./data/processed
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from presentation_state.config import PathResolver, resolve_config  # noqa: E402

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def gencode_ref_mapping(raw_monte: Path) -> dict:
    d: dict = {}
    with gzip.open(raw_monte / "gencode.v34.pc_translations.fa.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                a = line[1:].strip().split("|")
                rec = (a[2], a[6])
                for key in a[:2]:
                    d[key] = rec
                    d[key.split(".")[0]] = rec
    by_symbol: dict[str, set[str]] = defaultdict(set)
    for gid, symbol in d.values():
        by_symbol[symbol].add(gid.split(".")[0])
    with gzip.open(raw_monte / "gencode.v34.metadata.HGNC.gz", "rt") as fh:
        for line in fh:
            transcript, symbol, hgnc = line.strip().split("\t")
            ids = by_symbol.get(symbol, set())
            gid = next(iter(ids)) if len(ids) == 1 else hgnc
            if transcript not in d:
                d[transcript] = (gid, symbol)
    return d


def assess_accessions(raw: str | None, representative: str | None, mapping: dict, redundant=None) -> dict:
    accs = [x for x in str(raw or representative or "").split("|") if x]
    if redundant and str(redundant) not in {"NA", "na", "nan"}:
        accs.extend(str(redundant).split("|"))
    genes, gids, unresolved = set(), set(), []
    for a in accs:
        key = a.split("_")[0] if a.startswith("ENST") else a
        val = mapping.get(key)
        if not val:
            hit = re.match(r"(ENSP[0-9]+)(?:\.[0-9]+)?(?:-|$)", key)
            if hit:
                val = mapping.get(hit.group(1))
        if val:
            gids.add(val[0].split(".")[0])
            genes.add(val[1])
        else:
            unresolved.append(a)
    good = len(gids) == 1 and len(genes) == 1 and not unresolved
    return {
        "source_gene_set": "|".join(sorted(genes)),
        "source_gene_ids": "|".join(sorted(gids)),
        "n_source_genes": len(gids),
        "unresolved_accessions": "|".join(unresolved),
        "source_gene_unambiguous": good,
        "gene": next(iter(genes)) if good else None,
    }


def stage_uniprot_topology(cfg: PathResolver, out_dir: Path) -> dict:
    raw_cache = cfg.raw("annotations", "uniprot_human_topology.tsv.gz")
    dest = out_dir / "uniprot_human_topology.tsv.gz"
    if raw_cache.exists() and not dest.exists():
        shutil.copyfile(raw_cache, dest)
    elif not dest.exists():
        url = (
            "https://rest.uniprot.org/uniprotkb/stream?compressed=true"
            "&fields=accession%2Cgene_primary%2Clength%2Cft_signal%2Cft_transmem%2Ccc_subcellular_location"
            "&format=tsv&query=%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "presentation-state-deposit/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())

    up = pd.read_csv(dest, sep="\t")
    up["has_signal_peptide"] = up["Signal peptide"].notna()
    up["has_transmembrane"] = up["Transmembrane"].notna()
    loc_str = up["Subcellular location [CC]"].astype(str)
    up["is_secretory_er_membrane"] = (
        up["has_signal_peptide"]
        | up["has_transmembrane"]
        | loc_str.str.contains(
            "Endoplasmic reticulum|Golgi|Secreted|Cell membrane|Extracellular|Lysosome|Endosome",
            case=False,
            na=False,
        )
    )
    up["is_nuclear_cytosolic"] = (~up["is_secretory_er_membrane"]) & loc_str.str.contains(
        "Nucleus|Cytoplasm|Cytosol|Chromosome", case=False, na=False
    )
    gene_topo = (
        up.dropna(subset=["Gene Names (primary)"])
        .groupby("Gene Names (primary)")
        .agg(
            n_entries=("Entry", "count"),
            protein_length=("Length", "median"),
            has_signal_peptide=("has_signal_peptide", "any"),
            has_transmembrane=("has_transmembrane", "any"),
            is_secretory_er_membrane=("is_secretory_er_membrane", "any"),
            is_nuclear_cytosolic=("is_nuclear_cytosolic", "any"),
        )
        .reset_index()
        .rename(columns={"Gene Names (primary)": "gene"})
    )
    gene_path = out_dir / "gene_topology_summary.tsv.gz"
    gene_topo.to_csv(gene_path, sep="\t", index=False)
    return {
        "uniprot_raw_rows": int(len(up)),
        "unique_primary_genes": int(len(gene_topo)),
        "signal_peptide_gene_fraction": float(gene_topo["has_signal_peptide"].mean()),
        "transmembrane_gene_fraction": float(gene_topo["has_transmembrane"].mean()),
        "secretory_er_membrane_fraction": float(gene_topo["is_secretory_er_membrane"].mean()),
        "uniprot_raw_sha256": sha256_file(dest),
        "gene_topology_sha256": sha256_file(gene_path),
    }


def stage_monte_hlaii(cfg: PathResolver, out_dir: Path) -> dict:
    raw_monte = cfg.raw("visibility_monte")
    clin = pd.read_csv(cfg.processed("visibility_monte", "clinical.tsv"), sep="\t")
    mapping = gencode_ref_mapping(raw_monte)

    wb = openpyxl.load_workbook(raw_monte / "immunopeptides.xlsx", read_only=True, data_only=True)
    it = wb["Fig3,5_Canonical_HLA-II_LUAD"].iter_rows(values_only=True)
    head = next(it)
    df = pd.DataFrame(list(it), columns=head)
    wb.close()

    assert len(df) == 52485, f"Unexpected HLA-II row count: {len(df)}"
    df.insert(0, "author_excel_row", np.arange(2, len(df) + 2))
    df["sample_id"] = df["directory"]
    assert set(df["sample_id"]) == set(clin["sample_id"])
    df["peptide"] = df["sequence"].astype(str).str.upper()

    src = pd.DataFrame(
        [
            assess_accessions(d["accession_numbers"], d["accession_number"], mapping, d["redundant_accession_numbers"])
            for d in df[["accession_numbers", "accession_number", "redundant_accession_numbers"]].to_dict("records")
        ]
    )
    for k in src.columns:
        df[k] = src[k]

    df["canonical_amino_acids"] = df["peptide"].map(lambda s: set(s) <= AA_SET)
    df["has_variable_residue_modification"] = df["sequence"].ne(df["peptide"])
    df["has_nterm_modification"] = ~df["nterm"].isin(["Hydrogen", None, "NA", "na"])
    df["has_sequence_tie"] = df["sequenceMulti"].notna() & ~df["sequenceMulti"].isin(["NA", "na", ""])
    df["detected_positive_signal"] = pd.to_numeric(df["totalIntensity"], errors="coerce").gt(0)
    df["primary_eligible"] = (
        df["source_gene_unambiguous"]
        & df["canonical_amino_acids"]
        & (~df["has_variable_residue_modification"])
        & (~df["has_nterm_modification"])
        & df["length"].between(12, 25)
        & (~df["has_sequence_tie"])
    )

    global_genes = df.groupby("peptide")["source_gene_set"].agg(lambda x: set(g for v in x for g in str(v).split("|") if g))
    conflict = set(global_genes[global_genes.map(len) > 1].index)
    out = df[df["primary_eligible"] & df["detected_positive_signal"] & (~df["peptide"].isin(conflict))].copy()
    out["totalIntensity"] = pd.to_numeric(out["totalIntensity"], errors="coerce")
    out = out.sort_values(["totalIntensity", "score", "author_excel_row"], ascending=[False, False, True]).drop_duplicates(["sample_id", "peptide"])
    out["intensity"] = out["totalIntensity"]

    keep_cols = ["sample_id", "peptide", "gene", "intensity", "length", "author_excel_row", "accession_number", "accession_numbers", "source_gene_ids"]
    dest = out_dir / "monte_hlaii_strict.tsv.gz"
    out[keep_cols].sort_values(["sample_id", "peptide"]).to_csv(dest, sep="\t", index=False)
    return {
        "author_rows": int(len(df)),
        "conflict_sequences_excluded": int(len(conflict)),
        "strict_rows": int(len(out)),
        "strict_unique_peptides": int(out["peptide"].nunique()),
        "strict_unique_genes": int(out["gene"].nunique()),
        "counts_by_patient": {k: int(v) for k, v in out["sample_id"].value_counts().sort_index().items()},
        "monte_hlaii_strict_sha256": sha256_file(dest),
    }


def _clean_seq(s) -> str:
    if s is None:
        return ""
    seq = str(s).strip().upper()
    return seq if (len(seq) >= 8 and set(seq) <= AA_SET) else ""


def parse_stopfer2022(stopfer_dir: Path, out_dir: Path) -> dict:
    sd1_specs = [
        ("1", "SKMEL28", "in_vitro_bini", 3.0),
        ("3", "A375", "in_vitro_bini", 3.0),
        ("4", "IPC298", "in_vitro_bini", 3.0),
        ("5", "RPMI7951", "in_vitro_bini", 3.0),
        ("6", "SKMEL2", "in_vitro_bini", 3.0),
        ("7", "SKMEL5", "in_vitro_bini_10nM", 3.0),
        ("8", "SKMEL5", "in_vitro_bini_100nM", 3.0),
        ("10", "SKMEL5", "in_vitro_bini_1uM", 3.0),
        ("11", "SKMEL5", "in_vitro_trametinib", 3.0),
    ]
    records = []
    wb1 = openpyxl.load_workbook(stopfer_dir / "pnas.2208900119.sd01.xlsx", read_only=True, data_only=True)
    for tab, cell_line, exp_grp, t_days in sd1_specs:
        ws = wb1[tab]
        rows = list(ws.iter_rows(values_only=True))
        h_idx = next(i for i, r in enumerate(rows[:12]) if r and any(str(x).strip().lower() in ("sequence", "peptide") for x in r if x is not None))
        hdr = [str(x).strip() if x is not None else "" for x in rows[h_idx]]
        seq_ci = next(i for i, c in enumerate(hdr) if c.lower() in ("sequence", "peptide"))
        gene_ci = next((i for i, c in enumerate(hdr) if c in ("Gene", "GN")), None)
        prot_ci = next((i for i, c in enumerate(hdr) if "protein" in c.lower()), None)
        samp_cols = [
            (ci, cname)
            for ci, cname in enumerate(hdr)
            if ci not in (seq_ci, gene_ci)
            and cname
            and not any(skip in cname.lower() for skip in ("protein", "master", "modif", "ave", "log", "rank", "pval", "p-val", "length", "fdr", "adj"))
            and any(k in cname.lower() for k in ("dmso", "10nm", "100nm", "100 nm", "1um", "1 um", "tram"))
        ]
        for r in rows[h_idx + 1 :]:
            if r is None or len(r) <= seq_ci:
                continue
            pep = _clean_seq(r[seq_ci])
            if not pep:
                continue
            gene = str(r[gene_ci]).strip() if (gene_ci is not None and r[gene_ci] is not None) else ""
            if not gene and prot_ci is not None and prot_ci < len(r) and r[prot_ci] is not None:
                gm = re.search(r"GN=([A-Za-z0-9_-]+)", str(r[prot_ci]))
                if gm:
                    gene = gm.group(1)
            for ci, cname in samp_cols:
                try:
                    fval = float(r[ci])
                except (TypeError, ValueError, IndexError):
                    continue
                if not np.isfinite(fval) or fval <= 0:
                    continue
                clow = cname.lower().replace(" ", "")
                if "dmso" in clow:
                    cond, dose = "DMSO", "0"
                elif "10nm" in clow:
                    cond, dose = ("Trametinib" if "tram" in exp_grp else "Binimetinib"), "10nM"
                elif "100nm" in clow:
                    cond, dose = ("Trametinib" if "tram" in exp_grp else "Binimetinib"), "100nM"
                elif "1um" in clow:
                    cond, dose = ("Trametinib" if "tram" in exp_grp else "Binimetinib"), "1uM"
                else:
                    cond, dose = "Treated", cname
                rep_m = re.search(r"(\d+)$", clow)
                rep = int(rep_m.group(1)) if rep_m else 1
                records.append({
                    "source_file": "sd01", "tab": f"sd01_{tab}", "cell_line": cell_line,
                    "experiment_group": exp_grp, "condition": cond, "dose": dose,
                    "time_days": t_days, "replicate": rep,
                    "sample_id": f"S22_{tab}_{cell_line}_{cond}_{dose}_r{rep}",
                    "peptide": pep, "length": len(pep), "gene": gene, "intensity": fval,
                })
    wb1.close()

    wb2 = openpyxl.load_workbook(stopfer_dir / "pnas.2208900119.sd02.xlsx", read_only=True, data_only=True)
    for sh in wb2.sheetnames:
        rows = list(wb2[sh].iter_rows(values_only=True))
        h_idx = next(i for i, r in enumerate(rows[:12]) if r and any("sequence" in str(x).strip().lower() for x in r if x is not None))
        hdr = [str(x).strip() if x is not None else "" for x in rows[h_idx]]
        seq_ci = next(i for i, c in enumerate(hdr) if "sequence" in c.lower())
        gene_ci = next((i for i, c in enumerate(hdr) if c in ("Gene", "GN")), None)
        prot_ci = next((i for i, c in enumerate(hdr) if "protein" in c.lower()), None)
        cell_line = sh.split("_")[0]
        samp_cols = [
            (ci, cname)
            for ci, cname in enumerate(hdr)
            if ci not in (seq_ci, gene_ci)
            and cname
            and not any(skip in cname.lower() for skip in ("protein", "master", "modif", "ave", "log", "rank", "pval", "length"))
            and any(k in cname.lower() for k in ("control", "bini", "encor"))
        ]
        for r in rows[h_idx + 1 :]:
            if r is None or len(r) <= seq_ci:
                continue
            pep = _clean_seq(r[seq_ci])
            if not pep:
                continue
            gene = str(r[gene_ci]).strip() if (gene_ci is not None and r[gene_ci] is not None) else ""
            if not gene and prot_ci is not None and prot_ci < len(r) and r[prot_ci] is not None:
                gm = re.search(r"GN=([A-Za-z0-9_-]+)", str(r[prot_ci]))
                if gm:
                    gene = gm.group(1)
            for ci, cname in samp_cols:
                try:
                    fval = float(r[ci])
                except (TypeError, ValueError, IndexError):
                    continue
                if not np.isfinite(fval) or fval <= 0:
                    continue
                clow = cname.lower().replace(" ", "")
                day_m = re.match(r"(\d+)d", clow)
                t_days = float(day_m.group(1)) if day_m else 0.0
                cond = "Control" if "control" in clow else ("BiniEncor" if ("bini" in clow and "encor" in clow) else ("Encorafenib" if "encor" in clow else "Binimetinib"))
                rep_m = re.search(r"(\d+)$", clow)
                rep = int(rep_m.group(1)) if rep_m else 1
                records.append({
                    "source_file": "sd02", "tab": f"sd02_{sh}", "cell_line": cell_line,
                    "experiment_group": f"xenograft_{sh}", "condition": cond, "dose": "in_vivo",
                    "time_days": t_days, "replicate": rep,
                    "sample_id": f"S22_CLX_{sh}_{cond}_{int(t_days)}d_r{rep}",
                    "peptide": pep, "length": len(pep), "gene": gene, "intensity": fval,
                })
    wb2.close()

    pep_df = (
        pd.DataFrame(records)
        .groupby(["source_file", "tab", "cell_line", "experiment_group", "condition", "dose", "time_days", "replicate", "sample_id", "peptide", "length"], as_index=False)
        .agg(gene=("gene", "first"), intensity=("intensity", "sum"))
    )
    pep_path = out_dir / "stopfer2022_peptides.tsv.gz"
    pep_df.to_csv(pep_path, sep="\t", index=False)

    wb3 = openpyxl.load_workbook(stopfer_dir / "pnas.2208900119.sd03.xlsx", read_only=True, data_only=True)
    r3 = list(wb3.active.iter_rows(values_only=True))
    wb3.close()
    rna_df = pd.DataFrame([
        {"gene": str(row[1]).strip(), "rna_l2fc_100nM": pd.to_numeric(row[2], errors="coerce"), "rna_l2fc_1uM": pd.to_numeric(row[3], errors="coerce"), "rna_padj_100nM": pd.to_numeric(row[4], errors="coerce"), "rna_padj_1uM": pd.to_numeric(row[5], errors="coerce")}
        for row in r3[2:] if row and row[1] is not None
    ]).dropna(subset=["gene"]).groupby("gene", as_index=False).mean(numeric_only=True)

    wb4 = openpyxl.load_workbook(stopfer_dir / "pnas.2208900119.sd04.xlsx", read_only=True, data_only=True)
    r4 = list(wb4.active.iter_rows(values_only=True))
    wb4.close()
    prot_rows = []
    for row in r4[2:]:
        if row and row[0] is not None:
            d_l2 = [np.log2(v) for v in (pd.to_numeric(row[k], errors="coerce") for k in (2, 3, 4)) if pd.notna(v) and v > 0]
            m_l2 = [np.log2(v) for v in (pd.to_numeric(row[k], errors="coerce") for k in (5, 6, 7)) if pd.notna(v) and v > 0]
            if len(d_l2) >= 2 and len(m_l2) >= 2:
                prot_rows.append({"gene": str(row[0]).strip(), "prot_dmso_l2": float(np.mean(d_l2)), "prot_meki100nM_l2": float(np.mean(m_l2)), "prot_l2fc_100nM": float(np.mean(m_l2) - np.mean(d_l2)), "prot_se_100nM": float(np.sqrt(np.var(d_l2, ddof=1) / len(d_l2) + np.var(m_l2, ddof=1) / len(m_l2)))})
    prot_df = pd.DataFrame(prot_rows).groupby("gene", as_index=False).mean(numeric_only=True)

    wb5 = openpyxl.load_workbook(stopfer_dir / "pnas.2208900119.sd05.xlsx", read_only=True, data_only=True)
    r5 = list(wb5.active.iter_rows(values_only=True))
    wb5.close()
    ub_rows = []
    for row in r5[1:]:
        if row and len(row) >= 10 and row[3] is not None:
            d_l2 = [np.log2(v) for v in (pd.to_numeric(row[k], errors="coerce") for k in (4, 5, 6)) if pd.notna(v) and v > 0]
            m_l2 = [np.log2(v) for v in (pd.to_numeric(row[k], errors="coerce") for k in (7, 8, 9)) if pd.notna(v) and v > 0]
            if len(d_l2) >= 2 and len(m_l2) >= 2:
                ub_rows.append({"gene": str(row[3]).strip(), "ub_pep_l2fc": float(np.mean(m_l2) - np.mean(d_l2))})
    ub_df = pd.DataFrame(ub_rows).groupby("gene", as_index=False).agg(ub_l2fc_100nM=("ub_pep_l2fc", "mean"), ub_sites=("ub_pep_l2fc", "count"))
    sk5_path = out_dir / "stopfer2022_skmel5_multiomics.tsv.gz"
    rna_df.merge(prot_df, on="gene", how="outer").merge(ub_df, on="gene", how="outer").to_csv(sk5_path, sep="\t", index=False)
    return {"peptides_rows": int(len(pep_df)), "peptides_sha256": sha256_file(pep_path), "skmel5_multiomics_sha256": sha256_file(sk5_path)}


def _parse_single_bernhardt_table(args) -> tuple[list[dict], list[dict]]:
    tnum, fp_str, cell_line, drug, t1_h, t2_h = args
    wb = openpyxl.load_workbook(Path(fp_str), read_only=True, data_only=True)
    it = wb[wb.sheetnames[0]].iter_rows(values_only=True)
    _ = next(it)
    r1 = [str(x).strip() if x is not None else "" for x in next(it)]
    cidx = {c: i for i, c in enumerate(r1) if c}
    pep_rows, prot_rows = [], []

    if tnum == 4:
        for row in it:
            if row[0] is not None:
                pep = _clean_seq(row[0])
                if pep:
                    sym = str(row[cidx["Symbol"]]).strip() if row[cidx["Symbol"]] is not None else ""
                    allele = str(row[cidx["HLA allele"]]).strip() if row[cidx["HLA allele"]] is not None else ""
                    annot = str(row[cidx["Annotation"]]).strip() if row[cidx["Annotation"]] is not None else ""
                    for exp_tag, th in [("Ctrl_T1", 24), ("Ctrl_T2", 72), ("Treat_T1", 24), ("Treat_T2", 72)]:
                        int_m = pd.to_numeric(row[cidx[f"Intensity M {exp_tag}"]], errors="coerce")
                        int_h = pd.to_numeric(row[cidx[f"Intensity H {exp_tag}"]], errors="coerce")
                        int_l = pd.to_numeric(row[cidx[f"Intensity L {exp_tag}"]], errors="coerce")
                        rat_hm = pd.to_numeric(row[cidx[f"Ratio H/M normalized {exp_tag}"]], errors="coerce")
                        rcnt = pd.to_numeric(row[cidx[f"Ratio H/M count {exp_tag}"]], errors="coerce")
                        if (pd.notna(int_m) and int_m > 0) or (pd.notna(int_h) and int_h > 0) or (pd.notna(rat_hm) and rat_hm > 0):
                            l2_hm = float(np.log2(rat_hm)) if (pd.notna(rat_hm) and rat_hm > 0) else np.nan
                            m_val = float(int_m) if (pd.notna(int_m) and int_m > 0) else 0.0
                            h_val = float(int_h) if (pd.notna(int_h) and int_h > 0) else 0.0
                            # Table 4 label assignment: M = IFNg pulse, H = control pulse
                            pep_rows.append({
                                "table": f"Table{tnum}", "cell_line": cell_line, "intervention": drug,
                                "experiment_tag": exp_tag, "is_treated": int(exp_tag.startswith("Treat")),
                                "time_h": th, "peptide": pep, "length": len(pep), "gene": sym,
                                "hla_allele": allele, "category": annot, "intensity_M": m_val, "intensity_H": h_val,
                                "intensity_trt_pulse": m_val if exp_tag.startswith("Treat") else h_val,
                                "intensity_ref_pulse": h_val if exp_tag.startswith("Treat") else m_val,
                                "intensity_total": float(np.nansum([int_l, int_m, int_h])),
                                "norm_l2_hm": l2_hm,
                                "l2fc_trt_vs_ref": -l2_hm if (exp_tag.startswith("Treat") and pd.notna(l2_hm)) else l2_hm,
                                "ratio_reliable": int(pd.notna(rcnt) and rcnt >= 1),
                            })
            if len(row) > 128 and row[127] is not None:
                gm = re.search(r"GN=([A-Za-z0-9_-]+)", str(row[128] or ""))
                if gm:
                    l2_24 = pd.to_numeric(row[cidx["norm.med.log2.Ratio.H.M.normalized.IFNg_24h"]], errors="coerce")
                    l2_72 = pd.to_numeric(row[cidx["norm.med.log2.Ratio.H.M.normalized.IFNg_72h"]], errors="coerce")
                    prot_rows.append({
                        "gene": gm.group(1),
                        "prot_norm_l2_hm_24h": float(l2_24) if pd.notna(l2_24) else np.nan,
                        "prot_norm_l2_hm_72h": float(l2_72) if pd.notna(l2_72) else np.nan,
                        "prot_l2fc_ifng_24h": float(-l2_24) if pd.notna(l2_24) else np.nan,
                        "prot_l2fc_ifng_72h": float(-l2_72) if pd.notna(l2_72) else np.nan,
                    })
        wb.close()
        return pep_rows, prot_rows

    col_pep = cidx.get("Peptide.no.label", 0)
    seen = {}
    for row in it:
        pep = _clean_seq(row[col_pep] if col_pep < len(row) and row[col_pep] is not None else row[0])
        if not pep:
            continue
        exp_tag = str(row[cidx["Experiment_2"]] or "").strip()
        if exp_tag not in ("Ctrl_T1", "Ctrl_T2", "Treat_T1", "Treat_T2"):
            continue
        m_val = float(pd.to_numeric(row[cidx["summed_Intensity_M_over_Experiment"]], errors="coerce") or 0.0)
        h_val = float(pd.to_numeric(row[cidx["summed_Intensity_H_over_Experiment"]], errors="coerce") or 0.0)
        t_val = float(pd.to_numeric(row[cidx["summed_total_intensity"]], errors="coerce") or 0.0)
        norm_col = cidx.get(f"norm_Median_lg2_Ratio_H/M_corrected_{exp_tag}")
        l2_hm = pd.to_numeric(row[norm_col], errors="coerce") if norm_col is not None else np.nan
        rel_col = cidx.get("Treat_Ratio_Reliability" if exp_tag.startswith("Treat") else "Ctrl_Ratio_Reliability")
        is_rel = int(str(row[rel_col] or "").strip().lower() in ("reliable", "corrected")) if rel_col is not None else 0
        key = (exp_tag, pep)
        if key in seen:
            prev = seen[key]
            prev["intensity_M"] = max(prev["intensity_M"], m_val)
            prev["intensity_H"] = max(prev["intensity_H"], h_val)
            prev["intensity_trt_pulse"] = max(prev["intensity_trt_pulse"], h_val)
            prev["intensity_ref_pulse"] = max(prev["intensity_ref_pulse"], m_val)
            prev["intensity_total"] = max(prev["intensity_total"], t_val)
            prev["ratio_reliable"] = max(prev["ratio_reliable"], is_rel)
            if np.isnan(prev["norm_l2_hm"]) and pd.notna(l2_hm):
                prev["norm_l2_hm"] = prev["l2fc_trt_vs_ref"] = float(l2_hm)
            continue
        col_hla = cidx.get("filtered.HLA_allele", cidx.get("HLA_allele"))
        col_cat = cidx.get("Category")
        seen[key] = {
            "table": f"Table{tnum}", "cell_line": cell_line, "intervention": drug,
            "experiment_tag": exp_tag, "is_treated": int(exp_tag.startswith("Treat")),
            "time_h": t1_h if exp_tag.endswith("T1") else t2_h, "peptide": pep, "length": len(pep),
            "gene": str(row[cidx["Symbol"]] or "").strip(),
            "hla_allele": str(row[col_hla] or "").strip() if col_hla is not None else "",
            "category": str(row[col_cat] or "").strip() if col_cat is not None else "",
            "intensity_M": m_val, "intensity_H": h_val,
            "intensity_trt_pulse": h_val, "intensity_ref_pulse": m_val,
            "intensity_total": t_val, "norm_l2_hm": float(l2_hm) if pd.notna(l2_hm) else np.nan,
            "l2fc_trt_vs_ref": float(l2_hm) if pd.notna(l2_hm) else np.nan,
            "ratio_reliable": is_rel,
        }
    wb.close()
    return list(seen.values()), prot_rows


def parse_bernhardt2025(bernhardt_dir: Path, out_dir: Path) -> dict:
    specs = [
        (4, str(bernhardt_dir / "Table4.xlsx"), "MaMel63a", "IFNg", 24, 72),
        (7, str(bernhardt_dir / "Table7.xlsx"), "MaMel63a", "Trametinib", 24, 72),
        (8, str(bernhardt_dir / "Table8.xlsx"), "UKEMel105b", "Trametinib", 24, 72),
        (9, str(bernhardt_dir / "Table9.xlsx"), "MaMel63a", "VemuCobi", 24, 48),
        (10, str(bernhardt_dir / "Table10.xlsx"), "SKMel28_wt", "VemuCobi", 24, 72),
        (11, str(bernhardt_dir / "Table11.xlsx"), "SKMel28_dr", "VemuCobi", 24, 72),
        (12, str(bernhardt_dir / "Table12.xlsx"), "SKMel28_wtVSdr", "Resistance_dr_vs_wt", 24, 72),
    ]
    all_pep, all_prot = [], []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for p_rows, pr_rows in ex.map(_parse_single_bernhardt_table, specs):
            all_pep.extend(p_rows)
            all_prot.extend(pr_rows)
    pep_df = (
        pd.DataFrame(all_pep)
        .groupby(["table", "cell_line", "intervention", "experiment_tag", "is_treated", "time_h", "peptide", "length"], as_index=False)
        .agg(gene=("gene", "first"), hla_allele=("hla_allele", "first"), category=("category", "first"), intensity_M=("intensity_M", "max"), intensity_H=("intensity_H", "max"), intensity_trt_pulse=("intensity_trt_pulse", "max"), intensity_ref_pulse=("intensity_ref_pulse", "max"), intensity_total=("intensity_total", "max"), norm_l2_hm=("norm_l2_hm", "median"), l2fc_trt_vs_ref=("l2fc_trt_vs_ref", "median"), ratio_reliable=("ratio_reliable", "max"))
    )
    pep_df["sample_id"] = pep_df["table"] + "_" + pep_df["cell_line"] + "_" + pep_df["intervention"] + "_" + pep_df["experiment_tag"]
    pep_path = out_dir / "bernhardt2025_psilac_peptides.tsv.gz"
    pep_df.to_csv(pep_path, sep="\t", index=False)

    prot_df = pd.DataFrame(all_prot).groupby("gene", as_index=False).median(numeric_only=True)
    prot_path = out_dir / "bernhardt2025_mamel63a_ifng_proteome.tsv.gz"
    prot_df.to_csv(prot_path, sep="\t", index=False)

    wb5 = openpyxl.load_workbook(bernhardt_dir / "Table5.xlsx", read_only=True, data_only=True)
    mda_df = pd.DataFrame([{"gene": str(r[0]).strip(), "antigen_category": str(r[1]).strip()} for i, r in enumerate(wb5.active.iter_rows(values_only=True)) if i > 0 and r and r[0] is not None])
    wb5.close()
    mda_df.to_csv(out_dir / "bernhardt2025_mda_genes.tsv", sep="\t", index=False)
    return {"peptides_rows": int(len(pep_df)), "peptides_sha256": sha256_file(pep_path), "mamel63a_proteome_sha256": sha256_file(prot_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = resolve_config(data_dir=args.data_dir, processed_dir=args.processed_dir)
    ann_dir = cfg.annotations()
    fup_dir = cfg.followup()
    ann_dir.mkdir(parents=True, exist_ok=True)
    fup_dir.mkdir(parents=True, exist_ok=True)

    topo_meta = stage_uniprot_topology(cfg, ann_dir)
    hlaii_meta = stage_monte_hlaii(cfg, ann_dir)
    (ann_dir / "manifest.json").write_text(json.dumps({"topology": topo_meta, "monte_hlaii": hlaii_meta}, indent=2) + "\n")

    raw_fup = cfg.raw("followup")
    if (raw_fup / "stopfer2022").exists() and (raw_fup / "bernhardt2025").exists():
        st_meta = parse_stopfer2022(raw_fup / "stopfer2022", fup_dir)
        bn_meta = parse_bernhardt2025(raw_fup / "bernhardt2025", fup_dir)
        (fup_dir / "audit_manifest.json").write_text(json.dumps({"stopfer2022": st_meta, "bernhardt2025": bn_meta}, indent=2) + "\n")
    else:
        print("[note] Follow-up raw Excel supplements not found under data/raw/followup; skipping raw follow-up rebuild.")

    print("Staging complete:", ann_dir.name, fup_dir.name)


if __name__ == "__main__":
    main()
