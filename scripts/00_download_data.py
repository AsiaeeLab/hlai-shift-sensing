#!/usr/bin/env python3
"""Download public ProteomeXchange, MassIVE, GEO, Springer, Europe PMC, and UniProt inputs.

Usage:
    python3 scripts/00_download_data.py --data-dir ./data
    python3 scripts/00_download_data.py --data-dir ./data --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from presentation_state.config import resolve_config  # noqa: E402

UA = {"User-Agent": "presentation-state-deposit/1.0"}

DIRECT_DOWNLOADS = [
    # 1. Stopfer CDK4/6i melanoma (PXD017407, GSE144373)
    ("raw/stopfer/41467_2020_16588_MOESM3_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16588-9/MediaObjects/41467_2020_16588_MOESM3_ESM.xlsx"),
    ("raw/stopfer/41467_2020_16588_MOESM5_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16588-9/MediaObjects/41467_2020_16588_MOESM5_ESM.xlsx"),
    ("raw/stopfer/41467_2020_16588_MOESM10_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16588-9/MediaObjects/41467_2020_16588_MOESM10_ESM.xlsx"),
    ("raw/stopfer/GSE144373_lstopfer_l2tpm.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144373/suppl/GSE144373_lstopfer_l2tpm.txt.gz"),
    # 2. Newey colorectal organoids (PXD014017, PXD031634)
    ("raw/newey2023/supplementaryFiles.zip", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9999000/supplementaryFiles"),
    # 3. Velz medulloblastoma (PXD043770, GSE238195)
    ("raw/velz2025/41467_2025_56268_MOESM9_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-56268-0/MediaObjects/41467_2025_56268_MOESM9_ESM.xlsx"),
    ("raw/velz2025/41467_2025_56268_MOESM10_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-56268-0/MediaObjects/41467_2025_56268_MOESM10_ESM.xlsx"),
    ("raw/velz2025/41467_2025_56268_MOESM12_ESM.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-56268-0/MediaObjects/41467_2025_56268_MOESM12_ESM.xlsx"),
    ("raw/velz2025/GSE238195_tpm_gene_counts.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE238nnn/GSE238195/suppl/GSE238195_tpm_gene_counts.tsv.gz"),
    # 4. Abelin MONTE LUAD GENCODE v34 reference annotations
    ("raw/visibility_monte/gencode.v34.pc_translations.fa.gz", "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_34/gencode.v34.pc_translations.fa.gz"),
    ("raw/visibility_monte/gencode.v34.metadata.HGNC.gz", "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_34/gencode.v34.metadata.HGNC.gz"),
    # 5. Stopfer 2022 MEKi melanoma & Bernhardt 2025 pSILAC supplements
    ("raw/followup/PMC9894220_supp.zip", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9894220/supplementaryFiles"),
    ("raw/followup/PMC11744270_supp.zip", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11744270/supplementaryFiles"),
    ("raw/followup/stopfer2022/GSE185284_lstopfer_l2tpm.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185284/suppl/GSE185284_lstopfer_l2tpm.txt.gz"),
    ("raw/followup/stopfer2022/Filemap_MEKi_MHC_Pride.xlsx", "https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/03/PXD029860/Filemap_MEKi_MHC_Pride.xlsx"),
    ("raw/followup/bernhardt2025/sample_info.xlsx", "https://ftp.pride.ebi.ac.uk/pride/data/archive/2025/05/PXD054097/sample_info.xlsx"),
    # 6. UniProt reviewed human proteome topology (9606)
    (
        "raw/annotations/uniprot_human_topology.tsv.gz",
        "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Cgene_primary%2Clength%2Cft_signal%2Cft_transmem%2Ccc_subcellular_location&format=tsv&query=%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29",
    ),
]

PRIDE_ARCHIVES = {
    "PXD014017": ("raw/newey2023/original2019_maxquant", {"peptides.txt"}),
    "PXD065279": ("raw/visibility_mhc1tip", set()),
    "PXD006939": ("raw/visibility_replication/chong2018", set()),
    "PXD056426": ("raw/visibility_hap1", {"Search.zip", "Raw_file_list.xlsx"}),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=240) as resp, open(tmp, "wb") as fw:
        shutil.copyfileobj(resp, fw, length=1 << 20)
    tmp.replace(dest)
    print(f"[get ] {dest.name} ({dest.stat().st_size:,} bytes)")


def fetch_pride_files(acc: str, dest_dir: Path, wanted_names: set[str]) -> None:
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{acc}/files"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        files = json.loads(resp.read().decode("utf-8"))
    for entry in files:
        fname = entry.get("fileName") or ""
        if wanted_names and fname not in wanted_names:
            continue
        if not wanted_names and not (fname.endswith(".tsv") or fname.endswith(".txt") or fname.endswith(".xlsx") or fname.endswith(".gmt")):
            continue
        locs = entry.get("publicFileLocations") or []
        dl = next((l["value"] for l in locs if str(l.get("value", "")).startswith("https://")), None)
        if not dl:
            ftp = next((l["value"] for l in locs if str(l.get("value", "")).startswith("ftp://")), None)
            if ftp:
                dl = ftp.replace("ftp://", "https://", 1)
        if dl:
            fetch_url(dl, dest_dir / fname)


def verify_staged_inventory(cfg) -> None:
    required = [
        cfg.processed("velz2025", "RNA_gene_tpm.tsv.gz"),
        cfg.processed("newey2023", "peptides_long.tsv.gz"),
        cfg.processed("visibility_monte", "peptides_global_source_strict.tsv.gz"),
        cfg.processed("visibility_mhc1tip_gene_unique", "pdo", "peptides.tsv.gz"),
        cfg.processed("visibility_replication", "chong2018_gene_unique", "peptides.tsv.gz"),
        cfg.processed("visibility_hap1_gene_unique", "peptides.tsv.gz"),
    ]
    present = {str(p): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in required if p.exists()}
    print(f"Verified {len(present)}/{len(required)} core cohort processed tables.")
    if len(present) < len(required):
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(f"Missing required staged inputs: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(data_dir=args.data_dir, processed_dir=args.processed_dir)
    if not args.verify_only:
        for rel, url in DIRECT_DOWNLOADS:
            try:
                fetch_url(url, cfg.data_dir / rel)
            except Exception as exc:
                print(f"[warn] Could not fetch {rel}: {exc}")
        for pmcid, subdir in [("PMC9894220_supp.zip", "stopfer2022"), ("PMC11744270_supp.zip", "bernhardt2025")]:
            zp = cfg.raw("followup", pmcid)
            if zp.exists():
                with zipfile.ZipFile(zp, "r") as zf:
                    for zi in zf.infolist():
                        if zi.filename.endswith((".xlsx", ".pdf")):
                            target = cfg.raw("followup", subdir, zi.filename)
                            if not target.exists():
                                zf.extract(zi, cfg.raw("followup", subdir))
        for acc, (rel_dir, wanted) in PRIDE_ARCHIVES.items():
            try:
                fetch_pride_files(acc, cfg.data_dir / rel_dir, wanted)
            except Exception as exc:
                print(f"[warn] PRIDE {acc}: {exc}")
    verify_staged_inventory(cfg)


if __name__ == "__main__":
    main()
