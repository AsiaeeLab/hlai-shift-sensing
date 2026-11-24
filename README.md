# HLA-I immunopeptidome shifts reveal shared antigen-processing responses across human genotypes

**Authors:** Amir Asiaee<sup>1,*</sup>, Nebojša Jojić<sup>2</sup>, Elizabeth J. Phillips<sup>3,4</sup>, and Simon A. Mallal<sup>3,4,*</sup>  
<sup>1</sup>Department of Biostatistics, Vanderbilt University Medical Center, Nashville, TN, USA  
<sup>2</sup>Microsoft Research, Redmond, WA, USA  
<sup>3</sup>Department of Medicine, Vanderbilt University Medical Center, Nashville, TN, USA  
<sup>4</sup>Institute for Immunology and Infectious Diseases, Murdoch University, Perth, WA, Australia  
<sup>*</sup>Co-corresponding authors; Lead contact: `amir.asiaeetaheri@vumc.org`

---

## Overview

Human nucleated cells display thousands of endogenous peptides on HLA class I (`HLA-A`, `HLA-B`, `HLA-C`) molecules, yet individual peptide shifts decouple from parent-protein abundance because proteasomal substrates draw heavily on newly synthesized and polyubiquitylated proteins. Integrating eleven public human datasets spanning primary tumors, patient-derived organoids, melanoma cell lines, human pancreatic beta cells, and two CRISPR antigen-processing knockout atlases, this standalone executable deposit reproduces every analysis, Kitagawa–Oaxaca decomposition, external frozen-decoder test, publication figure, summary CSV table, and programmatic invariant check.

Key quantitative findings reproduced by this codebase:
1. **Concentration decoupling & paired baselines:** Across 8 proteome-matched perturbations, parent-protein fold changes explain only `0.4%` to `15.6%` (`1.2%` to `17.8%` under split-half reliability sensitivity) of surviving peptide variance, while IFN-gamma splits sibling peptides from the same protein by `>2-fold` (`26.7%` to `66.7%`; `2.5x` to `13.2x` above multiplicity-matched replicate noise nulls). Baseline-free single-biopsy decoding fails across primary medulloblastomas (`Velz` Coarse96-vs-RNA `ARI = 0.030`, Coarse96-vs-methylation `ARI = 0.105` vs. RNA-vs-methylation `ARI = 0.654`; proliferation MSE ratios `0.991` to `1.008`) and same-era colorectal organoids (`LOO BA = 0.403–0.594`), whereas paired ordering reaches `0.984` (`Coarse96`) and `1.000` (`AA20`).
2. **Cross-lineage & cross-study transfer + exact Kitagawa–Oaxaca decomposition:** Across 35 paired contrasts (`16` human lineages, `5` studies), full 9-mer positional marginals (`PosAA180`) classify IFN-gamma vs. MAPK inhibition at `LOLO AUROC = 0.948` (`0.916` under `I=L` harmonization) and strict study-plus-lineage-purged `LOSO AUROC = 0.756` (**`0.788` under `I=L` harmonization**, exceeding source-gene shares `0.776` and anchor-blind `Coarse96` `0.448`; `66` affine dimensions, Proposition 1). Exact symmetric Kitagawa–Oaxaca decomposition ($\Delta q = \sum_\ell \Delta w_\ell \bar{\mu}_\ell + \sum_\ell \bar{w}_\ell \Delta\mu_\ell$, retaining unassigned mass explicitly) projected onto frozen outer-fold `PosAA180 (I=L)` decoders shows `51.8%` between-locus / `48.2%` within-locus contribution in `LOLO`, and **`76.8%` within-locus motif remodeling** (`AUROC = 0.796`) vs. `23.2%` between-locus allocation (`AUROC = 0.780`) in strict cross-study `LOSO` (standalone `2-df` conditional `HLA-A/B/C` `LOSO = 0.924` vs. `1-df` coverage `0.168`). Trained solely on `Ma-Mel-63a` and `UKE-Mel-105b`, `PosAA180` separates sensitive (`wt`) from double-resistant (`dr`) `SK-Mel-28` across zero shared 4-digit HLA alleles (`AUROC = 1.000`, `exact P = 0.0357`, time-matched mock-adjusted DiD `+9.860` at 24 h and `+7.878` at 72 h, holding at `1.000` after purging all `877` `I=L`-shared training 9-mers).
3. **External experimental bridges, six-omic tumor flux, gatekeepers & persistence:** Evaluating the **frozen** 35-contrast `PosAA180 (I=L)` decoder with zero refitting shows that Type-I interferon (`IFN-alpha`) in `ECN90` human pancreatic beta cells (`Carré et al. 2025`, `PXD045265`) expands `HLA-B` share by `+26.4` to `+30.3` percentage points and scores strongly positive (`+4.857` replicate-frequency, `+4.465` presence, `+0.963` precursor intensity), while across isogenic `HAP1` `WT`, `ERAP1`-KO, `TAP1`-KO, and `TAP2`-KO cells `+/- IFN-gamma` (`Aranha et al. 2026`, `PXD085470` + `PXD085506`), all four genotypes induce positive within-locus motif remodeling (`+0.514` to `+6.407`) and IFN-gamma rescues repertoire breadth in `TAP1/2`-KO (`+3.154` and `+9.062`). Six-omic `Abelin MONTE` LUADs link `K-epsilon-GG` ubiquitylation to HLA-I yield (`+0.199` lysine-adjusted; within-gene `noIP +0.046`, two-way CGM `+0.056`) and endogenous `ERAP1` to shorter ligands within 16 shared 4-digit alleles (`r = -0.745`, exact `10!` whole-patient permutation `P = 0.00233`). Curated `UniProt FT_SIGNAL` spans retain `89.2%` (`TAP1`-KO) and `83.3%` (`TAP2`-KO) vs. `18.9%` for mature-chain peptides on the same proteins. Across `N = 6,998` baseline ligands, floor-imputed within-cohort Ridge reaches `LOBO AUROC = 0.725` for retention and `0.872` for high-occupancy persistence (`Predicted HLA affinity = 0.635` on fold-change durability).

4. **Positional anatomy, 9x19 difference motifs & multi-allele supertype averaging:** Single-position and leave-one-position-out ablations across `P1–P9` show that isolating `Anchors (P2, P9)` drops from `0.948` LOLO to `0.560` strict LOSO and `0.667` on `SK-Mel-28`, while `Bulge (P4–P7)` reaches `0.556` LOSO and `Pooled-AA` reaches `0.396`; only `Full (P1–P9)` sustains transfer across all three benchmarks (`0.916` LOLO, `0.788` LOSO, `1.000` `SK-Mel-28`, where dropping N-terminal `P1` collapses `SK-Mel-28` from `1.000` to `0.167`). Across all 16 benchmark donors, averaging up to six expressed class I alleles contracts baseline pairwise anchor Jensen–Shannon divergence by `-39.8%` (`0.599` bits across `703` single four-digit allele pairs to `0.360` bits across `66` donor pairs sharing zero four-digit alleles, `95.5%` sharing at least one Sidney–Sette HLA supertype).

---

## Public Datasets

| Narrative Cohort Name | Public Accessions | Primary Study | Biological System & Modalities |
|---|---|---|---|
| **Stopfer CDK4/6i melanoma** | `PXD017407`, `GSE144373` | Stopfer et al. (2020) | 4 melanoma lines (`SKMEL5/28/2`, `IPC298`), palbociclib vs. DMSO HipMHC TMT + RNA-seq |
| **Newey colorectal organoid** | `PXD014017`, `PXD031634` | Newey et al. (2019; 2023) | Colorectal PDOs (`CRC01/04/05/08`), 2018 IFN-gamma vs. trametinib + proteome/RNA |
| **Velz medulloblastoma** | `PXD043770`, `GSE238195`, `GSE250586` | Velz et al. (2025) | `N=28` primary medulloblastomas (HLA-I, bulk RNA-seq, 850k methylation) |
| **Bathini MHC1-TIP** | `PXD065279` | Bathini et al. (2026) | Colorectal PDOs (`ITO57T`, `ITO58M`, `ITO66T` +/- IFN-gamma) & `A375` (+/- TGF-beta) |
| **Chong ovarian** | `PXD006939` | Chong et al. (2018) | Ovarian carcinoma `UWB1.289` +/- IFN-gamma (`n=4` bio. reps/group, HLA-I + proteome) |
| **Shapiro HAP1 knockout** | `PXD056426` | Shapiro et al. (2025) | Near-haploid `HAP1` WT + 11 CRISPR APPM knockouts (HLA-I + whole proteome) |
| **Abelin MONTE LUAD** | `MSV000090437` | Abelin / MONTE (2023) | `N=10` 6-omic primary lung adenocarcinomas (`FT` + independent `noIP` aliquots) |
| **Stopfer MEKi melanoma** | `PXD029860`, `GSE185284` | Stopfer et al. (2022) | 6 melanoma lines TMT MEKi +/- BRAFi, xenograft time-course (Days 1–5), `SKMEL5` Ub |
| **Bernhardt pulsed-SILAC** | `PXD054097` | Bernhardt et al. (2025) | `Ma-Mel-63a`, `UKE-Mel-105b`, sensitive/resistant `SK-Mel-28 wt/dr` pSILAC (28 entries) |
| **Aranha HAP1 APPM-KO x IFNg** | `PXD085470`, `PXD085506` | Aranha et al. (2026) | Isogenic `HAP1` `WT`, `ERAP1`-KO, `TAP1`-KO, `TAP2`-KO +/- IFN-gamma (`n=3`, HLA-I + DIA proteome) |
| **Carré human beta-cell IFNa** | `PXD045265`, `PXD045211` | Carré et al. (2025) | `ECN90` pancreatic beta cells +/- IFN-alpha (`n=4`) & primary human islets |

---

## Repository Structure

```text
hlai-shift-sensing/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── Makefile
├── run_all.sh
├── verify_salvage.py                  # Full 56-check manuscript + numerical verifier
├── src/
│   └── presentation_state/
│       ├── __init__.py
│       ├── config.py                  # Portable DATA_DIR, PROCESSED_DIR, RESULTS_DIR, FIG_DIR resolver
│       ├── representations.py         # Bulge-4c/Coarse96 (66 affine dims), Pooled-AA, Full (P1-P9)
│       ├── decomposition.py           # Exact symmetric Kitagawa-Oaxaca mixture decomposition
│       └── manifest_validator.py      # LOLO & strict study+lineage-purged fold manifest validator
├── tests/
│   └── test_presentation_state.py     # Unit tests for Proposition 1, Kitagawa-Oaxaca, I=L, & manifests
├── data/processed/                    # Bundled 35-contrast 9-mer weights & external processed peptide tables
├── results/                           # Bundled module outputs, 35-row roster, fold manifests, & 31 CSV tables
├── figures/                           # All 14 vector PDF + PNG + SVG publication figures (7 main, 6 supp, GA)
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_preprocess.py
│   ├── 02_analyze_paired_shifts.py    # Runs Modules 1-5 and 7-9 (Kitagawa-Oaxaca, bridge, positional anatomy)
│   ├── 03_falsification_controls.py
│   ├── 04_make_figures_tables.py      # Generates all 14 figures & 31 summary CSV/TSV tables
│   ├── 05_verify_results.py           # Runs 47 standalone programmatic invariant checks (100% pass)
│   └── modules/
└── notebooks/
    ├── 00_index.{Rmd,html}
    ├── 01_data_loading_and_qc.{Rmd,html}
    ├── 02_presentation_state_model.{Rmd,html}
    ├── 03_cross_lineage_and_multi_omic_cohorts.{Rmd,html}
    ├── 04_censoring_gatekeepers_and_sensitivity.{Rmd,html}
    └── 05_antigen_persistence_and_figures.{Rmd,html}
```

---

## Quick Verification & Step-by-Step Reproduction

```bash
# Run unit tests and standalone 47-check numerical/algebraic/manifest verifier directly on bundled tables
python3 -m unittest discover -s tests -v
python3 scripts/05_verify_results.py

# Regenerate all 14 publication PDFs/PNGs and all 31 summary CSV tables from bundled results
python3 scripts/04_make_figures_tables.py --results-dir ./results --fig-dir ./figures

# Re-knit all 6 self-contained R Markdown reports
make report
```
