"""Fixed, outcome-independent summaries of measured human HLA class I ligands.

Implements the six-tier sequence representation hierarchy and Proposition 1 algebraic
affine-dimension check (66 degrees of freedom in 96-dimensional Coarse96 P4-P7 features).
"""

from __future__ import annotations

from itertools import combinations, product
import numpy as np
from scipy import sparse

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GROUPS = ("AILMVFWYP", "CNQSTG", "DE", "RHK")
CLASS_NAMES = ("hydrophobic", "polar", "acidic", "basic")
CLASS = {aa: k for k, letters in enumerate(GROUPS) for aa in letters}
PAIRS = tuple(combinations((3, 4, 5, 6), 2))  # zero-based P4-P7


def sequence_map(peptides, kind="coarse96", seed=0):
    """Map canonical 9-mer peptides to fixed chemical or positional feature vectors."""
    mapping = CLASS
    if kind.startswith("random96"):
        rng = np.random.default_rng(seed)
        aa = rng.permutation(list(AMINO_ACIDS))
        sizes = np.cumsum([0] + [len(g) for g in GROUPS])
        mapping = {a: k for k in range(4) for a in aa[sizes[k] : sizes[k + 1]]}
    pairs = PAIRS
    if kind == "anchor16":
        pairs = ((1, 8),)
    elif kind == "flank96":
        pairs = tuple(combinations((0, 1, 7, 8), 2))
    if kind in ("coarse96", "anchor16", "flank96") or kind.startswith("random96"):
        mat = np.zeros((len(peptides), len(pairs) * 16), dtype=float)
        categories = tuple(f"group{k}" for k in range(4)) if kind.startswith("random96") else CLASS_NAMES
        names = [f"P{i + 1}P{j + 1}_{a}_{b}" for i, j in pairs for a, b in product(categories, repeat=2)]
        for row, p in enumerate(peptides):
            if len(p) != 9 or any(a not in CLASS for a in p):
                raise ValueError(f"Expected canonical nine-residue peptide: {p}")
            for k, (i, j) in enumerate(pairs):
                mat[row, 16 * k + 4 * mapping[p[i]] + mapping[p[j]]] = 1.0
        return mat, names
    if kind == "aa20":
        mat = np.array([[p.count(a) / len(p) for a in AMINO_ACIDS] for p in peptides])
        return mat, list(AMINO_ACIDS)
    if kind == "position80":
        mat = np.zeros((len(peptides), 80))
        for row, p in enumerate(peptides):
            for k, i in enumerate((3, 4, 5, 6)):
                mat[row, 20 * k + AMINO_ACIDS.index(p[i])] = 1.0
        return mat, [f"P{i}_{a}" for i in (4, 5, 6, 7) for a in AMINO_ACIDS]
    raise ValueError(kind)


def relative_signal(signals):
    signals = np.asarray(signals, dtype=float)
    if not np.isfinite(signals).all() or (signals < 0).any():
        raise ValueError("Signals must be finite, nonnegative.")
    totals = signals.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Sample has no eligible measured signal.")
    return signals / totals


def protein_map(gene_lists):
    groups = [sorted(set(g)) or ["UNMAPPED"] for g in gene_lists]
    genes = sorted({gene for group in groups for gene in group})
    index = {g: i for i, g in enumerate(genes)}
    rows, cols, vals = [], [], []
    for i, group in enumerate(groups):
        for gene in group:
            rows.append(i)
            cols.append(index[gene])
            vals.append(1.0 / len(group))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), len(genes))), genes


def construct(signals, peptides, gene_lists, random_seeds=()):
    pi = relative_signal(signals)
    presence = relative_signal(np.asarray(signals) > 0)
    matrices = {"peptide": pi}
    names = {"peptide": list(peptides)}
    m, genes = protein_map(gene_lists)
    matrices["protein"] = np.asarray(pi @ m)
    names["protein"] = genes
    for kind in ("coarse96", "aa20", "anchor16", "flank96", "position80"):
        phi, labels = sequence_map(peptides, kind)
        matrices[kind] = pi @ phi
        names[kind] = labels
    phi, labels = sequence_map(peptides, "coarse96")
    matrices["count96"] = presence @ phi
    names["count96"] = labels
    for seed in random_seeds:
        kind = f"random96_{seed}"
        phi, labels = sequence_map(peptides, kind, seed=seed)
        matrices[kind] = pi @ phi
        names[kind] = labels
    return matrices, names


def geometry_check():
    words = list(product(range(4), repeat=4))
    pair_idx = tuple(combinations(range(4), 2))
    encoding = np.zeros((256, 96))
    for i, word in enumerate(words):
        for k, (a, b) in enumerate(pair_idx):
            encoding[i, 16 * k + 4 * word[a] + word[b]] = 1.0
    return int(np.linalg.matrix_rank(encoding)), int(np.linalg.matrix_rank(encoding - encoding.mean(0)))
