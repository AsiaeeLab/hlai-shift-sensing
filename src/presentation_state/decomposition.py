"""Exact symmetric Kitagawa-Oaxaca mixture decomposition and frozen-decoder projection.

Separates paired repertoire composition shifts into between-group mass allocation
and within-group conditional motif shifts while retaining unassigned mass explicitly
and isolating groups absent at either baseline or perturbation.
"""

from __future__ import annotations

import numpy as np


def decompose_kitagawa_oaxaca(
    w0: np.ndarray,
    w1: np.ndarray,
    feature_mass0: np.ndarray,
    feature_mass1: np.ndarray,
    tolerance: float = 1e-10,
) -> dict:
    """Decompose paired weighted feature sums a_t = w_t * mu_t across L groups.

    For groups observed at both times (w0 > tolerance and w1 > tolerance):
        Delta q = sum_l Delta w_l * (mu_l1 + mu_l0) / 2
                + sum_l (w_l1 + w_l0) / 2 * Delta mu_l
                + unresolved_emergence_or_loss.
    """
    w0, w1, a0, a1 = [np.asarray(x, dtype=float) for x in (w0, w1, feature_mass0, feature_mass1)]
    if w0.ndim != 1 or w1.shape != w0.shape or a0.ndim != 2 or a1.shape != a0.shape or a0.shape[0] != len(w0):
        raise ValueError("Expected group weights (L,) and feature masses (L, D)")
    if any(not np.isfinite(arr).all() for arr in (w0, w1, a0, a1)):
        raise ValueError("All inputs must be finite")
    if any(np.min(arr) < -tolerance for arr in (w0, w1, a0, a1)):
        raise ValueError("Group weights and feature masses must be nonnegative")
    if not (np.isclose(w0.sum(), 1.0, atol=tolerance, rtol=0) and np.isclose(w1.sum(), 1.0, atol=tolerance, rtol=0)):
        raise ValueError("Retain all categories including unassigned; weights must sum to one")
    for w, a in ((w0, a0), (w1, a1)):
        if np.any(np.abs(a[w <= tolerance]) > tolerance):
            raise ValueError("A zero-mass group cannot carry nonzero feature mass")

    L, D = a0.shape
    common = (w0 > tolerance) & (w1 > tolerance)
    between_by_group = np.zeros((L, D), dtype=float)
    within_by_group = np.zeros((L, D), dtype=float)
    emerg_by_group = np.zeros((L, D), dtype=float)
    for g in range(L):
        if common[g]:
            mu0 = a0[g] / w0[g]
            mu1 = a1[g] / w1[g]
            between_by_group[g] = (w1[g] - w0[g]) * 0.5 * (mu1 + mu0)
            within_by_group[g] = 0.5 * (w1[g] + w0[g]) * (mu1 - mu0)
        else:
            emerg_by_group[g] = a1[g] - a0[g]

    between = between_by_group.sum(axis=0)
    within = within_by_group.sum(axis=0)
    unresolved = emerg_by_group.sum(axis=0)
    total = (a1 - a0).sum(axis=0)
    err = float(np.max(np.abs(total - between - within - unresolved)))
    if err > tolerance:
        raise ArithmeticError(f"Decomposition residual {err} exceeds tolerance {tolerance}")
    return {
        "between_group": between,
        "within_group": within,
        "unresolved_emergence_or_loss": unresolved,
        "between_by_group": between_by_group,
        "within_by_group": within_by_group,
        "emerg_by_group": emerg_by_group,
        "total": total,
        "common_support_groups": common,
        "max_absolute_residual": err,
    }


def project_onto_frozen_decoder(decomp: dict, intercept: float, weights: np.ndarray) -> dict:
    """Project exact decomposition components onto fixed linear decoder weights."""
    w = np.asarray(weights, dtype=float).ravel()
    s_bet = float(w @ decomp["between_group"])
    s_wit = float(w @ decomp["within_group"])
    s_emg = float(w @ decomp["unresolved_emergence_or_loss"])
    s_tot = float(w @ decomp["total"])
    return {
        "intercept": float(intercept),
        "score_total": float(intercept + s_tot),
        "linear_total": s_tot,
        "linear_between": s_bet,
        "linear_within": s_wit,
        "linear_emergence": s_emg,
        "linear_between_by_group": (decomp["between_by_group"] @ w).tolist(),
        "linear_within_by_group": (decomp["within_by_group"] @ w).tolist(),
        "projection_residual": float(abs(s_tot - (s_bet + s_wit + s_emg))),
    }
