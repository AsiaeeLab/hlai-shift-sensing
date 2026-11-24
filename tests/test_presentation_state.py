"""Unit tests for sequence geometry, symmetric Kitagawa-Oaxaca decomposition, and fold manifests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presentation_state import (
    AMINO_ACIDS,
    GROUPS,
    decompose_kitagawa_oaxaca,
    geometry_check,
    project_onto_frozen_decoder,
    sequence_map,
    validate_design_manifests,
)


class TestPresentationState(unittest.TestCase):
    def test_proposition1_rank_67_affine_66(self) -> None:
        rank_lin, rank_aff = geometry_check()
        self.assertEqual(rank_lin, 67)
        self.assertEqual(rank_aff, 66)
        self.assertEqual(GROUPS, ("AILMVFWYP", "CNQSTG", "DE", "RHK"))

    def test_kitagawa_oaxaca_exact_identity_and_projection(self) -> None:
        rng = np.random.default_rng(20261119)
        for _ in range(100):
            w0 = rng.dirichlet(np.ones(4))
            w1 = rng.dirichlet(np.ones(4))
            mu0 = rng.dirichlet(np.ones(20), size=4)
            mu1 = rng.dirichlet(np.ones(20), size=4)
            res = decompose_kitagawa_oaxaca(w0, w1, w0[:, None] * mu0, w1[:, None] * mu1)
            self.assertLess(res["max_absolute_residual"], 1e-14)
            beta = rng.normal(size=20)
            proj = project_onto_frozen_decoder(res, intercept=-0.35, weights=beta)
            self.assertLess(proj["projection_residual"], 1e-14)
            self.assertAlmostEqual(
                proj["score_total"],
                -0.35 + proj["linear_between"] + proj["linear_within"] + proj["linear_emergence"],
                places=12,
            )

    def test_kitagawa_oaxaca_emergent_group_isolated(self) -> None:
        w0 = np.array([1.0, 0.0])
        w1 = np.array([0.6, 0.4])
        a0 = np.array([[0.8, 0.2], [0.0, 0.0]])
        a1 = np.array([[0.48, 0.12], [0.1, 0.3]])
        res = decompose_kitagawa_oaxaca(w0, w1, a0, a1)
        np.testing.assert_allclose(res["unresolved_emergence_or_loss"], [0.1, 0.3], atol=1e-12)
        self.assertTrue(bool(res["common_support_groups"][0]))
        self.assertFalse(bool(res["common_support_groups"][1]))

    def test_isobaric_il_linearity(self) -> None:
        peps = ["AIFKLMNPQ", "ALFKLMNPQ", "ACDEFGHIK"]
        dq = np.array([0.2, 0.3, -0.5])
        peps_il = [p.replace("I", "L") for p in peps]
        aa_idx = {a: i for i, a in enumerate(AMINO_ACIDS)}

        def pos180(plist: list[str]) -> np.ndarray:
            m = np.zeros((len(plist), 180))
            for r, p in enumerate(plist):
                for pos in range(9):
                    m[r, 20 * pos + aa_idx[p[pos]]] = 1.0
            return m

        v_direct = dq @ pos180(peps_il)
        # Collapsed unique I=L peptides: 'ALFKLMNPQ' gets 0.2 + 0.3 = 0.5, 'ACDEFGHLK' gets -0.5
        v_collapsed = np.array([0.5, -0.5]) @ pos180(["ALFKLMNPQ", "ACDEFGHLK"])
        np.testing.assert_allclose(v_direct, v_collapsed, atol=1e-15)
        c96_orig, _ = sequence_map(peps, "coarse96")
        c96_il, _ = sequence_map(peps_il, "coarse96")
        np.testing.assert_allclose(c96_orig, c96_il, atol=1e-15)

    def test_manifest_validator_passes_clean_and_rejects_overlap(self) -> None:
        contrasts = pd.DataFrame([
            {"contrast_id": "c1", "study": "S1", "lineage": "L1", "label": "IFNg", "baseline_sample_ids": "b1", "treated_sample_ids": "t1"},
            {"contrast_id": "c2", "study": "S1", "lineage": "L2", "label": "MAPKi", "baseline_sample_ids": "b2", "treated_sample_ids": "t2"},
            {"contrast_id": "c3", "study": "S2", "lineage": "L3", "label": "IFNg", "baseline_sample_ids": "b3", "treated_sample_ids": "t3"},
            {"contrast_id": "c4", "study": "S2", "lineage": "L4", "label": "MAPKi", "baseline_sample_ids": "b4", "treated_sample_ids": "t4"},
        ])
        folds_ok = pd.DataFrame([
            {"fold_id": "F1", "contrast_id": "c1", "role": "test"},
            {"fold_id": "F1", "contrast_id": "c2", "role": "test"},
            {"fold_id": "F1", "contrast_id": "c3", "role": "train"},
            {"fold_id": "F1", "contrast_id": "c4", "role": "train"},
        ])
        out_ok = validate_design_manifests(contrasts, folds_ok, strict_study=True)
        self.assertTrue(out_ok["all_structural_checks_pass"])

        contrasts_leak = contrasts.copy()
        contrasts_leak.loc[2, "baseline_sample_ids"] = "b1"
        out_leak = validate_design_manifests(contrasts_leak, folds_ok, strict_study=True)
        self.assertFalse(out_leak["all_structural_checks_pass"])


if __name__ == "__main__":
    unittest.main()
