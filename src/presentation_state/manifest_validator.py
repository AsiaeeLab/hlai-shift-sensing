"""Structural contrast and fold manifest validator for paired-shift evaluations."""

from __future__ import annotations

import pandas as pd

REQUIRED_CONTRAST_COLUMNS = {
    "contrast_id",
    "study",
    "lineage",
    "label",
    "baseline_sample_ids",
    "treated_sample_ids",
}
REQUIRED_FOLD_COLUMNS = {"fold_id", "contrast_id", "role"}


def _normalize_required_fields(frame: pd.DataFrame, columns: set[str], name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in sorted(columns):
        values = result[column]
        if values.isna().any():
            raise ValueError(f"{name}.{column} contains a missing value")
        values = values.astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"{name}.{column} contains a blank value")
        result[column] = values
    return result


def _sample_ids(value) -> set[str]:
    if pd.isna(value):
        raise ValueError("Canonical biological sample IDs cannot be missing")
    parts = [part.strip() for part in str(value).split(";")]
    if any(not part for part in parts):
        raise ValueError("Canonical biological sample ID lists contain an empty identifier")
    return set(parts)


def _union_ids(values) -> set[str]:
    out: set[str] = set()
    for v in values:
        out.update(_sample_ids(v))
    return out


def validate_design_manifests(contrasts: pd.DataFrame, folds: pd.DataFrame, strict_study: bool = False) -> dict:
    missing_c = REQUIRED_CONTRAST_COLUMNS - set(contrasts.columns)
    missing_f = REQUIRED_FOLD_COLUMNS - set(folds.columns)
    if missing_c or missing_f:
        raise ValueError(f"Missing contrast columns {sorted(missing_c)} or fold columns {sorted(missing_f)}")
    if contrasts.empty or folds.empty:
        raise ValueError("Manifests must contain at least one row")

    c = _normalize_required_fields(contrasts, REQUIRED_CONTRAST_COLUMNS, "contrasts")
    f = _normalize_required_fields(folds, REQUIRED_FOLD_COLUMNS, "folds")
    for col in ("baseline_sample_ids", "treated_sample_ids"):
        for cid, val in zip(c["contrast_id"], c[col]):
            _sample_ids(val)

    if c["contrast_id"].duplicated().any():
        raise ValueError("contrast_id is not unique")
    if not set(f["role"]) <= {"train", "test"}:
        raise ValueError("role must be train or test")
    if not set(f["contrast_id"]) <= set(c["contrast_id"]):
        raise ValueError("Fold manifest references an unknown contrast_id")
    if f.duplicated(["fold_id", "contrast_id"]).any():
        raise ValueError("Duplicate (fold_id, contrast_id) entry in folds")

    n_labels = int(c["label"].nunique())
    records = []
    for fold_id, grp in f.groupby("fold_id", sort=False):
        if set(grp["role"]) != {"train", "test"}:
            raise ValueError(f"Fold {fold_id!r} must contain both train and test roles")
        merged = grp.merge(c, on="contrast_id", validate="many_to_one")
        tr = merged[merged["role"] == "train"]
        te = merged[merged["role"] == "test"]
        tr_ids = _union_ids(tr["baseline_sample_ids"]) | _union_ids(tr["treated_sample_ids"])
        te_ids = _union_ids(te["baseline_sample_ids"]) | _union_ids(te["treated_sample_ids"])
        unseen_labels = sorted(set(te["label"]) - set(tr["label"]))
        train_two = bool(tr["label"].nunique() >= 2)
        auc_def = bool(n_labels == 2 and te["label"].nunique() == 2)
        row = {
            "fold_id": str(fold_id),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "overlap_contrasts": sorted(set(tr["contrast_id"]) & set(te["contrast_id"])),
            "overlap_lineages": sorted(set(tr["lineage"]) & set(te["lineage"])),
            "overlap_studies": sorted(set(tr["study"]) & set(te["study"])),
            "overlap_biological_samples": sorted(tr_ids & te_ids),
            "training_classification_label_checks_pass": bool(train_two and not unseen_labels),
            "auc_defined_in_test_fold": auc_def,
        }
        row["structural_pass"] = bool(
            not row["overlap_contrasts"]
            and not row["overlap_lineages"]
            and not row["overlap_biological_samples"]
            and (not strict_study or not row["overlap_studies"])
        )
        records.append(row)

    return {
        "folds": records,
        "all_structural_checks_pass": bool(records) and all(r["structural_pass"] for r in records),
        "all_training_classification_label_checks_pass": bool(records) and all(r["training_classification_label_checks_pass"] for r in records),
        "n_unique_contrasts": int(len(c)),
        "n_unique_lineages": int(c["lineage"].nunique()),
        "n_unique_studies": int(c["study"].nunique()),
    }
