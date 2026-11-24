"""Environment- and CLI-driven path resolution with zero machine-specific paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(keys: tuple[str, ...], default: Path) -> Path:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return Path(v).expanduser().resolve()
    return default.resolve()


@dataclass(frozen=True)
class PathResolver:
    data_dir: Path
    processed_dir: Path
    upstream_dir: Path
    results_dir: Path
    fig_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def raw(self, *parts: str) -> Path:
        return self.raw_dir.joinpath(*parts)

    def processed(self, *parts: str) -> Path:
        cand = self.processed_dir.joinpath(*parts)
        if cand.exists() or self.processed_dir == self.data_dir / "processed":
            return cand
        fallback = self.data_dir.joinpath("processed", *parts)
        if fallback.exists():
            return fallback
        return cand

    def annotations(self, *parts: str) -> Path:
        for sub in ("annotations", "salvage", ""):
            cand = self.processed_dir.joinpath(sub, *parts) if sub else self.processed_dir.joinpath(*parts)
            if cand.exists():
                return cand
        return self.processed_dir.joinpath("annotations", *parts)

    def followup(self, *parts: str) -> Path:
        for sub in ("followup", "salvage_followup", ""):
            cand = self.processed_dir.joinpath(sub, *parts) if sub else self.processed_dir.joinpath(*parts)
            if cand.exists():
                return cand
        return self.processed_dir.joinpath("followup", *parts)

    def upstream(self, *parts: str) -> Path:
        cand = self.upstream_dir.joinpath(*parts)
        if cand.exists():
            return cand
        return self.data_dir.joinpath(*parts)

    def results(self, *parts: str) -> Path:
        p = self.results_dir.joinpath(*parts)
        return p

    def figures(self, *parts: str) -> Path:
        p = self.fig_dir.joinpath(*parts)
        return p


def resolve_config(
    data_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    upstream_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    fig_dir: Path | str | None = None,
) -> PathResolver:
    dd = Path(data_dir).expanduser().resolve() if data_dir else _env_path(("DATA_DIR", "DATA_CACHE"), REPO_ROOT / "data")
    pd_path = (
        Path(processed_dir).expanduser().resolve()
        if processed_dir
        else _env_path(("PROCESSED_DIR", "PROCESSED_CACHE"), dd / "processed")
    )
    ud = (
        Path(upstream_dir).expanduser().resolve()
        if upstream_dir
        else _env_path(("UPSTREAM_DIR",), REPO_ROOT)
    )
    rd = (
        Path(results_dir).expanduser().resolve()
        if results_dir
        else _env_path(("RESULTS_DIR", "RESULTS_CACHE"), REPO_ROOT / "results")
    )
    fd = Path(fig_dir).expanduser().resolve() if fig_dir else _env_path(("FIG_DIR",), REPO_ROOT / "figures")
    rd.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    return PathResolver(dd, pd_path, ud, rd, fd)
