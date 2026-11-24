"""Core representation, decomposition, manifest-validation, and configuration utilities."""

from .config import PathResolver, resolve_config
from .decomposition import decompose_kitagawa_oaxaca, project_onto_frozen_decoder
from .manifest_validator import validate_design_manifests
from .representations import AMINO_ACIDS, CLASS, CLASS_NAMES, GROUPS, PAIRS, construct, geometry_check, sequence_map

__all__ = [
    "PathResolver",
    "resolve_config",
    "AMINO_ACIDS",
    "CLASS",
    "CLASS_NAMES",
    "GROUPS",
    "PAIRS",
    "sequence_map",
    "construct",
    "geometry_check",
    "decompose_kitagawa_oaxaca",
    "project_onto_frozen_decoder",
    "validate_design_manifests",
]
