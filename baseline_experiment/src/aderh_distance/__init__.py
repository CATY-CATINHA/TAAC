"""Distance-concentration experiments for the original ADERH detector."""

from .data_generation import DEFAULT_DIMENSIONS, SyntheticDataset, make_nested_datasets
from .metrics import RelativeContrastResult, relative_contrast

__all__ = [
    "DEFAULT_DIMENSIONS",
    "RelativeContrastResult",
    "SyntheticDataset",
    "make_nested_datasets",
    "relative_contrast",
]

