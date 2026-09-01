"""Dataset discovery and temporal-alignment utilities."""

from .naming import CANONICAL_PHASES, PROTOCOL_PHASES
from .ecg_polarity import apply_ecg_polarity, get_ecg_multiplier

__all__ = [
    "CANONICAL_PHASES", "PROTOCOL_PHASES", "get_ecg_multiplier",
    "apply_ecg_polarity",
]
