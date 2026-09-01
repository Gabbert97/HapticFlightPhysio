"""Canonical naming rules for physiological and behavioral recordings."""

from __future__ import annotations

import re
from pathlib import Path

CANONICAL_PHASES = (
    "baseline",
    "pre_test",
    "test_1",
    "test_2",
    "test_3",
    "evaluation",
)

PROTOCOL_PHASES = {
    10: "pre_test",
    20: "test_1",
    30: "test_2",
    40: "test_3",
    50: "evaluation",
}

_NORMALIZED_PHASES = {
    re.sub(r"[^a-z0-9]", "", phase.lower()): phase
    for phase in CANONICAL_PHASES
}

BEHAVIOR_FILENAME_RE = re.compile(
    r"^OutputPattern(?P<protocol_code>10|20|30|40|50)_(?P<trial_index>\d+)\.csv$",
    re.IGNORECASE,
)

SUBJECT_DIR_RE = re.compile(r"^Subject_(?P<participant_id>\d+)$", re.IGNORECASE)


def canonical_phase_from_hdf5(path: Path) -> str | None:
    """Map an HDF5 stem to a canonical phase, ignoring case/punctuation."""
    normalized = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    return _NORMALIZED_PHASES.get(normalized)


def parse_behavior_filename(path: Path) -> tuple[int, str, int] | None:
    """Return protocol code, canonical phase, and literal filename suffix."""
    match = BEHAVIOR_FILENAME_RE.fullmatch(path.name)
    if match is None:
        return None
    protocol_code = int(match.group("protocol_code"))
    return protocol_code, PROTOCOL_PHASES[protocol_code], int(match.group("trial_index"))


def participant_numeric_id(path: Path) -> int | None:
    """Extract the numeric ID from a Subject_<id> directory name."""
    match = SUBJECT_DIR_RE.fullmatch(path.name)
    return int(match.group("participant_id")) if match else None
