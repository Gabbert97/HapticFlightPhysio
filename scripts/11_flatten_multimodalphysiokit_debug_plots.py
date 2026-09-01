#!/usr/bin/env python3
"""Move existing MultimodalPhysioKit debug PNGs into flat modality folders."""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/multimodalphysiokit_debug"
MANIFEST = ROOT / "metadata/multimodalphysiokit_debug_plot_manifest.csv"
MODALITIES = ("ecg", "eda", "resp", "temp", "fnirs")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Manifest has no header")

    before = sorted(OUTPUT.glob("**/*.png"))
    if len(before) != len(rows):
        raise ValueError(f"PNG/manifest mismatch before move: {len(before)} != {len(rows)}")
    manifest_sources = {ROOT / row["plot_path"] for row in rows}
    if manifest_sources != set(before):
        raise ValueError("Manifest paths do not exactly identify the existing PNG set")

    mapping: list[tuple[Path, Path, dict[str, str]]] = []
    destinations: set[Path] = set()
    for row in rows:
        source = ROOT / row["plot_path"]
        modality = row["modality"]
        if modality not in MODALITIES:
            raise ValueError(f"Unexpected modality: {modality}")
        trial = int(row["trial_number"])
        filename = (
            f"{row['participant_id']}_{row['phase']}_trial_{trial:02d}_"
            f"{source.stem}.png"
        )
        destination = OUTPUT / modality / filename
        if destination in destinations:
            raise FileExistsError(f"Destination collision: {destination}")
        if destination.exists() and destination != source:
            raise FileExistsError(f"Unexpected existing destination: {destination}")
        destinations.add(destination)
        mapping.append((source, destination, row))

    original_hashes = {source: digest(source) for source, _, _ in mapping}
    for modality in MODALITIES:
        (OUTPUT / modality).mkdir(parents=True, exist_ok=True)
    for source, destination, _ in mapping:
        source.replace(destination)

    after = sorted(OUTPUT.glob("**/*.png"))
    if len(after) != len(before) or set(after) != destinations:
        raise RuntimeError("Post-move PNG validation failed; old hierarchy retained")
    for source, destination, _ in mapping:
        if digest(destination) != original_hashes[source]:
            raise RuntimeError(f"Image hash changed during move: {destination}")

    for _, destination, row in mapping:
        row["plot_path"] = destination.relative_to(ROOT).as_posix()
    temporary = MANIFEST.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, MANIFEST)

    # Remove only the known macOS artifact, then remove verified-empty directories.
    ds_store = OUTPUT / "Subject_1/evaluation/trial_01/.DS_Store"
    if ds_store.exists():
        ds_store.unlink()
    removed_directories = 0
    modality_paths = {OUTPUT / modality for modality in MODALITIES}
    for directory in sorted((p for p in OUTPUT.glob("**/*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        if directory in modality_paths:
            continue
        directory.rmdir()
        removed_directories += 1

    top = {path.name for path in OUTPUT.iterdir() if path.is_dir()}
    nested = [p for modality in modality_paths for p in modality.iterdir() if p.is_dir()]
    final_pngs = sorted(OUTPUT.glob("**/*.png"))
    manifest_missing = [row["plot_path"] for row in rows if not (ROOT / row["plot_path"]).exists()]
    if top != set(MODALITIES) or nested or len(final_pngs) != len(before) or manifest_missing:
        raise RuntimeError("Final flat-layout validation failed")
    counts = Counter(row["modality"] for row in rows)
    print(f"total_before={len(before)}")
    print(f"total_after={len(final_pngs)}")
    print(f"counts={dict(counts)}")
    print("collisions=0")
    print(f"old_directories_removed={removed_directories}")
    print(f"manifest_rows_updated={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
