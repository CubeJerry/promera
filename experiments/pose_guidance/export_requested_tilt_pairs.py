#!/usr/bin/env python3
"""Export seven requested-angle cone backbones and matched fixed-reference controls.

Selection is based on guide angle only, never recovered angle or confidence.
Requires only the Python standard library on the HPC.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

COLORS = [
    (0.30, 0.51, 0.76), (0.28, 0.66, 0.69), (0.41, 0.64, 0.46),
    (0.73, 0.68, 0.28), (0.84, 0.49, 0.33), (0.75, 0.43, 0.58),
    (0.48, 0.44, 0.73),
]


def backbone_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        needed = {"arm", "kind", "cif", "replicate", "approach_degrees",
                  "guide_approach_deviation_degrees", "guide_axis_error_degrees",
                  "guide_azimuth_degrees"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"Missing expected columns in {path}")
        return [row for row in reader if row["kind"] == "backbone"
                and row["arm"] in {"cone16", "anchors16"}]


def key_of(row: dict[str, str]) -> tuple[int, int]:
    match = re.search(r"/sample(\d+)/backbone\.cif$", row["cif"])
    if not match:
        raise ValueError(f"Not an original generated backbone: {row['cif']}")
    return int(row["replicate"]), int(match.group(1))


def choose_seven(rows: list[dict[str, str]]):
    fixed = {}
    for row in rows:
        if row["arm"] == "anchors16":
            key = key_of(row)
            if key in fixed:
                raise ValueError(f"Duplicate fixed control: {key}")
            fixed[key] = row
    selected = []
    for low in range(0, 35, 5):
        candidates = [r for r in rows if r["arm"] == "cone16"
                      and low <= float(r["guide_approach_deviation_degrees"]) < low + 5]
        if not candidates:
            raise ValueError(f"No guide in {low}-{low + 5} degree band")
        guided = min(candidates, key=lambda r: (
            abs(float(r["guide_approach_deviation_degrees"]) - (low + 2.5)),
            key_of(r),
        ))
        key = key_of(guided)
        if key not in fixed:
            raise ValueError(f"Missing matched fixed control for {key}")
        selected.append((low, guided, fixed[key]))
    return selected


def original_cif(path: str, experiment: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if "/results/" not in path:
        raise FileNotFoundError(f"Not a result path: {path}")
    candidate = experiment / "results" / path.split("/results/", 1)[1]
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot find original CIF: {path} or {candidate}")
    return candidate


def pymol_overlay(antigen: str, binder: str) -> str:
    lines = [
        "# From this extracted directory, run: pymol overlay.pml",
        "# Colored = cone-guided; grey = matching fixed controls (initially hidden).",
        "reinitialize", "bg_color white", "set auto_zoom, off",
        "set orthoscopic, on", "set antialias, 2", "set ray_shadows, off",
    ]
    for index in range(1, 8):
        lines += [
            f"load guided_{index:02d}.cif, guided_{index:02d}",
            f"load fixed_{index:02d}.cif, fixed_{index:02d}",
        ]
    reference = f"(guided_01 and chain {antigen} and name CA)"
    for index in range(1, 8):
        if index != 1:
            lines.append(
                f"align (guided_{index:02d} and chain {antigen} and name CA), {reference}"
            )
        lines.append(
            f"align (fixed_{index:02d} and chain {antigen} and name CA), {reference}"
        )
    lines += [
        "hide everything, all",
        f"show surface, (guided_01 and chain {antigen})",
        f"color gray80, (guided_01 and chain {antigen})",
        f"set transparency, 0.45, (guided_01 and chain {antigen})",
    ]
    for index, (red, green, blue) in enumerate(COLORS, 1):
        lines += [
            f"set_color tilt_color_{index:02d}, [{red:.3f}, {green:.3f}, {blue:.3f}]",
            f"show cartoon, (guided_{index:02d} and chain {binder})",
            f"color tilt_color_{index:02d}, (guided_{index:02d} and chain {binder})",
            f"show cartoon, (fixed_{index:02d} and chain {binder})",
            f"color gray50, (fixed_{index:02d} and chain {binder})",
            f"set cartoon_transparency, 0.55, fixed_{index:02d}",
        ]
    lines += [
        "group guided_backbones, guided_0*",
        "group fixed_controls, fixed_0*",
        *[f"disable fixed_{index:02d}" for index in range(1, 8)],
        f"orient (guided_01 and chain {antigen})",
        "zoom all, buffer=4",
        "set auto_zoom, on",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="List representatives without copying CIFs")
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    chosen = choose_seven(backbone_rows(experiment / "analysis" / "structures.csv"))

    fields = ["requested_band_degrees", "replicate", "backbone_index",
              "requested_guide_degrees", "recovered_guided_degrees",
              "recovered_fixed_degrees", "guide_direction_error_degrees",
              "guide_azimuth_degrees", "guided_source", "fixed_source"]
    records = []
    for low, guided, fixed in chosen:
        rep, index = key_of(guided)
        records.append({
            "requested_band_degrees": f"{low}-{low + 5}",
            "replicate": rep, "backbone_index": index,
            "requested_guide_degrees": f"{float(guided['guide_approach_deviation_degrees']):.3f}",
            "recovered_guided_degrees": f"{float(guided['approach_degrees']):.3f}",
            "recovered_fixed_degrees": f"{float(fixed['approach_degrees']):.3f}",
            "guide_direction_error_degrees": f"{float(guided['guide_axis_error_degrees']):.3f}",
            "guide_azimuth_degrees": f"{float(guided['guide_azimuth_degrees']):.3f}",
            "guided_source": guided["cif"], "fixed_source": fixed["cif"],
        })
    print("Requested band | guided request -> recovered | fixed recovered | replicate/sample")
    for row in records:
        print(f" {row['requested_band_degrees']:>5} deg | "
              f"{row['requested_guide_degrees']:>6} -> {row['recovered_guided_degrees']:>6} deg | "
              f"{row['recovered_fixed_degrees']:>6} deg | "
              f"rep{row['replicate']}/sample{row['backbone_index']}")
    if args.dry_run:
        return

    sources = [(original_cif(g["cif"], experiment),
                original_cif(f["cif"], experiment)) for _, g, f in chosen]
    plan_path = experiment / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    antigen = str((plan.get("targeted_site") or {}).get("chain")
                  or next(iter(plan.get("target_sequences") or {"A": None})))
    binder = str(plan.get("binder_chain") or "B")
    if not all(re.fullmatch(r"[A-Za-z0-9]", chain) for chain in (antigen, binder)):
        raise ValueError("PyMOL export requires one-character antigen/binder chain IDs")

    directory = experiment / "seven_requested_tilt_pairs"
    directory.mkdir(exist_ok=True)
    for index, (guided, fixed) in enumerate(sources, 1):
        shutil.copy2(guided, directory / f"guided_{index:02d}.cif")
        shutil.copy2(fixed, directory / f"fixed_{index:02d}.cif")
    with (directory / "selections.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (directory / "overlay.pml").write_text(pymol_overlay(antigen, binder), encoding="utf-8")
    (directory / "README.txt").write_text(
        "7 cone-guided backbones selected by REQUESTED 5-degree bins, not by output.\n"
        "Each is paired with its replicate/sample-matched anchors16 control.\n"
        "Open overlay.pml in PyMOL; guided structures are colored and fixed controls\n"
        "are grey and initially disabled. Toggle individual fixed_NN objects to compare.\n"
        "See selections.csv for measured tilt, guide angle and direction error.\n",
        encoding="utf-8",
    )
    destination = experiment / "seven_requested_tilt_pairs.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(directory.iterdir()):
            if file.is_file():
                archive.write(file, arcname=file.name)
    print(f"\nPyMOL overlay: {directory / 'overlay.pml'}")
    print(f"Portable ZIP: {destination}")


if __name__ == "__main__":
    main()
