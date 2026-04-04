#!/usr/bin/env python3
"""
Convert ADNI native download folders (DICOM) into minimal BIDS ASL layout for
Quality-Check-Toolbox (sub-*/perf/*_asl.nii[.gz], *_aslcontext.tsv).

Requires:
  - dcm2niix on PATH: https://github.com/rordenlab/dcm2niix/releases
  - pip install nibabel

Example:
  python scripts/adni_to_bids_asl.py \\
    --adni "C:/Users/Public/Quality-Check-Toolbox/data/ADNI" \\
    --bids  "C:/Users/Public/Quality-Check-Toolbox/data/ADNI_BIDS"

Notes:
  - Only folders whose path contains a series keyword (default: PASL, PCASL, ASL)
    are converted. Adjust --series-keywords if your series names differ.
  - aslcontext.tsv is written as alternating control / label for every volume.
    Verify this matches your sequence (ADNI PASL is typically interleaved).
  - If dcm2niix produces several NIfTIs, the first 4-D volume is used.
  - M0: add a separate run with --series-keywords M0 --output-suffix _m0scan
    only if you have a distinct M0 series; naming must match BIDS (m0scan).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    print("Install nibabel: pip install nibabel", file=sys.stderr)
    sys.exit(1)


def _adni_subject_to_bids_id(folder_name: str) -> str:
    """002_S_0413 -> sub-002S0413"""
    m = re.match(r"^(\d+)_S_(\d+)$", folder_name)
    if m:
        return f"sub-{m.group(1)}S{m.group(2)}"
    safe = re.sub(r"[^a-zA-Z0-9]", "", folder_name)
    return f"sub-{safe}"


def _session_from_path(dicom_dir: Path) -> str | None:
    """
    Try .../2017-06-21_13_23_38.0/... -> ses-20170621.
    If not found, return None (single-session BIDS layout).
    """
    for part in dicom_dir.parts:
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})_", part)
        if m:
            return f"ses-{m.group(1)}{m.group(2)}{m.group(3)}"
    return None


def _find_dicom_leaf_dirs(adni_root: Path, keywords: tuple[str, ...]) -> list[Path]:
    """Directories that directly contain .dcm files under a path matching keywords."""
    kw_up = tuple(k.upper() for k in keywords)
    seen: set[Path] = set()
    out: list[Path] = []

    for dcm in adni_root.rglob("*.dcm"):
        parent = dcm.parent
        if parent in seen:
            continue
        path_upper = str(parent).upper()
        if not any(k in path_upper for k in kw_up):
            continue
        if not any(p.is_file() and p.suffix.lower() == ".dcm" for p in parent.iterdir()):
            continue
        seen.add(parent)
        out.append(parent)

    return sorted(out)


def _run_dcm2niix(dicom_dir: Path, out_dir: Path, dcm2niix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        dcm2niix,
        "-z",
        "y",
        "-b",
        "y",
        "-o",
        str(out_dir),
        "-f",
        "%p_%s",
        str(dicom_dir),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _pick_asl_nifti(nifti_dir: Path) -> Path | None:
    """Prefer 4-D .nii.gz; else largest 4-D .nii."""
    candidates = sorted(nifti_dir.glob("*.nii.gz")) + sorted(nifti_dir.glob("*.nii"))
    best: tuple[int, Path] | None = None
    for p in candidates:
        try:
            img = nib.load(p)
            shape = img.shape
            nd = len(shape)
            if nd == 4:
                nvol = int(shape[3])
                if best is None or nvol > best[0]:
                    best = (nvol, p)
        except Exception:
            continue
    return best[1] if best else None


def _write_aslcontext(n_volumes: int, path: Path) -> None:
    lines = ["volume_type"]
    for i in range(n_volumes):
        lines.append("control" if i % 2 == 0 else "label")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_dataset_description(bids_root: Path) -> None:
    desc = bids_root / "dataset_description.json"
    if desc.exists():
        return
    desc.write_text(
        json.dumps(
            {
                "Name": "ADNI ASL (converted)",
                "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ADNI DICOM -> minimal BIDS ASL for QC Toolbox")
    ap.add_argument("--adni", type=Path, required=True, help="ADNI download root (subject folders)")
    ap.add_argument("--bids", type=Path, required=True, help="Output BIDS root")
    ap.add_argument(
        "--series-keywords",
        nargs="+",
        default=["PASL", "PCASL", "ASL"],
        help="Path must contain one of these (case-insensitive) to be converted",
    )
    ap.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix executable name or path")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List DICOM folders that would be converted, then exit",
    )
    args = ap.parse_args()

    adni_root = args.adni.resolve()
    bids_root = args.bids.resolve()
    if not adni_root.is_dir():
        print(f"Not a directory: {adni_root}", file=sys.stderr)
        return 1

    try:
        subprocess.run(
            [args.dcm2niix, "-h"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "dcm2niix not found. Install from "
            "https://github.com/rordenlab/dcm2niix/releases and add to PATH.",
            file=sys.stderr,
        )
        return 1

    leaves = _find_dicom_leaf_dirs(adni_root, tuple(args.series_keywords))
    if not leaves:
        print(
            f"No DICOM folders found under {adni_root} matching keywords {args.series_keywords}.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        for p in leaves:
            print(p)
        print(f"Total: {len(leaves)} folder(s)")
        return 0

    bids_root.mkdir(parents=True, exist_ok=True)
    _ensure_dataset_description(bids_root)

    converted = 0
    for dicom_dir in leaves:
        rel = dicom_dir.relative_to(adni_root)
        parts = rel.parts
        if not parts:
            continue
        subject_folder = adni_root / parts[0]
        if not subject_folder.is_dir():
            continue
        bids_sub = _adni_subject_to_bids_id(subject_folder.name)
        ses = _session_from_path(dicom_dir)

        if ses:
            perf = bids_root / bids_sub / ses / "perf"
            prefix = f"{bids_sub}_{ses}"
        else:
            perf = bids_root / bids_sub / "perf"
            prefix = bids_sub

        perf.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="dcm2niix_") as tmp:
            tmp_path = Path(tmp)
            try:
                _run_dcm2niix(dicom_dir, tmp_path, args.dcm2niix)
            except subprocess.CalledProcessError as e:
                print(f"[skip] dcm2niix failed for {dicom_dir}:\n{e.stderr}", file=sys.stderr)
                continue

            nii = _pick_asl_nifti(tmp_path)
            if nii is None:
                print(f"[skip] No 4-D NIfTI in {dicom_dir}", file=sys.stderr)
                continue

            img = nib.load(nii)
            nvol = int(img.shape[3]) if img.ndim == 4 else 0
            if nvol < 2:
                print(f"[skip] Need >=2 volumes in {nii} (got {nvol})", file=sys.stderr)
                continue

            dest = perf / f"{prefix}_asl.nii.gz"
            shutil.copy2(nii, dest)
            _write_aslcontext(nvol, perf / f"{prefix}_aslcontext.tsv")

            json_sidecars = list(tmp_path.glob("*.json"))
            if json_sidecars:
                shutil.copy2(json_sidecars[0], perf / f"{prefix}_asl.json")

            print(f"OK {bids_sub} -> {dest}")
            converted += 1

    print(f"Done. Converted {converted} scan(s) -> {bids_root}")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
