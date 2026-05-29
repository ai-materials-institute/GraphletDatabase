#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build graphlet JSONs, dynamic bin centers, and histogram JSONs from a CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from graphlet_core import run_csv_graphlet_histogram_build  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_base = Path(PROJECT_ROOT)
    default_out_root = default_base / "ICSD_Features"

    parser = argparse.ArgumentParser(
        description="Build graphlets, dynamic bin centers, and histograms from CIF paths in a CSV."
    )
    parser.add_argument(
        "--csv-path",
        default=str(default_base / "Misc" / "merged_superconductor_data.csv"),
        help="Path to the CSV file containing a 'cif' column.",
    )
    parser.add_argument(
        "--cif-column",
        default="cif",
        help="CSV column containing CIF paths. Default: 'cif'.",
    )
    parser.add_argument(
        "--out-root",
        default=str(default_out_root),
        help="Root output directory. Default: <project>/ICSD_Features",
    )
    parser.add_argument(
        "--graphlet-dir-name",
        default="CSV_Graphlets",
        help="Subdirectory name for graphlet JSON outputs.",
    )
    parser.add_argument(
        "--histogram-dir-name",
        default="Classification_Histograms",
        help="Subdirectory name for histogram JSON outputs.",
    )
    parser.add_argument(
        "--bin-centers-name",
        default="csv_dynamic_bin_centers.json",
        help="Filename for the dynamic bin-centers JSON.",
    )
    parser.add_argument(
        "--manifest-name",
        default="classification_histogram_manifest.json",
        help="Filename for the workflow manifest JSON.",
    )
    parser.add_argument(
        "--progress-log-name",
        default="classification_histogram_progress.log",
        help="Filename for the timestamped progress log.",
    )
    parser.add_argument(
        "--missing-name",
        default="csv_missing_cifs.txt",
        help="Filename for the missing-CIF list.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of bins per feature. Default: 20.",
    )
    parser.add_argument(
        "--hist-density",
        action="store_true",
        help="Store normalized histogram heights instead of raw counts.",
    )
    parser.add_argument(
        "--recompute-bins",
        action="store_true",
        help="Recompute and overwrite the bin-centers file even if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_csv_graphlet_histogram_build(
        csv_path=args.csv_path,
        cif_column=args.cif_column,
        out_root=args.out_root,
        graphlet_dir_name=args.graphlet_dir_name,
        histogram_dir_name=args.histogram_dir_name,
        bin_centers_name=args.bin_centers_name,
        manifest_name=args.manifest_name,
        progress_log_name=args.progress_log_name,
        missing_name=args.missing_name,
        num_bins=args.num_bins,
        hist_density=args.hist_density,
        recompute_bins=args.recompute_bins,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
