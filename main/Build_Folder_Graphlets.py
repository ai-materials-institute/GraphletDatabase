#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build graphlet JSON files from CIF files in an input directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from graphlet_core import run_folder_graphlet_build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graphlet JSON files from CIF files in an input directory."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input CIF files.")
    parser.add_argument("--output-dir", required=True, help="Directory where graphlet JSON files are written.")
    parser.add_argument("--pattern", default="*.cif", help="Glob pattern for CIF files. Default: '*.cif'.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover CIF files under --input-dir.",
    )
    parser.add_argument("--suffix", default="_graphlet.json", help="Output suffix for graphlet JSON files.")
    parser.add_argument(
        "--atomic-radii-path",
        default=os.path.join(PROJECT_ROOT, "config", "atomic_radii.json"),
        help="Path to atomic_radii.json.",
    )
    parser.add_argument(
        "--atomic-features-path",
        default=os.path.join(PROJECT_ROOT, "config", "Filtered_atomic_features.json"),
        help="Path to Filtered_atomic_features.json.",
    )
    parser.add_argument(
        "--exclude-compact-features",
        action="store_true",
        help="Do not include raw_features_counts in output graphlet JSON files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one CIF fails.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed CIF files. Default: 100.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional path for manifest JSON. Default: <output-dir>/graphlet_build_manifest.json",
    )
    parser.add_argument(
        "--progress-log",
        default=None,
        help="Optional path for a timestamped progress log file.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Requested number of worker processes. Default: 20.",
    )
    parser.add_argument(
        "--cpu-cap",
        type=int,
        default=20,
        help="Hard upper bound for workers regardless of --max-workers. Default: 20.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum submitted-but-not-finished tasks. Default: 3x effective workers.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between heartbeat status updates when no task completes. Default: 30.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write state checkpoint every N finished worker tasks. Default: 25.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional status JSON path. Default: <output-dir>/graphlet_build_state.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_folder_graphlet_build(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        suffix=args.suffix,
        atomic_radii_path=args.atomic_radii_path,
        atomic_features_path=args.atomic_features_path,
        include_compact_features=not args.exclude_compact_features,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        progress_every=args.progress_every,
        manifest_path=args.manifest_path,
        progress_log=args.progress_log,
        max_workers=args.max_workers,
        cpu_cap=args.cpu_cap,
        max_in_flight=args.max_in_flight,
        monitor_interval=args.monitor_interval,
        checkpoint_every=args.checkpoint_every,
        state_path=args.state_path,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
