#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entrypoints for graphlet-featurization.

This module defines parser factories and thin command wrappers for the
folder-based graphlet build, one-command graphlet pipeline, CSV-driven
histogram workflow, and compact CIF-to-histogram workflow exposed by
``pyproject.toml`` console scripts.

Author: Aaditya Panigrahi
"""

from __future__ import annotations

__author__ = "Aaditya Panigrahi"

import argparse
import json
import os
from pathlib import Path

from graphlet_core import (
    collect_graphlet_json_paths,
    run_csv_graphlet_histogram_build,
    run_folder_graphlet_build,
    run_full_graphlet_histogram_workflow,
    run_graphlet_histogram_build,
    run_graphlet_pipeline,
)


def build_folder_parser() -> argparse.ArgumentParser:
    """
    Create the parser for folder-based CIF to graphlet JSON builds.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with graphlet build options such as input/output
        paths, worker limits, checkpointing, and overwrite behavior.
    """
    parser = argparse.ArgumentParser(
        description="Build graphlet JSON files from CIF files in an input directory."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input CIF files.")
    parser.add_argument("--output-dir", required=True, help="Directory where graphlet JSON files are written.")
    parser.add_argument("--pattern", default="*.cif", help="Glob pattern for CIF files. Default: '*.cif'.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover CIF files under --input-dir.")
    parser.add_argument("--suffix", default="_graphlet.json", help="Output suffix for graphlet JSON files.")
    parser.add_argument("--atomic-radii-path", default=None, help="Optional path to atomic_radii.json.")
    parser.add_argument(
        "--atomic-features-path",
        default=None,
        help="Optional path to Filtered_atomic_features.json.",
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
    parser.add_argument("--progress-log", default=None, help="Optional path for a timestamped progress log file.")
    parser.add_argument("--max-workers", type=int, default=20, help="Requested number of worker processes.")
    parser.add_argument("--cpu-cap", type=int, default=20, help="Hard upper bound for worker count.")
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
        help="Seconds between heartbeat status updates when no task completes.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write state checkpoint every N finished worker tasks.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional status JSON path. Default: <output-dir>/graphlet_build_state.json",
    )
    return parser


def build_folder_main(argv: list[str] | None = None) -> None:
    """
    Run the folder graphlet build CLI and print the manifest JSON.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, arguments are read from
        ``sys.argv`` by ``argparse``.

    Returns
    -------
    None
        The workflow manifest is written to stdout as formatted JSON.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` for invalid arguments or by the workflow when
        no matching CIF files are found.
    RuntimeError
        Raised when fail-fast mode is enabled and a worker reports a failure.
    """
    args = build_folder_parser().parse_args(argv)
    kwargs = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "suffix": args.suffix,
        "include_compact_features": not args.exclude_compact_features,
        "overwrite": args.overwrite,
        "fail_fast": args.fail_fast,
        "progress_every": args.progress_every,
        "manifest_path": args.manifest_path,
        "progress_log": args.progress_log,
        "max_workers": args.max_workers,
        "cpu_cap": args.cpu_cap,
        "max_in_flight": args.max_in_flight,
        "monitor_interval": args.monitor_interval,
        "checkpoint_every": args.checkpoint_every,
        "state_path": args.state_path,
    }
    if args.atomic_radii_path:
        kwargs["atomic_radii_path"] = args.atomic_radii_path
    if args.atomic_features_path:
        kwargs["atomic_features_path"] = args.atomic_features_path
    manifest = run_folder_graphlet_build(**kwargs)
    print(json.dumps(manifest, indent=2))


def build_csv_parser() -> argparse.ArgumentParser:
    """
    Create the parser for CSV-driven graphlet and histogram workflows.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with CSV path, CIF column, output directory, binning,
        and progress-log options.
    """
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Build graphlets, dynamic bin centers, and histograms from CIF paths in a CSV."
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the CSV file containing a CIF column.",
    )
    parser.add_argument("--cif-column", default="cif", help="CSV column containing CIF paths. Default: 'cif'.")
    parser.add_argument(
        "--out-root",
        default=str(repo_root / "ICSD_Features"),
        help="Root output directory.",
    )
    parser.add_argument("--graphlet-dir-name", default="CSV_Graphlets", help="Subdirectory name for graphlet JSON outputs.")
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
    parser.add_argument("--missing-name", default="csv_missing_cifs.txt", help="Filename for the missing-CIF list.")
    parser.add_argument("--num-bins", type=int, default=20, help="Number of bins per feature. Default: 20.")
    parser.add_argument("--hist-density", action="store_true", help="Store normalized histogram heights.")
    parser.add_argument(
        "--recompute-bins",
        action="store_true",
        help="Recompute and overwrite the bin-centers file even if it exists.",
    )
    return parser


def build_csv_main(argv: list[str] | None = None) -> None:
    """
    Run the CSV workflow CLI and print the workflow manifest JSON.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, ``argparse`` reads from
        ``sys.argv``.

    Returns
    -------
    None
        The workflow manifest is written to stdout as formatted JSON.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` for invalid arguments or by the workflow when
        the CSV contains no usable CIF paths.
    """
    args = build_csv_parser().parse_args(argv)
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


def build_pipeline_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the one-command CIF-to-histogram pipeline.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with input, output, binning, reference-subset, and
        worker options.
    """
    parser = argparse.ArgumentParser(
        description="Run the recommended CIF directory -> graphlets -> bin centers -> histograms pipeline."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input CIF files.")
    parser.add_argument("--out-root", required=True, help="Root directory where all pipeline artifacts are written.")
    parser.add_argument("--pattern", default="*.cif", help="Glob pattern for CIF files. Default: '*.cif'.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover CIF files under --input-dir.")
    parser.add_argument("--graphlet-dir-name", default="graphlets", help="Subdirectory name for graphlet JSON outputs.")
    parser.add_argument("--histogram-dir-name", default="histograms", help="Subdirectory name for histogram JSON outputs.")
    parser.add_argument("--bin-centers-name", default="bin_centers.json", help="Filename for shared bin centers.")
    parser.add_argument("--manifest-name", default="pipeline_manifest.json", help="Filename for the pipeline manifest.")
    parser.add_argument(
        "--reference-list",
        default=None,
        help="Optional newline-delimited file of graphlet JSON paths used to derive bin centers.",
    )
    parser.add_argument(
        "--reference-limit",
        type=int,
        default=None,
        help="Use only the first N generated graphlet JSONs as the bin-center reference set.",
    )
    parser.add_argument("--num-bins", type=int, default=20, help="Number of bins per feature. Default: 20.")
    parser.add_argument(
        "--bin-width-factor",
        type=float,
        default=1.0,
        help="Bin-width scaling factor for dynamic bin range estimation. Default: 1.0.",
    )
    parser.add_argument("--hist-density", action="store_true", help="Normalize histogram heights.")
    parser.add_argument("--recompute-bins", action="store_true", help="Recompute bin centers if the file exists.")
    parser.add_argument("--overwrite-graphlets", action="store_true", help="Rebuild existing graphlet JSON files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first CIF failure.")
    parser.add_argument("--max-workers", type=int, default=20, help="Requested number of graphlet worker processes.")
    parser.add_argument("--cpu-cap", type=int, default=20, help="Hard upper bound for graphlet worker count.")
    return parser


def pipeline_main(argv: list[str] | None = None) -> None:
    """
    Run the one-command graphlet featurization pipeline CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, ``argparse`` reads from
        ``sys.argv``.

    Returns
    -------
    None
        The pipeline manifest is written to stdout as formatted JSON.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` for invalid arguments or by the workflow when
        no CIF/reference files are available.
    """
    args = build_pipeline_parser().parse_args(argv)
    reference_graphlet_paths = _read_path_list(args.reference_list) if args.reference_list else None
    manifest = run_graphlet_pipeline(
        input_dir=args.input_dir,
        out_root=args.out_root,
        pattern=args.pattern,
        recursive=args.recursive,
        graphlet_dir_name=args.graphlet_dir_name,
        histogram_dir_name=args.histogram_dir_name,
        bin_centers_name=args.bin_centers_name,
        manifest_name=args.manifest_name,
        reference_graphlet_paths=reference_graphlet_paths,
        reference_limit=args.reference_limit,
        num_bins=args.num_bins,
        bin_width_factor=args.bin_width_factor,
        hist_density=args.hist_density,
        recompute_bins=args.recompute_bins,
        overwrite_graphlets=args.overwrite_graphlets,
        fail_fast=args.fail_fast,
        max_workers=args.max_workers,
        cpu_cap=args.cpu_cap,
    )
    print(json.dumps(manifest, indent=2))


def build_histograms_parser() -> argparse.ArgumentParser:
    """
    Create the parser for graphlet JSON to histogram JSON builds.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for graphlet inputs, optional bin-reference inputs,
        bin-center reuse or derivation, resume behavior, and histogram output.
    """
    parser = argparse.ArgumentParser(
        description="Build histogram JSON files from existing graphlet JSON files."
    )
    parser.add_argument(
        "graphlet_paths",
        nargs="*",
        help="Explicit graphlet JSON files to histogram.",
    )
    parser.add_argument(
        "--graphlet-dir",
        action="append",
        default=[],
        help="Directory of graphlet JSON files. May be supplied more than once.",
    )
    parser.add_argument(
        "--graphlet-list",
        action="append",
        default=[],
        help="Newline-delimited graphlet JSON path list. May be supplied more than once.",
    )
    parser.add_argument(
        "--pattern",
        default="*_graphlet.json",
        help="Glob pattern for --graphlet-dir scans. Default: '*_graphlet.json'.",
    )
    parser.add_argument(
        "--reference-graphlet-dir",
        action="append",
        default=[],
        help="Directory of graphlet JSON files used only to derive bin centers.",
    )
    parser.add_argument(
        "--reference-graphlet-list",
        action="append",
        default=[],
        help="Newline-delimited graphlet JSON path list used only to derive bin centers.",
    )
    parser.add_argument(
        "--reference-graphlet",
        action="append",
        default=[],
        help="Explicit graphlet JSON file used only to derive bin centers. May be repeated.",
    )
    parser.add_argument(
        "--reference-pattern",
        default=None,
        help="Glob pattern for --reference-graphlet-dir scans. Default: same as --pattern.",
    )
    parser.add_argument(
        "--histogram-out-dir",
        required=True,
        help="Directory where histogram JSON files will be written.",
    )
    parser.add_argument(
        "--bin-centers-path",
        default=None,
        help="Optional bin-centers JSON path. Existing files are reused unless --recompute-bins is passed.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest JSON path. Default: <histogram-out-dir>/histogram_build_manifest.json.",
    )
    parser.add_argument(
        "--progress-log",
        default=None,
        help="Optional progress log path. Default: <histogram-out-dir>/histogram_build_progress.log.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional live state JSON path. Default: <histogram-out-dir>/histogram_build_state.json.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N graphlet files. Default: 1000.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Worker processes for histogram writing. Default: 1.",
    )
    parser.add_argument("--num-bins", type=int, default=20, help="Number of bins per feature. Default: 20.")
    parser.add_argument(
        "--bin-width-factor",
        type=float,
        default=1.0,
        help="Bin-width scaling factor for dynamic bin range estimation. Default: 1.0.",
    )
    parser.add_argument("--hist-density", action="store_true", help="Normalize histogram heights.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise when graphlet features are missing from the bin-center definition.",
    )
    parser.add_argument(
        "--recompute-bins",
        action="store_true",
        help="Recompute bin centers even when --bin-centers-path already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing histogram files only when their stored build settings match.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first graphlet histogram failure.")
    return parser


def build_histograms_main(argv: list[str] | None = None) -> None:
    """
    Run the graphlet JSON to histogram JSON build CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, ``argparse`` reads from
        ``sys.argv``.

    Returns
    -------
    None
        The histogram build manifest is written to stdout as formatted JSON.

    Raises
    ------
    SystemExit
        Raised when no graphlet inputs are provided or parsing fails.
    RuntimeError
        Raised when fail-fast mode is enabled and a graphlet fails.
    """
    args = build_histograms_parser().parse_args(argv)
    graphlet_paths = collect_graphlet_json_paths(
        graphlet_dirs=args.graphlet_dir,
        graphlet_lists=args.graphlet_list,
        graphlet_paths=args.graphlet_paths,
        pattern=args.pattern,
    )
    reference_pattern = args.reference_pattern or args.pattern
    reference_paths = collect_graphlet_json_paths(
        graphlet_dirs=args.reference_graphlet_dir,
        graphlet_lists=args.reference_graphlet_list,
        graphlet_paths=args.reference_graphlet,
        pattern=reference_pattern,
    )
    manifest = run_graphlet_histogram_build(
        graphlet_paths=graphlet_paths,
        histogram_out_dir=args.histogram_out_dir,
        reference_graphlet_paths=reference_paths or None,
        bin_centers_path=args.bin_centers_path,
        manifest_path=args.manifest_path,
        progress_log=args.progress_log,
        state_path=args.state_path,
        progress_every=args.progress_every,
        max_workers=args.max_workers,
        num_bins=args.num_bins,
        bin_width_factor=args.bin_width_factor,
        hist_density=args.hist_density,
        strict=args.strict,
        recompute_bins=args.recompute_bins,
        resume=args.resume,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(manifest, indent=2))


def compact_main(argv: list[str] | None = None) -> None:
    """
    Run the compact CIF-to-graphlet-to-histogram workflow CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, ``argparse`` reads from
        ``sys.argv``.

    Returns
    -------
    None
        The compact workflow result summary is written to stdout as formatted
        JSON.

    Raises
    ------
    SystemExit
        Raised when argument parsing fails or no CIF files can be resolved
        from the supplied inputs.
    """
    args = build_compact_parser().parse_args(argv)
    cif_paths = _expand_cif_inputs(args.cif_inputs)
    if not cif_paths:
        raise SystemExit("No CIF files were found in the provided inputs.")

    result = run_full_graphlet_histogram_workflow(
        cif_paths,
        graphlet_out_dir=args.graphlet_out_dir,
        bin_centers_path=args.bin_centers_path,
        histogram_out_dir=args.histogram_out_dir,
        num_bins=args.num_bins,
        bin_width_factor=args.bin_width_factor,
        hist_density=args.hist_density,
        strict=args.strict,
        prefer_existing_bins=not args.recompute_bins,
        include_compact_features=not args.exclude_compact_features,
    )
    print(json.dumps(result, indent=2))


def build_main_parser() -> argparse.ArgumentParser:
    """
    Create the top-level command dispatcher parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``pipeline``, ``build-folder``, ``build-csv``,
        ``build-histograms``, and ``compact-workflow`` subcommands registered.
    """
    parser = argparse.ArgumentParser(description="graphlet-featurization command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "pipeline",
        add_help=False,
        help="Run the recommended CIF directory to histogram pipeline.",
    )
    subparsers.add_parser(
        "build-folder",
        add_help=False,
        help="Build graphlet JSONs from a folder of CIF files.",
    )
    subparsers.add_parser(
        "build-csv",
        add_help=False,
        help="Build graphlets and histograms from a CSV of CIF paths.",
    )
    subparsers.add_parser(
        "build-histograms",
        add_help=False,
        help="Build histogram JSONs from existing graphlet JSON files.",
    )
    subparsers.add_parser(
        "compact-workflow",
        add_help=False,
        help="Run the end-to-end graphlet-to-histogram workflow.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Dispatch the top-level CLI command to the selected subcommand.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments to parse. If None, ``argparse`` reads from
        ``sys.argv``.

    Returns
    -------
    None
        Control is delegated to the selected subcommand entrypoint.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` for invalid or missing subcommands.
    """
    parser = build_main_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "pipeline":
        pipeline_main(remaining)
        return
    if args.command == "build-folder":
        build_folder_main(remaining)
        return
    if args.command == "build-csv":
        build_csv_main(remaining)
        return
    if args.command == "build-histograms":
        build_histograms_main(remaining)
        return
    compact_main(remaining)


def _default_csv_out_root() -> str:
    """
    Return the default output root for CSV workflows.

    Returns
    -------
    str
        Absolute path-like string pointing at the repository-level
        ``ICSD_Features`` directory.
    """
    return str(_repo_root() / "ICSD_Features")


def build_compact_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the end-to-end compact histogram workflow.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for CIF inputs, graphlet output directory,
        bin-center file, histogram output directory, and binning options.
    """
    parser = argparse.ArgumentParser(
        description="Run the CIF -> graphlet JSON -> bin centers -> histogram JSON workflow."
    )
    parser.add_argument(
        "cif_inputs",
        nargs="+",
        help="One or more CIF files or directories containing CIF files.",
    )
    parser.add_argument(
        "--graphlet-out-dir",
        required=True,
        help="Directory where graphlet JSON files will be written.",
    )
    parser.add_argument(
        "--bin-centers-path",
        required=True,
        help="Path where the bin-centers JSON file will be written or reused.",
    )
    parser.add_argument(
        "--histogram-out-dir",
        required=True,
        help="Directory where histogram JSON files will be written.",
    )
    parser.add_argument("--num-bins", type=int, default=20, help="Number of bins per feature. Default: 20.")
    parser.add_argument(
        "--bin-width-factor",
        type=float,
        default=1.0,
        help="Bin-width scaling factor for the dynamic graphlet outer range. Default: 1.0.",
    )
    parser.add_argument(
        "--hist-density",
        action="store_true",
        help="Normalize histogram heights instead of storing raw counts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise if a compact feature JSON contains a feature without a stored bin center.",
    )
    parser.add_argument(
        "--recompute-bins",
        action="store_true",
        help="Force regeneration of the bin-centers file even if it already exists.",
    )
    parser.add_argument(
        "--exclude-compact-features",
        action="store_true",
        help="Do not include raw_features_counts in the graphlet JSON files.",
    )
    return parser


def _expand_cif_inputs(inputs: list[str]) -> list[str]:
    """
    Expand file and directory CLI inputs into a stable list of CIF paths.

    Parameters
    ----------
    inputs : list of str
        File paths or directories supplied on the command line. Directories are
        searched non-recursively for ``*.cif`` files.

    Returns
    -------
    list of str
        Ordered CIF paths with duplicates removed.
    """
    cif_paths: list[str] = []
    for item in inputs:
        path = Path(os.path.abspath(os.path.expanduser(item)))
        if path.is_dir():
            cif_paths.extend(sorted(str(p) for p in path.glob("*.cif")))
        else:
            cif_paths.append(str(path))

    seen: set[str] = set()
    ordered: list[str] = []
    for path in cif_paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _read_path_list(path: str) -> list[str]:
    """
    Read a newline-delimited path list.

    Parameters
    ----------
    path : str
        Text file containing one path per line. Blank lines and lines starting
        with ``#`` are ignored.

    Returns
    -------
    list of str
        Ordered paths from the file.
    """
    with open(path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _repo_root() -> Path:
    """
    Return the repository root inferred from this source file path.

    Returns
    -------
    pathlib.Path
        Repository root path, computed relative to ``src/graphlet_cli.py``.
    """
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
