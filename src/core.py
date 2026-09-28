#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphlet and histogram workflow utilities.

This module provides the central implementation for building graphlet feature
payloads from CIF files, deriving dynamic or fixed histogram bin centers,
serializing histogram payloads, and running batch workflows used by the CLI.

Notes
-----
The public workflow functions write JSON artifacts rather than pickles so that
long-running featurization jobs can be inspected, resumed, and consumed by
downstream distance or kernel routines.

Author: Aaditya Panigrahi
"""

from __future__ import annotations

__author__ = "Aaditya Panigrahi"

import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure as PMGStructure

from graphlets import Create_Graphlets


_HISTOGRAM_WORKER_BIN_PAYLOAD: Dict[str, Any] | None = None
_HISTOGRAM_WORKER_BUILD_SETTINGS: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

DEFAULT_ATOMIC_RADII_JSON = os.path.join(CONFIG_DIR, "atomic_radii.json")
DEFAULT_ATOMIC_FEATURES_JSON = os.path.join(CONFIG_DIR, "Filtered_atomic_features.json")

DEFAULT_CLASSIFICATION_BIN_PICKLE = os.path.join(CONFIG_DIR, "bin_centers_classification.pkl")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def resolve_path(path: str | os.PathLike[str]) -> str:
    """
    Return an absolute path with user and environment variables expanded.

    Parameters
    ----------
    path : str or os.PathLike
        Path value to normalize.

    Returns
    -------
    str
        Absolute path string.
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def ensure_dir(path: str | os.PathLike[str]) -> str:
    """
    Create a directory and return its absolute path.

    Parameters
    ----------
    path : str or os.PathLike
        Directory path to create.

    Returns
    -------
    str
        Absolute directory path.
    """
    out = resolve_path(path)
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    """
    Create the parent directory for a path if needed.

    Parameters
    ----------
    path : str or os.PathLike
        File path whose parent directory should exist.

    Returns
    -------
    None
        The parent directory is created as a side effect.
    """
    Path(resolve_path(path)).parent.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """
    Return a local wall-clock timestamp string.

    Returns
    -------
    str
        Timestamp formatted as ``YYYY-mm-dd HH:MM:SS``.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ProgressLogger:
    """
    Minimal stdout and optional file logger for long-running jobs.

    Parameters
    ----------
    log_path : str or os.PathLike or None, optional
        Optional file path that receives the same timestamped messages printed
        to stdout.

    Attributes
    ----------
    log_path : str or None
        Absolute log-file path, or None when file logging is disabled.
    """

    def __init__(self, log_path: str | os.PathLike[str] | None = None):
        """
        Initialize a logger that mirrors messages to an optional file.

        Parameters
        ----------
        log_path : str or os.PathLike or None, optional
            Destination log file. If None, messages are only printed.
        """
        self.log_path = resolve_path(log_path) if log_path else None
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """
        Write a timestamped message to stdout and the configured log file.

        Parameters
        ----------
        message : str
            Message body to log.

        Returns
        -------
        None
            The message is written to stdout and, if configured, the log file.
        """
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")


def progress_step(index: int, total: int, label: str) -> str:
    """
    Format a simple progress label.

    Parameters
    ----------
    index : int
        Current completed item count.
    total : int
        Total number of items.
    label : str
        Prefix label for the progress string.

    Returns
    -------
    str
        String formatted as ``"<label> <index>/<total>"``.
    """
    return f"{label} {index}/{total}"


def collect_paths(
    root_dir: str | os.PathLike[str],
    *,
    pattern: str,
    recursive: bool = False,
    files_only: bool = True,
) -> list[str]:
    """
    Collect matching paths under a root directory.

    Parameters
    ----------
    root_dir : str or os.PathLike
        Directory to search.
    pattern : str
        Glob pattern to match.
    recursive : bool, optional
        If True, search recursively with ``Path.rglob``. Default is False.
    files_only : bool, optional
        If True, return only files. Default is True.

    Returns
    -------
    list of str
        Deterministically sorted matching paths.

    Raises
    ------
    FileNotFoundError
        If ``root_dir`` does not exist.
    NotADirectoryError
        If ``root_dir`` is not a directory.
    """
    root = Path(resolve_path(root_dir))
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    iterator: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
    if files_only:
        return sorted(str(path) for path in iterator if path.is_file())
    return sorted(str(path) for path in iterator)


def collect_cif_paths(
    root_dir: str | os.PathLike[str],
    *,
    recursive: bool = False,
    pattern: str = "*.cif",
) -> list[str]:
    """
    Collect CIF file paths under a root directory.

    Parameters
    ----------
    root_dir : str or os.PathLike
        Directory to search.
    recursive : bool, optional
        If True, search subdirectories recursively. Default is False.
    pattern : str, optional
        Glob pattern for CIF files. Default is ``"*.cif"``.

    Returns
    -------
    list of str
        Sorted CIF file paths.
    """
    return collect_paths(root_dir, pattern=pattern, recursive=recursive, files_only=True)


def _dedupe_preserve_order(paths: Iterable[str]) -> list[str]:
    """
    Return absolute paths with duplicates removed while preserving first use.

    Parameters
    ----------
    paths : iterable of str
        Path strings to normalize and de-duplicate.

    Returns
    -------
    list of str
        Absolute paths in first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        resolved = resolve_path(path)
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _read_path_list_file(path: str | os.PathLike[str]) -> list[str]:
    """
    Read newline-delimited paths, ignoring blank lines and comments.

    Parameters
    ----------
    path : str or os.PathLike
        Text file containing one path per line.

    Returns
    -------
    list of str
        Paths as written in the list file, with environment/user expansion
        applied by callers.
    """
    with open(resolve_path(path), "r") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def collect_graphlet_json_paths(
    *,
    graphlet_dirs: Sequence[str] | None = None,
    graphlet_lists: Sequence[str] | None = None,
    graphlet_paths: Sequence[str] | None = None,
    pattern: str = "*_graphlet.json",
) -> list[str]:
    """
    Collect graphlet JSON paths from directories, list files, and explicit paths.

    Directory scans are intentionally non-recursive. Directory matches are
    sorted; list-file and explicit paths keep their provided order.

    Parameters
    ----------
    graphlet_dirs : sequence of str or None, optional
        Directories to scan for graphlet JSON files.
    graphlet_lists : sequence of str or None, optional
        Newline-delimited path-list files.
    graphlet_paths : sequence of str or None, optional
        Explicit graphlet JSON paths.
    pattern : str, optional
        Glob used for each directory scan. Default is ``"*_graphlet.json"``.

    Returns
    -------
    list of str
        Absolute, de-duplicated graphlet JSON paths.
    """
    collected: list[str] = []
    for graphlet_dir in graphlet_dirs or []:
        collected.extend(collect_paths(graphlet_dir, pattern=pattern, recursive=False, files_only=True))
    for graphlet_list in graphlet_lists or []:
        collected.extend(_read_path_list_file(graphlet_list))
    collected.extend(graphlet_paths or [])
    return _dedupe_preserve_order(collected)


def write_json(path: str | os.PathLike[str], payload: dict, *, indent: int = 2) -> str:
    """
    Write a JSON payload to disk.

    Parameters
    ----------
    path : str or os.PathLike
        Destination JSON path.
    payload : dict
        JSON-serializable object to write.
    indent : int, optional
        Indentation passed to ``json.dump``. Default is 2.

    Returns
    -------
    str
        Absolute output path.
    """
    out_path = resolve_path(path)
    _ensure_parent(out_path)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=indent)
    return out_path


def write_json_atomic(path: str | os.PathLike[str], payload: dict, *, indent: int = 2) -> str:
    """
    Atomically write a JSON payload to disk.

    Parameters
    ----------
    path : str or os.PathLike
        Destination JSON path.
    payload : dict
        JSON-serializable object to write.
    indent : int, optional
        Indentation passed to ``json.dump``. Default is 2.

    Returns
    -------
    str
        Absolute output path.

    Notes
    -----
    Data are first written to a process-specific temporary file and then moved
    into place with ``os.replace``.
    """
    out_path = resolve_path(path)
    _ensure_parent(out_path)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp_path, out_path)
    return out_path


def _load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Load a JSON object from disk.

    Parameters
    ----------
    path : str or os.PathLike
        JSON file to load.

    Returns
    -------
    dict
        Parsed JSON payload.
    """
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def _format_duration(seconds: float | None) -> str:
    """
    Render a compact human-readable duration string.

    Parameters
    ----------
    seconds : float or None
        Duration in seconds. Non-finite, negative, and None values are treated
        as unknown.

    Returns
    -------
    str
        Duration formatted as seconds, minutes, or hours, or ``"unknown"``.
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Graphlet and histogram workflow
# ---------------------------------------------------------------------------

def _load_atomic_data(
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load atomic radii and elemental feature dictionaries.

    Parameters
    ----------
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level scalar features.

    Returns
    -------
    atomic_radii : dict
        Mapping of element symbols to radius values.
    atomic_features_dict : dict
        Mapping of element symbols to feature dictionaries.
    """
    with open(resolve_path(atomic_radii_path), "r") as f:
        atomic_radii = json.load(f)
    with open(resolve_path(atomic_features_path), "r") as f:
        atomic_features_dict = json.load(f)
    return atomic_radii, atomic_features_dict


def _iter_feature_items(compact_payload: Dict[str, Any]) -> Iterable[Tuple[str, List[List[float]]]]:
    """
    Yield feature names and value-count pairs from a compact payload.

    Parameters
    ----------
    compact_payload : dict
        Graphlet JSON payload containing ``raw_features_counts``.

    Yields
    ------
    feat_name : str
        Feature name.
    pairs : list of list of float
        Compact ``[value, count]`` pairs for the feature.
    """
    feature_groups = compact_payload.get("raw_features_counts", {})
    for _group_name, feat_dict in feature_groups.items():
        for feat_name, pairs in feat_dict.items():
            yield feat_name, pairs


def _pairs_to_arrays(pairs: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split compact value-count pairs into arrays.

    Parameters
    ----------
    pairs : sequence of sequence of float
        Compact feature entries, each shaped as ``[value, count]``.

    Returns
    -------
    values : numpy.ndarray
        One-dimensional feature values.
    counts : numpy.ndarray
        One-dimensional counts corresponding to ``values``.

    Raises
    ------
    ValueError
        If ``pairs`` is not a two-column array-like object.
    """
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Each feature entry must be a list of [value, count] pairs.")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def _expand_pairs(pairs: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Expand compact value-count pairs into repeated sample values.

    Parameters
    ----------
    pairs : sequence of sequence of float
        Compact feature entries, each shaped as ``[value, count]``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array in which each value is repeated by its count.
    """
    values, counts = _pairs_to_arrays(pairs)
    if values.size == 0:
        return np.array([], dtype=float)
    return np.repeat(values, counts.astype(int))


def _weighted_percentile(values: np.ndarray, counts: np.ndarray, percentile: float) -> float:
    """
    Compute a percentile from compact value-count arrays.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional feature values.
    counts : numpy.ndarray
        Non-negative counts or weights for each value.
    percentile : float
        Percentile in the inclusive range ``[0, 100]``.

    Returns
    -------
    float
        Weighted percentile value.

    Raises
    ------
    ValueError
        If no positive-weight finite values are available.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mask = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    values = values[mask]
    counts = counts[mask]
    if values.size == 0:
        raise ValueError("Cannot compute weighted percentile from empty values.")

    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    cumulative = np.cumsum(counts)
    threshold = float(percentile) / 100.0 * cumulative[-1]
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _pygraphlets_bin_width_from_counts(
    values: np.ndarray,
    counts: np.ndarray,
    bin_width_factor: float = 1.0,
) -> float:
    """
    Estimate dynamic histogram bin width from compact value-count arrays.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional feature values.
    counts : numpy.ndarray
        Counts or weights for each value.
    bin_width_factor : float, optional
        Multiplicative scale factor applied to the estimated width.

    Returns
    -------
    float
        Positive bin width.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mask = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    values = values[mask]
    counts = counts[mask]
    if values.size == 0:
        raise ValueError("Cannot derive a bin width from an empty feature value list.")

    if len(np.unique(values)) == 1:
        return 0.1

    n = float(np.sum(counts))
    iqr = _weighted_percentile(values, counts, 75) - _weighted_percentile(values, counts, 25)
    if iqr > 0:
        width = 2 * iqr / np.cbrt(n)
    else:
        sturges_width = (float(np.max(values)) - float(np.min(values))) / (np.log2(n) + 1)
        width = sturges_width if sturges_width > 0 else 0.1
    if width < 0.01:
        width = 0.1
    return float(width * bin_width_factor)


def _pygraphlets_outer_edges_from_counts(
    values: np.ndarray,
    counts: np.ndarray,
    bin_width_factor: float = 1.0,
) -> np.ndarray:
    """
    Compute dynamic outer edges from compact value-count arrays.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional feature values.
    counts : numpy.ndarray
        Counts or weights for each value.
    bin_width_factor : float, optional
        Multiplicative scale factor for the estimated width.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of outer bin edges.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mask = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    values = values[mask]
    counts = counts[mask]
    if values.size == 0:
        raise ValueError("Cannot derive edges from an empty feature value list.")

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    bin_width = _pygraphlets_bin_width_from_counts(
        values,
        counts,
        bin_width_factor=bin_width_factor,
    )
    return np.arange(min_val - bin_width / 2, max_val + 3 * bin_width / 2, bin_width, dtype=float)


def _pygraphlets_bin_width(values: np.ndarray, bin_width_factor: float = 1.0) -> float:
    """
    Estimate the dynamic histogram bin width used by the graphlet workflow.

    Parameters
    ----------
    values : numpy.ndarray
        Feature values used to estimate the bin width.
    bin_width_factor : float, optional
        Multiplicative scale factor applied to the estimated width. Default is
        1.0.

    Returns
    -------
    float
        Positive bin width.

    Raises
    ------
    ValueError
        If no finite feature values are available.

    Notes
    -----
    The width uses the Freedman-Diaconis rule when possible, falls back to a
    Sturges-style width, and enforces a minimum width of 0.1 for very narrow or
    constant data.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot derive a bin width from an empty feature value list.")

    if len(np.unique(values)) == 1:
        return 0.1

    n = len(values)
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    if iqr > 0:
        width = 2 * iqr / np.cbrt(n)
    else:
        sturges_width = (float(np.max(values)) - float(np.min(values))) / (np.log2(n) + 1)
        width = sturges_width if sturges_width > 0 else 0.1
    if width < 0.01:
        width = 0.1
    return float(width * bin_width_factor)


def _pygraphlets_outer_edges(values: np.ndarray, bin_width_factor: float = 1.0) -> np.ndarray:
    """
    Compute outer histogram edges using the dynamic graphlet bin-width rule.

    Parameters
    ----------
    values : numpy.ndarray
        Feature values used to determine the histogram range.
    bin_width_factor : float, optional
        Multiplicative scale factor passed to ``_pygraphlets_bin_width``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of outer bin edges.

    Raises
    ------
    ValueError
        If no finite feature values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot derive edges from an empty feature value list.")

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    bin_width = _pygraphlets_bin_width(values, bin_width_factor=bin_width_factor)
    return np.arange(min_val - bin_width / 2, max_val + 3 * bin_width / 2, bin_width, dtype=float)


def _fixed_count_edges_from_outer_range(outer_edges: np.ndarray, num_bins: int = 20) -> np.ndarray:
    """
    Convert an outer edge span into evenly spaced bin edges.

    Parameters
    ----------
    outer_edges : numpy.ndarray
        Edges defining the full outer histogram range.
    num_bins : int, optional
        Number of equal-width bins to create. Default is 20.

    Returns
    -------
    numpy.ndarray
        Array of length ``num_bins + 1`` containing fixed-count bin edges.

    Raises
    ------
    ValueError
        If fewer than two outer edges are provided or ``num_bins < 1``.
    """
    outer_edges = np.asarray(outer_edges, dtype=float)
    if outer_edges.size < 2:
        raise ValueError("At least two outer edges are required.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")
    return np.linspace(float(outer_edges[0]), float(outer_edges[-1]), num_bins + 1, dtype=float)


def _edges_to_centers(edges: np.ndarray) -> np.ndarray:
    """
    Convert bin edges to midpoint bin centers.

    Parameters
    ----------
    edges : numpy.ndarray
        One-dimensional bin edge array.

    Returns
    -------
    numpy.ndarray
        Midpoint centers between adjacent edges.

    Raises
    ------
    ValueError
        If fewer than two edges are provided.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        raise ValueError("At least two edges are required to compute bin centers.")
    return 0.5 * (edges[:-1] + edges[1:])


def _nearest_center_counts(
    values: np.ndarray,
    counts: np.ndarray,
    centers: np.ndarray,
    hist_density: bool,
) -> np.ndarray:
    """
    Aggregate weighted feature values into nearest-center bins.

    Parameters
    ----------
    values : numpy.ndarray
        Feature values to bin.
    counts : numpy.ndarray
        Weights or counts for each value.
    centers : numpy.ndarray
        Fixed bin centers.
    hist_density : bool
        If True, normalize counts so the returned vector sums to one when it
        has positive mass.

    Returns
    -------
    numpy.ndarray
        Histogram heights aligned with ``centers``.
    """
    if centers.size == 0:
        return np.array([], dtype=float)

    if values.size == 0:
        out = np.zeros_like(centers, dtype=float)
    else:
        diffs = np.abs(values[:, None] - centers[None, :])
        idx = np.argmin(diffs, axis=1)
        out = np.bincount(idx, weights=counts, minlength=len(centers)).astype(float)

    if hist_density:
        denom = out.sum()
        if denom > 0:
            out /= denom
    return out


def _padded_hist_array(
    hist_names: List[str],
    hist_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> List[List[List[float]]]:
    """
    Build a padded histogram array for JSON serialization.

    Parameters
    ----------
    hist_names : list of str
        Histogram feature names in output order.
    hist_map : dict
        Mapping from histogram name to ``(centers, heights)`` arrays.

    Returns
    -------
    list of list of list of float
        JSON-safe array shaped as ``(n_histograms, max_nbins, 2)``. Unused bins
        are padded with ``-1.0``.
    """
    max_nbins = max((len(hist_map[name][0]) for name in hist_names), default=0)
    hist_array = np.full((len(hist_names), max_nbins, 2), -1.0, dtype=float)
    for hi, feat in enumerate(hist_names):
        centers, heights = hist_map[feat]
        nb = len(centers)
        hist_array[hi, :nb, 0] = centers
        hist_array[hi, :nb, 1] = heights
    return hist_array.tolist()


def build_compact_feature_payload_from_cif(
    cif_path: str,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
) -> Dict[str, Any]:
    """
    Build a compact graphlet feature payload from one CIF file.

    Parameters
    ----------
    cif_path : str
        Input CIF file path.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.

    Returns
    -------
    dict
        JSON-ready graphlet payload containing compact value-count feature
        dictionaries and structure metadata.
    """
    cif_path = resolve_path(cif_path)
    atomic_radii, atomic_features_dict = _load_atomic_data(
        atomic_radii_path=atomic_radii_path,
        atomic_features_path=atomic_features_path,
    )

    structure = PMGStructure.from_file(cif_path).get_primitive_structure()
    graphlet = Create_Graphlets(structure, atomic_radii)
    graphlet.Get_1_site_graphlets()
    graphlet.Get_2_site_graphlets()
    graphlet.Get_3_site_graphlets()
    graphlet.get_features(atomic_features_dict)
    return graphlet.get_json_payload(cif_path=cif_path, feature_mode="counts", include_graphlets=False)


def build_graphlet_payload_from_cif(
    cif_path: str,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    *,
    include_compact_features: bool = True,
) -> Dict[str, Any]:
    """
    Build a full graphlet JSON payload from one CIF file.

    Parameters
    ----------
    cif_path : str
        Input CIF file path.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.
    include_compact_features : bool, optional
        If True, store compact value-count features. If False, store raw
        feature value lists. Default is True.

    Returns
    -------
    dict
        JSON-ready graphlet payload containing metadata, graphlet records, and
        either compact or raw feature values.
    """
    cif_path = resolve_path(cif_path)
    atomic_radii, atomic_features_dict = _load_atomic_data(
        atomic_radii_path=atomic_radii_path,
        atomic_features_path=atomic_features_path,
    )

    structure = PMGStructure.from_file(cif_path).get_primitive_structure()
    graphlet = Create_Graphlets(structure, atomic_radii)
    graphlet.Get_1_site_graphlets()
    graphlet.Get_2_site_graphlets()
    graphlet.Get_3_site_graphlets()
    graphlet.get_features(atomic_features_dict)
    return graphlet.get_json_payload(
        cif_path=cif_path,
        feature_mode="counts" if include_compact_features else "raw",
        include_graphlets=True,
    )


def build_and_save_graphlet_json(
    cif_path: str,
    out_path: str,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    *,
    include_compact_features: bool = True,
) -> str:
    """
    Build and write a graphlet JSON payload for one CIF file.

    Parameters
    ----------
    cif_path : str
        Input CIF file path.
    out_path : str
        Destination JSON path.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.
    include_compact_features : bool, optional
        If True, write compact value-count features. Default is True.

    Returns
    -------
    str
        Absolute output JSON path.
    """
    payload = build_graphlet_payload_from_cif(
        cif_path,
        atomic_radii_path=atomic_radii_path,
        atomic_features_path=atomic_features_path,
        include_compact_features=include_compact_features,
    )
    return write_json(out_path, payload, indent=2)


def batch_build_graphlet_jsons(
    cif_paths: Sequence[str],
    *,
    graphlet_out_dir: str,
    suffix: str = "_graphlet.json",
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    include_compact_features: bool = True,
) -> List[str]:
    """
    Build graphlet JSON files for multiple CIF paths.

    Parameters
    ----------
    cif_paths : sequence of str
        Input CIF file paths.
    graphlet_out_dir : str
        Directory where graphlet JSON files are written.
    suffix : str, optional
        Filename suffix appended to each CIF stem. Default is
        ``"_graphlet.json"``.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.
    include_compact_features : bool, optional
        If True, include compact value-count feature dictionaries. Default is
        True.

    Returns
    -------
    list of str
        Output graphlet JSON paths in the same order as ``cif_paths``.
    """
    out_dir = ensure_dir(graphlet_out_dir)
    out_paths: List[str] = []
    for cif_path in cif_paths:
        cif_path = resolve_path(cif_path)
        stem = Path(cif_path).stem
        out_path = os.path.join(out_dir, f"{stem}{suffix}")
        build_and_save_graphlet_json(
            cif_path,
            out_path,
            atomic_radii_path=atomic_radii_path,
            atomic_features_path=atomic_features_path,
            include_compact_features=include_compact_features,
        )
        out_paths.append(out_path)
    return out_paths


def derive_dynamic_bin_centers(
    compact_json_paths: Sequence[str],
    out_path: Optional[str] = None,
    *,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
    logger: ProgressLogger | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 1000,
) -> Dict[str, Any]:
    """
    Derive dynamic histogram bin centers from compact graphlet JSON files.

    Parameters
    ----------
    compact_json_paths : sequence of str
        Graphlet JSON files containing ``raw_features_counts``.
    out_path : str or None, optional
        Destination JSON path for the derived bin-center payload. If None, the
        payload is returned without being written.
    num_bins : int, optional
        Number of bins to derive for each feature. Default is 20.
    bin_width_factor : float, optional
        Scale factor used when estimating the outer dynamic graphlet bin
        range. Default is 1.0.
    logger : ProgressLogger or None, optional
        Optional progress logger for large reference sets.
    progress_callback : callable or None, optional
        Optional callback receiving ``(completed, total)`` for state updates.
    progress_every : int, optional
        Number of graphlet files between progress messages. Default is 1000.

    Returns
    -------
    dict
        Bin-center payload with feature names, bin centers, bin edges, padded
        2D centers, outer ranges, and provenance.
    """
    compact_json_paths = list(compact_json_paths)
    progress_every = max(1, int(progress_every))
    aggregated: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    total_paths = len(compact_json_paths)
    if logger:
        logger.log(f"Deriving dynamic bin centers from {total_paths} graphlet JSON files")
    for idx, path in enumerate(compact_json_paths, start=1):
        payload = _load_json(path)
        for feat_name, pairs in _iter_feature_items(payload):
            values, counts = _pairs_to_arrays(pairs)
            if values.size:
                aggregated.setdefault(feat_name, []).append((values, counts))
        if logger and (idx == 1 or idx == total_paths or idx % progress_every == 0):
            logger.log(f"{progress_step(idx, total_paths, 'Bin reference graphlets')} (features={len(aggregated)})")
        if progress_callback and (idx == 1 or idx == total_paths or idx % progress_every == 0):
            progress_callback(idx, total_paths)

    feature_names = sorted(aggregated.keys())
    bin_centers: Dict[str, List[float]] = {}
    bin_edges: Dict[str, List[float]] = {}
    outer_ranges: Dict[str, List[float]] = {}

    for feat_name in feature_names:
        feat_pairs = aggregated[feat_name]
        values = np.concatenate([pair_values for pair_values, _pair_counts in feat_pairs])
        counts = np.concatenate([pair_counts for _pair_values, pair_counts in feat_pairs])
        outer_edges = _pygraphlets_outer_edges_from_counts(
            values,
            counts,
            bin_width_factor=bin_width_factor,
        )
        edges = _fixed_count_edges_from_outer_range(outer_edges, num_bins=num_bins)
        centers = _edges_to_centers(edges)
        bin_edges[feat_name] = edges.tolist()
        bin_centers[feat_name] = centers.tolist()
        outer_ranges[feat_name] = [float(outer_edges[0]), float(outer_edges[-1])]

    max_bins = max((len(v) for v in bin_centers.values()), default=0)
    bin_centers_2d = []
    for feat_name in feature_names:
        centers = list(bin_centers[feat_name])
        padded = centers + [-1.0] * (max_bins - len(centers))
        bin_centers_2d.append(padded)

    payload = {
        "version": 1,
        "binning_mode": "dynamic",
        "feature_names": feature_names,
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "bin_centers_2d": bin_centers_2d,
        "outer_ranges": outer_ranges,
        "source_compact_jsons": [resolve_path(p) for p in compact_json_paths],
        "num_bins": int(num_bins),
        "range_strategy": "pygraphlets_fd_outer_range_fixed_count",
        "bin_width_factor": float(bin_width_factor),
    }
    if out_path is not None:
        write_json(out_path, payload, indent=2)
        if logger:
            logger.log(f"Wrote dynamic bin centers to {resolve_path(out_path)}")
    return payload


def load_predefined_bin_centers(path: str, *, fmt: str = "auto") -> Dict[str, Any]:
    """
    Load bin centers from JSON or legacy pickle format.

    Parameters
    ----------
    path : str
        Bin-center file path.
    fmt : {"auto", "json", "pickle"}, optional
        File format to load. ``"auto"`` selects pickle for ``.pkl`` paths and
        JSON otherwise.

    Returns
    -------
    dict
        Normalized bin-center payload containing ``feature_names`` and
        ``bin_centers``.

    Raises
    ------
    ValueError
        If the JSON payload is missing required keys or ``fmt`` is invalid.
    """
    path = resolve_path(path)
    if fmt == "auto":
        fmt = "pickle" if path.endswith(".pkl") else "json"

    if fmt == "json":
        payload = _load_json(path)
        if "feature_names" not in payload or "bin_centers" not in payload:
            raise ValueError("JSON bin-center file must contain 'feature_names' and 'bin_centers'.")
        return payload

    if fmt == "pickle":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        feature_names = obj["histogram_feature_names"]
        raw_2d = np.asarray(obj["bin_centers_list"], dtype=float)
        bin_centers = {}
        for idx, feat_name in enumerate(feature_names):
            centers = [float(v) for v in raw_2d[idx].tolist() if v != -1]
            bin_centers[feat_name] = centers
        return {
            "version": 1,
            "binning_mode": "fixed_2d",
            "feature_names": list(feature_names),
            "bin_centers": bin_centers,
            "source_path": path,
        }

    raise ValueError("fmt must be one of: 'auto', 'json', 'pickle'.")


def ensure_bin_centers(
    compact_json_paths: Sequence[str],
    *,
    bin_centers_path: str,
    prefer_existing: bool = True,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Load bin centers from disk or derive and save dynamic bin centers.

    Parameters
    ----------
    compact_json_paths : sequence of str
        Graphlet JSON files used to derive bins when needed.
    bin_centers_path : str
        File path to load from or write to.
    prefer_existing : bool, optional
        If True and ``bin_centers_path`` exists, reuse it. Default is True.
    num_bins : int, optional
        Number of bins to derive per feature when a new file is needed.
    bin_width_factor : float, optional
        Scale factor for dynamic range estimation. Default is 1.0.

    Returns
    -------
    dict
        Loaded or newly derived bin-center payload.
    """
    bin_centers_path = resolve_path(bin_centers_path)
    if prefer_existing and os.path.exists(bin_centers_path):
        return load_predefined_bin_centers(bin_centers_path, fmt="auto")
    return derive_dynamic_bin_centers(
        compact_json_paths,
        out_path=bin_centers_path,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )


def histogram_compact_feature_payload(
    compact_payload: Dict[str, Any],
    bin_center_payload: Dict[str, Any],
    *,
    hist_density: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Convert a compact graphlet payload into histogram features.

    Parameters
    ----------
    compact_payload : dict
        Graphlet payload containing compact value-count feature dictionaries.
    bin_center_payload : dict
        Payload containing ``feature_names`` and ``bin_centers``.
    hist_density : bool, optional
        If True, normalize each histogram's heights. Default is False.
    strict : bool, optional
        If True, raise when compact features lack bin centers. Default is
        False.

    Returns
    -------
    dict
        Histogram payload containing padded histogram arrays, flattened
        per-bin features, magpie-like mean/std features, and metadata.

    Raises
    ------
    KeyError
        If ``strict`` is True and the compact payload contains unknown
        features.
    """
    feature_to_pairs = {feat_name: pairs for feat_name, pairs in _iter_feature_items(compact_payload)}
    hist_names = list(bin_center_payload["feature_names"])
    bin_center_map = bin_center_payload["bin_centers"]

    if strict:
        unknown = [feat for feat in feature_to_pairs if feat not in bin_center_map]
        if unknown:
            raise KeyError(f"Compact feature JSON contains features without bin centers: {unknown}")

    hist_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    feat_bin_names: List[str] = []
    feat_bin_values: List[float] = []
    feat_magpie_names: List[str] = []
    feat_magpie_values: List[float] = []

    for feat_name in hist_names:
        values, counts = _pairs_to_arrays(feature_to_pairs.get(feat_name, []))
        centers = np.asarray(bin_center_map[feat_name], dtype=float)
        heights = _nearest_center_counts(values, counts, centers, hist_density=hist_density)

        hist_map[feat_name] = (centers, heights)
        feat_bin_names.extend([f"{feat_name}={center}" for center in centers.tolist()])
        feat_bin_values.extend(heights.tolist())

        expanded = np.repeat(values, counts.astype(int)) if values.size else np.array([], dtype=float)
        mean_v = float(np.mean(expanded)) if expanded.size else 0.0
        std_v = float(np.std(expanded)) if expanded.size else 0.0
        feat_magpie_names.extend([f"{feat_name}_cumulant=1", f"{feat_name}_cumulant=2"])
        feat_magpie_values.extend([mean_v, std_v])

    return {
        "metadata": compact_payload.get("metadata", {}),
        "histogram_mode": bin_center_payload.get("binning_mode", "fixed_2d"),
        "bin_center_source": bin_center_payload.get("source_path"),
        "hist_density": bool(hist_density),
        "hist_names": hist_names,
        "hist_array": _padded_hist_array(hist_names, hist_map),
        "feat_bin_names": feat_bin_names,
        "feat_bin_values": feat_bin_values,
        "feat_magpie_names": feat_magpie_names,
        "feat_magpie_values": feat_magpie_values,
        "histogram_feat_names": hist_names,
        "histogram_features": _padded_hist_array(hist_names, hist_map),
        "formula": compact_payload.get("metadata", {}).get("reduced_formula"),
    }


def histogram_compact_feature_json(
    compact_json_path: str,
    bin_center_payload: Dict[str, Any],
    *,
    out_path: Optional[str] = None,
    hist_density: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load one compact graphlet JSON and create a histogram payload.

    Parameters
    ----------
    compact_json_path : str
        Input graphlet JSON path.
    bin_center_payload : dict
        Payload containing feature names and bin centers.
    out_path : str or None, optional
        Destination histogram JSON path. If None, no file is written.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    strict : bool, optional
        If True, raise for compact features without stored bin centers.
        Default is False.

    Returns
    -------
    dict
        Histogram payload.
    """
    payload = _load_json(compact_json_path)
    hist_payload = histogram_compact_feature_payload(
        payload,
        bin_center_payload,
        hist_density=hist_density,
        strict=strict,
    )
    if out_path is not None:
        write_json(out_path, hist_payload, indent=2)
    return hist_payload


def batch_histogram_compact_feature_jsons(
    compact_json_paths: Sequence[str],
    *,
    bin_centers_path: str,
    histogram_out_dir: str,
    hist_density: bool = False,
    strict: bool = False,
    prefer_existing_bins: bool = True,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
) -> List[str]:
    """
    Build histogram JSON files from graphlet JSON files.

    Parameters
    ----------
    compact_json_paths : sequence of str
        Input graphlet JSON files containing compact feature counts.
    bin_centers_path : str
        Path to load or write bin-center definitions.
    histogram_out_dir : str
        Directory where histogram JSON files are written.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    strict : bool, optional
        If True, raise for compact features without stored bin centers.
        Default is False.
    prefer_existing_bins : bool, optional
        If True, reuse an existing bin-center file. Default is True.
    num_bins : int, optional
        Number of bins used when deriving new bin centers. Default is 20.
    bin_width_factor : float, optional
        Scale factor for dynamic range estimation. Default is 1.0.

    Returns
    -------
    list of str
        Output histogram JSON paths.
    """
    bin_payload = ensure_bin_centers(
        compact_json_paths,
        bin_centers_path=bin_centers_path,
        prefer_existing=prefer_existing_bins,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )

    out_dir = ensure_dir(histogram_out_dir)
    out_paths = []
    for compact_path in compact_json_paths:
        stem = Path(compact_path).stem
        out_path = os.path.join(out_dir, f"{stem}_histogram.json")
        histogram_compact_feature_json(
            compact_path,
            bin_payload,
            out_path=out_path,
            hist_density=hist_density,
            strict=strict,
        )
        out_paths.append(out_path)
    return out_paths


def _bin_center_fingerprint(bin_center_payload: Dict[str, Any]) -> str:
    """
    Fingerprint the effective bin-center definition.

    Parameters
    ----------
    bin_center_payload : dict
        Loaded or derived bin-center payload.

    Returns
    -------
    str
        SHA256 digest for the binning content that controls histograms.
    """
    effective_payload = {
        "binning_mode": bin_center_payload.get("binning_mode"),
        "feature_names": bin_center_payload.get("feature_names", []),
        "bin_centers": bin_center_payload.get("bin_centers", {}),
        "bin_edges": bin_center_payload.get("bin_edges", {}),
        "num_bins": bin_center_payload.get("num_bins"),
        "range_strategy": bin_center_payload.get("range_strategy"),
        "bin_width_factor": bin_center_payload.get("bin_width_factor"),
    }
    raw = json.dumps(effective_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _histogram_build_settings(
    *,
    bin_center_payload: Dict[str, Any],
    hist_density: bool,
    strict: bool,
    num_bins: int,
    bin_width_factor: float,
) -> Dict[str, Any]:
    """
    Build the settings block used for histogram resume compatibility.

    Parameters
    ----------
    bin_center_payload : dict
        Loaded or derived bin-center payload.
    hist_density : bool
        Whether histogram heights are normalized.
    strict : bool
        Whether unknown compact features raise errors.
    num_bins : int
        Requested number of bins used when deriving new bin centers.
    bin_width_factor : float
        Requested dynamic range scale factor.

    Returns
    -------
    dict
        JSON-safe settings block.
    """
    return {
        "histogram_builder_version": 1,
        "bin_centers_fingerprint": _bin_center_fingerprint(bin_center_payload),
        "hist_density": bool(hist_density),
        "strict": bool(strict),
        "num_bins": int(num_bins),
        "bin_width_factor": float(bin_width_factor),
    }


def _histogram_settings_match(path: str, expected_settings: Dict[str, Any]) -> bool:
    """
    Return True when an existing histogram has compatible build settings.

    Parameters
    ----------
    path : str
        Existing histogram JSON path.
    expected_settings : dict
        Current build settings.

    Returns
    -------
    bool
        True if the existing file can be safely reused.
    """
    try:
        payload = _load_json(path)
    except Exception:
        return False
    return payload.get("build_settings") == expected_settings


def _histogram_worker_init(bin_centers_path: str, build_settings: Dict[str, Any]) -> None:
    """
    Initialize per-process histogram build state.

    Parameters
    ----------
    bin_centers_path : str
        Bin-center JSON or pickle path.
    build_settings : dict
        Build settings to embed in each histogram JSON.

    Returns
    -------
    None
        The loaded bin payload and settings are stored in module globals.
    """
    global _HISTOGRAM_WORKER_BIN_PAYLOAD
    global _HISTOGRAM_WORKER_BUILD_SETTINGS
    _HISTOGRAM_WORKER_BIN_PAYLOAD = load_predefined_bin_centers(bin_centers_path, fmt="auto")
    _HISTOGRAM_WORKER_BUILD_SETTINGS = build_settings


def _histogram_build_worker(job: Tuple[str, str, str, bool, bool]) -> Dict[str, Any]:
    """
    Build one histogram JSON in a worker process.

    Parameters
    ----------
    job : tuple
        ``(graphlet_path, out_path, bin_centers_path, hist_density, strict)``.

    Returns
    -------
    dict
        Worker result with success or failure metadata.
    """
    graphlet_path, out_path, bin_centers_path, hist_density, strict = job
    start = time.time()
    try:
        bin_payload = _HISTOGRAM_WORKER_BIN_PAYLOAD
        build_settings = _HISTOGRAM_WORKER_BUILD_SETTINGS
        if bin_payload is None or build_settings is None:
            bin_payload = load_predefined_bin_centers(bin_centers_path, fmt="auto")
            build_settings = _histogram_build_settings(
                bin_center_payload=bin_payload,
                hist_density=hist_density,
                strict=strict,
                num_bins=int(bin_payload.get("num_bins", 0) or 0),
                bin_width_factor=float(bin_payload.get("bin_width_factor", 1.0) or 1.0),
            )

        compact_payload = _load_json(graphlet_path)
        hist_payload = histogram_compact_feature_payload(
            compact_payload,
            bin_payload,
            hist_density=hist_density,
            strict=strict,
        )
        hist_payload["build_settings"] = build_settings
        hist_payload["source_graphlet_path"] = resolve_path(graphlet_path)
        hist_payload["bin_centers_path"] = resolve_path(bin_centers_path)
        write_json(out_path, hist_payload, indent=2)
        return {
            "ok": True,
            "graphlet_path": resolve_path(graphlet_path),
            "out_path": resolve_path(out_path),
            "elapsed_s": time.time() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "graphlet_path": resolve_path(graphlet_path),
            "out_path": resolve_path(out_path),
            "elapsed_s": time.time() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_graphlet_histogram_build(
    *,
    graphlet_paths: Sequence[str],
    histogram_out_dir: str,
    reference_graphlet_paths: Sequence[str] | None = None,
    bin_centers_path: str | None = None,
    manifest_path: str | None = None,
    progress_log: str | None = None,
    state_path: str | None = None,
    progress_every: int = 1000,
    max_workers: int = 1,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
    hist_density: bool = False,
    strict: bool = False,
    recompute_bins: bool = False,
    resume: bool = False,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Build histogram JSON files from existing graphlet JSON files.

    Parameters
    ----------
    graphlet_paths : sequence of str
        Graphlet JSON files to convert to histograms.
    histogram_out_dir : str
        Directory where histogram JSON files are written.
    reference_graphlet_paths : sequence of str or None, optional
        Optional graphlet JSON files used only to derive bin centers.
    bin_centers_path : str or None, optional
        Bin-center JSON path to reuse or write. If None, a default
        ``bin_centers.json`` is written under ``histogram_out_dir`` and bins
        are derived dynamically for this run.
    manifest_path : str or None, optional
        Manifest path. Defaults to
        ``<histogram_out_dir>/histogram_build_manifest.json``.
    progress_log : str or None, optional
        Progress log path. Defaults to
        ``<histogram_out_dir>/histogram_build_progress.log``.
    state_path : str or None, optional
        Live state JSON path. Defaults to
        ``<histogram_out_dir>/histogram_build_state.json``.
    progress_every : int, optional
        Number of graphlet files between progress messages. Default is 1000.
    max_workers : int, optional
        Number of worker processes for histogram writing. Default is 1.
    num_bins : int, optional
        Number of bins used when deriving dynamic bins. Default is 20.
    bin_width_factor : float, optional
        Dynamic range scale factor. Default is 1.0.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    strict : bool, optional
        If True, raise when a graphlet has features without bins.
    recompute_bins : bool, optional
        If True, rebuild ``bin_centers_path`` even if it exists.
    resume : bool, optional
        If True, skip existing histograms only when their build settings match.
    fail_fast : bool, optional
        If True, stop at the first histogram build failure.

    Returns
    -------
    dict
        Manifest summarizing inputs, bin-center provenance, outputs, skipped
        files, failures, and build settings.

    Raises
    ------
    SystemExit
        If no graphlet inputs or no bin-reference graphlets are available.
    RuntimeError
        If ``fail_fast`` is True and a graphlet fails.
    """
    graphlet_paths = _dedupe_preserve_order(graphlet_paths)
    if not graphlet_paths:
        raise SystemExit("No graphlet JSON files were specified for histogram building.")

    histogram_out_dir = ensure_dir(histogram_out_dir)
    manifest_path = resolve_path(manifest_path) if manifest_path else os.path.join(
        histogram_out_dir, "histogram_build_manifest.json"
    )
    progress_log = resolve_path(progress_log) if progress_log else os.path.join(
        histogram_out_dir, "histogram_build_progress.log"
    )
    state_path = resolve_path(state_path) if state_path else os.path.join(
        histogram_out_dir, "histogram_build_state.json"
    )
    progress_every = max(1, int(progress_every))
    max_workers_eff = max(1, min(int(max_workers), int(os.cpu_count() or 1)))
    logger = ProgressLogger(progress_log)
    start_epoch = time.time()

    def _write_histogram_state(
        *,
        status: str,
        phase: str,
        num_written: int = 0,
        num_skipped: int = 0,
        num_failed: int = 0,
        current_index: int = 0,
        total_items: int | None = None,
        failures: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Write the live histogram-build state JSON.

        Parameters
        ----------
        status : str
            Overall build status, for example ``"starting"`` or ``"running"``.
        phase : str
            Name of the current build phase.
        num_written : int, optional
            Histograms written so far. Default is 0.
        num_skipped : int, optional
            Histograms skipped because a compatible output already exists.
            Default is 0.
        num_failed : int, optional
            Histograms that failed. Default is 0.
        current_index : int, optional
            Items completed in the current phase. Default is 0.
        total_items : int or None, optional
            Items in the current phase. If None, the number of graphlet JSONs
            is used.
        failures : list of dict or None, optional
            Failure records; the most recent 50 are written.

        Returns
        -------
        None
            The state is written atomically to ``state_path``.
        """
        elapsed_s = time.time() - start_epoch
        total = len(graphlet_paths) if total_items is None else int(total_items)
        completed = int(current_index)
        pending = max(0, total - completed)
        rate = (completed * 60.0 / elapsed_s) if elapsed_s > 0 else 0.0
        eta_s = (pending / (completed / elapsed_s)) if completed > 0 and elapsed_s > 0 else None
        payload = {
            "status": status,
            "phase": phase,
            "last_update": timestamp(),
            "histogram_out_dir": histogram_out_dir,
            "manifest_path": manifest_path,
            "progress_log": progress_log,
            "state_path": state_path,
            "bin_centers_path": resolve_path(bin_centers_path) if bin_centers_path else None,
            "counts": {
                "num_graphlet_jsons": len(graphlet_paths),
                "num_reference_graphlets": len(reference_graphlet_paths or []),
                "num_current_phase_total": total,
                "num_current_phase_completed": completed,
                "num_current_phase_pending": pending,
                "num_histograms_written": int(num_written),
                "num_histograms_skipped": int(num_skipped),
                "num_histograms_failed": int(num_failed),
            },
            "settings": {
                "num_bins": int(num_bins),
                "bin_width_factor": float(bin_width_factor),
                "hist_density": bool(hist_density),
                "strict": bool(strict),
                "resume": bool(resume),
                "fail_fast": bool(fail_fast),
                "recompute_bins": bool(recompute_bins),
                "max_workers": int(max_workers_eff),
            },
            "timing": {
                "elapsed_seconds": elapsed_s,
                "elapsed_human": _format_duration(elapsed_s),
                "rate_items_per_min": rate,
                "eta_seconds": eta_s,
                "eta_human": _format_duration(eta_s),
            },
            "recent_failures": (failures or [])[-50:],
        }
        write_json_atomic(state_path, payload, indent=2)

    logger.log("Starting graphlet histogram build")
    logger.log(f"Histogram input graphlets: {len(graphlet_paths)}")
    logger.log(f"Histogram output dir: {histogram_out_dir}")
    _write_histogram_state(status="starting", phase="setup")

    if bin_centers_path is None:
        bin_centers_path = os.path.join(histogram_out_dir, "bin_centers.json")
        derive_bins = True
        bin_source = "derived_default"
    else:
        bin_centers_path = resolve_path(bin_centers_path)
        derive_bins = bool(recompute_bins) or not os.path.exists(bin_centers_path)
        bin_source = "recomputed" if recompute_bins else ("derived" if derive_bins else "reused")

    if derive_bins:
        reference_paths = _dedupe_preserve_order(reference_graphlet_paths or graphlet_paths)
        if not reference_paths:
            raise SystemExit("No reference graphlet JSON files are available for bin-center derivation.")
        _write_histogram_state(
            status="running",
            phase="deriving_bin_centers",
            total_items=len(reference_paths),
        )
        bin_payload = derive_dynamic_bin_centers(
            reference_paths,
            out_path=bin_centers_path,
            num_bins=num_bins,
            bin_width_factor=bin_width_factor,
            logger=logger,
            progress_callback=lambda completed, total: _write_histogram_state(
                status="running",
                phase="deriving_bin_centers",
                total_items=total,
                current_index=completed,
                failures=[],
            ),
            progress_every=progress_every,
        )
    else:
        reference_paths = _dedupe_preserve_order(reference_graphlet_paths or [])
        logger.log(f"Reusing existing bin centers from {bin_centers_path}")
        bin_payload = load_predefined_bin_centers(bin_centers_path, fmt="auto")

    build_settings = _histogram_build_settings(
        bin_center_payload=bin_payload,
        hist_density=hist_density,
        strict=strict,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )

    histogram_paths: list[str] = []
    skipped_paths: list[str] = []
    rebuilt_paths: list[str] = []
    failures: list[dict[str, str]] = []
    logger.log(
        "Histogram settings: "
        f"num_bins={int(num_bins)}, bin_width_factor={float(bin_width_factor)}, "
        f"hist_density={bool(hist_density)}, resume={bool(resume)}, max_workers={max_workers_eff}"
    )
    logger.log(f"Building histograms for {len(graphlet_paths)} graphlet JSON files")

    jobs: list[tuple[str, str, str, bool, bool]] = []
    for graphlet_path in graphlet_paths:
        out_path = os.path.join(histogram_out_dir, f"{Path(graphlet_path).stem}_histogram.json")
        if resume and os.path.exists(out_path) and _histogram_settings_match(out_path, build_settings):
            histogram_paths.append(resolve_path(out_path))
            skipped_paths.append(resolve_path(out_path))
            continue
        jobs.append((graphlet_path, out_path, bin_centers_path, bool(hist_density), bool(strict)))

    if skipped_paths:
        logger.log(f"Resume-compatible histograms skipped before submission: {len(skipped_paths)}")

    def _handle_histogram_result(result: Dict[str, Any], completed_count: int) -> bool:
        """
        Record one histogram worker result and report progress.

        Parameters
        ----------
        result : dict
            Worker result with an ``"ok"`` flag and either ``"out_path"`` or
            the error details.
        completed_count : int
            Number of histogram jobs completed so far, including this one.

        Returns
        -------
        bool
            False when the result is a failure and ``fail_fast`` is set, so the
            build should stop; True otherwise.
        """
        if result.get("ok"):
            histogram_paths.append(str(result["out_path"]))
            rebuilt_paths.append(str(result["out_path"]))
        else:
            failure = {
                "graphlet_path": str(result.get("graphlet_path", "")),
                "out_path": str(result.get("out_path", "")),
                "error_type": str(result.get("error_type", "RuntimeError")),
                "error": str(result.get("error", "Unknown error")),
            }
            failures.append(failure)
            logger.log(
                "FAILED "
                f"{failure['graphlet_path']}: {failure['error_type']}: {failure['error']}"
            )
            if fail_fast:
                return False

        if (
            completed_count == 1
            or completed_count == len(graphlet_paths)
            or completed_count % progress_every == 0
        ):
            logger.log(
                f"{progress_step(completed_count, len(graphlet_paths), 'Histograms')} "
                f"(written={len(rebuilt_paths)}, skipped={len(skipped_paths)}, failed={len(failures)})"
            )
            _write_histogram_state(
                status="running",
                phase="building_histograms",
                num_written=len(rebuilt_paths),
                num_skipped=len(skipped_paths),
                num_failed=len(failures),
                current_index=completed_count,
                failures=failures,
            )
        return True

    completed_histogram_count = len(skipped_paths)
    if jobs and max_workers_eff > 1:
        logger.log(f"Submitting {len(jobs)} histogram jobs to {max_workers_eff} worker processes")
        with cf.ProcessPoolExecutor(
            max_workers=max_workers_eff,
            initializer=_histogram_worker_init,
            initargs=(bin_centers_path, build_settings),
        ) as executor:
            for result in executor.map(_histogram_build_worker, jobs, chunksize=10):
                completed_histogram_count += 1
                if not _handle_histogram_result(result, completed_histogram_count):
                    break
    else:
        for job in jobs:
            result = _histogram_build_worker(job)
            completed_histogram_count += 1
            if not _handle_histogram_result(result, completed_histogram_count):
                break

    if skipped_paths and not jobs:
        logger.log(
            f"{progress_step(len(graphlet_paths), len(graphlet_paths), 'Histograms')} "
            f"(written={len(rebuilt_paths)}, skipped={len(skipped_paths)}, failed={len(failures)})"
        )
        _write_histogram_state(
            status="running",
            phase="building_histograms",
            num_written=len(rebuilt_paths),
            num_skipped=len(skipped_paths),
            num_failed=len(failures),
            current_index=len(graphlet_paths),
            failures=failures,
        )

    elapsed_s = time.time() - start_epoch
    manifest = {
        "histogram_out_dir": histogram_out_dir,
        "manifest_path": manifest_path,
        "progress_log": progress_log,
        "state_path": state_path,
        "graphlet_paths": graphlet_paths,
        "reference_graphlet_paths": reference_paths,
        "bin_centers_path": resolve_path(bin_centers_path),
        "bin_centers_source": bin_source,
        "build_settings": build_settings,
        "num_graphlet_jsons": len(graphlet_paths),
        "num_reference_graphlets": len(reference_paths),
        "num_histogram_jsons": len(histogram_paths),
        "num_histograms_written": len(rebuilt_paths),
        "num_histograms_skipped": len(skipped_paths),
        "num_histograms_failed": len(failures),
        "histogram_paths": histogram_paths,
        "written_histogram_paths": rebuilt_paths,
        "skipped_histogram_paths": skipped_paths,
        "failures": failures,
        "resume": bool(resume),
        "fail_fast": bool(fail_fast),
        "recompute_bins": bool(recompute_bins),
        "num_bins": int(num_bins),
        "bin_width_factor": float(bin_width_factor),
        "hist_density": bool(hist_density),
        "strict": bool(strict),
        "elapsed_seconds": elapsed_s,
        "elapsed_human": _format_duration(elapsed_s),
    }
    write_json(manifest_path, manifest, indent=2)
    final_status = "failed" if fail_fast and failures else "completed"
    _write_histogram_state(
        status=final_status,
        phase="finished",
        num_written=len(rebuilt_paths),
        num_skipped=len(skipped_paths),
        num_failed=len(failures),
        current_index=len(graphlet_paths),
        failures=failures,
    )
    logger.log(f"Wrote manifest to {manifest_path}")
    logger.log(
        "Histogram build completed "
        f"(written={len(rebuilt_paths)}, skipped={len(skipped_paths)}, failed={len(failures)})"
    )

    if fail_fast and failures:
        raise RuntimeError(
            "Fail-fast triggered after a graphlet histogram failure. "
            f"See manifest for details: {manifest_path}"
        )
    return manifest


def run_full_graphlet_histogram_workflow(
    cif_paths: Sequence[str],
    *,
    graphlet_out_dir: str,
    bin_centers_path: str,
    histogram_out_dir: str,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
    hist_density: bool = False,
    strict: bool = False,
    prefer_existing_bins: bool = True,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    include_compact_features: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete CIF to graphlet JSON to histogram JSON workflow.

    Parameters
    ----------
    cif_paths : sequence of str
        Input CIF file paths.
    graphlet_out_dir : str
        Directory where graphlet JSON files are written.
    bin_centers_path : str
        Path to load or write bin-center definitions.
    histogram_out_dir : str
        Directory where histogram JSON files are written.
    num_bins : int, optional
        Number of bins used when deriving new bin centers. Default is 20.
    bin_width_factor : float, optional
        Scale factor for dynamic range estimation. Default is 1.0.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    strict : bool, optional
        If True, raise for compact features without stored bin centers.
        Default is False.
    prefer_existing_bins : bool, optional
        If True, reuse an existing bin-center file. Default is True.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.
    include_compact_features : bool, optional
        If True, graphlet JSON files include compact value-count features.
        Default is True.

    Returns
    -------
    dict
        Workflow summary with input paths, output paths, binning settings, and
        artifact counts.
    """
    graphlet_paths = batch_build_graphlet_jsons(
        cif_paths,
        graphlet_out_dir=graphlet_out_dir,
        atomic_radii_path=atomic_radii_path,
        atomic_features_path=atomic_features_path,
        include_compact_features=include_compact_features,
    )

    bin_payload = ensure_bin_centers(
        graphlet_paths,
        bin_centers_path=bin_centers_path,
        prefer_existing=prefer_existing_bins,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )

    histogram_paths = batch_histogram_compact_feature_jsons(
        graphlet_paths,
        bin_centers_path=bin_centers_path,
        histogram_out_dir=histogram_out_dir,
        hist_density=hist_density,
        strict=strict,
        prefer_existing_bins=True,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )

    return {
        "cif_paths": [resolve_path(p) for p in cif_paths],
        "graphlet_paths": graphlet_paths,
        "bin_centers_path": resolve_path(bin_centers_path),
        "histogram_paths": histogram_paths,
        "num_input_cifs": len(cif_paths),
        "num_graphlet_jsons": len(graphlet_paths),
        "num_histogram_jsons": len(histogram_paths),
        "num_bins": int(num_bins),
        "hist_density": bool(hist_density),
        "range_strategy": bin_payload.get("range_strategy"),
    }


# ---------------------------------------------------------------------------
# High-level batch runners used by CLI entrypoints
# ---------------------------------------------------------------------------

def _graphlet_build_worker(
    *,
    cif_path: str,
    out_path: str,
    atomic_radii_path: str,
    atomic_features_path: str,
    include_compact_features: bool,
) -> Dict[str, Any]:
    """
    Build one graphlet JSON payload inside a process-pool worker.

    Parameters
    ----------
    cif_path : str
        Input CIF file path.
    out_path : str
        Destination graphlet JSON path.
    atomic_radii_path : str
        JSON file containing element radii.
    atomic_features_path : str
        JSON file containing element-level feature values.
    include_compact_features : bool
        If True, include compact value-count features in the output payload.

    Returns
    -------
    dict
        Worker result. Successful results contain ``ok=True``, paths, and
        elapsed seconds; failed results contain ``ok=False`` plus error
        metadata.
    """
    start = time.time()
    try:
        build_and_save_graphlet_json(
            cif_path,
            out_path,
            atomic_radii_path=atomic_radii_path,
            atomic_features_path=atomic_features_path,
            include_compact_features=include_compact_features,
        )
        return {
            "ok": True,
            "cif_path": resolve_path(cif_path),
            "out_path": resolve_path(out_path),
            "elapsed_s": time.time() - start,
        }
    except Exception as exc:  # pragma: no cover - worker safety
        return {
            "ok": False,
            "cif_path": resolve_path(cif_path),
            "out_path": resolve_path(out_path),
            "elapsed_s": time.time() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_folder_graphlet_build(
    *,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.cif",
    recursive: bool = False,
    suffix: str = "_graphlet.json",
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    include_compact_features: bool = True,
    overwrite: bool = False,
    fail_fast: bool = False,
    progress_every: int = 100,
    manifest_path: str | None = None,
    progress_log: str | None = None,
    max_workers: int = 20,
    cpu_cap: int = 20,
    max_in_flight: int | None = None,
    monitor_interval: float = 30.0,
    checkpoint_every: int = 25,
    state_path: str | None = None,
) -> Dict[str, Any]:
    """
    Folder-based CIF -> graphlet JSON workflow with parallel processing.

    Resume behavior is file-based by default:
    existing output JSON files are skipped unless ``overwrite=True``.

    Parameters
    ----------
    input_dir : str
        Directory containing CIF files.
    output_dir : str
        Directory where graphlet JSON files and default state files are
        written.
    pattern : str, optional
        Glob pattern used to discover CIF files. Default is ``"*.cif"``.
    recursive : bool, optional
        If True, discover CIF files recursively. Default is False.
    suffix : str, optional
        Suffix appended to each CIF stem for output filenames. Default is
        ``"_graphlet.json"``.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.
    include_compact_features : bool, optional
        If True, include compact value-count features in graphlet outputs.
        Default is True.
    overwrite : bool, optional
        If True, rebuild existing graphlet JSON files. Default is False.
    fail_fast : bool, optional
        If True, stop submitting new jobs after the first failure. Default is
        False.
    progress_every : int, optional
        Number of completed items between progress log messages. Default is
        100.
    manifest_path : str or None, optional
        Optional manifest JSON path. If None, a default file is created in
        ``output_dir``.
    progress_log : str or None, optional
        Optional progress log file path.
    max_workers : int, optional
        Requested process-pool worker count. Default is 20.
    cpu_cap : int, optional
        Hard upper bound on effective worker count. Default is 20.
    max_in_flight : int or None, optional
        Maximum number of submitted but unfinished jobs. If None, defaults to
        three times the effective worker count.
    monitor_interval : float, optional
        Seconds between heartbeat updates when no jobs finish. Default is
        30.0.
    checkpoint_every : int, optional
        Number of finished worker tasks between state-file checkpoints.
        Default is 25.
    state_path : str or None, optional
        Optional live state JSON path. If None, a default file is created in
        ``output_dir``.

    Returns
    -------
    dict
        Manifest payload summarizing inputs, outputs, parallel settings,
        counts, timing, and failures.

    Raises
    ------
    SystemExit
        If no CIF files are found.
    RuntimeError
        If ``fail_fast`` is True and at least one CIF fails.
    KeyboardInterrupt
        Re-raised after writing an interrupted state checkpoint.
    """
    input_dir = resolve_path(input_dir)
    output_dir = ensure_dir(output_dir)
    cif_paths = collect_cif_paths(input_dir, recursive=recursive, pattern=pattern)
    if not cif_paths:
        raise SystemExit(
            f"No CIF files found in {input_dir} with pattern '{pattern}' (recursive={bool(recursive)})."
        )

    manifest_path = resolve_path(manifest_path) if manifest_path else os.path.join(
        output_dir, "graphlet_build_manifest.json"
    )
    progress_log = resolve_path(progress_log) if progress_log else None
    state_path = resolve_path(state_path) if state_path else os.path.join(
        output_dir, "graphlet_build_state.json"
    )
    logger = ProgressLogger(progress_log)

    logger.log(f"Input dir: {input_dir}")
    logger.log(f"Output dir: {output_dir}")
    logger.log(f"CIF files discovered: {len(cif_paths)}")

    cpu_count = max(1, int(os.cpu_count() or 1))
    max_workers_req = max(1, int(max_workers))
    cpu_cap = max(1, int(cpu_cap))
    max_workers_eff = min(max_workers_req, cpu_cap, cpu_count)
    monitor_interval = max(5.0, float(monitor_interval))
    checkpoint_every = max(1, int(checkpoint_every))
    if max_in_flight is None:
        max_in_flight_eff = max(max_workers_eff, max_workers_eff * 3)
    else:
        max_in_flight_eff = max(max_workers_eff, int(max_in_flight))

    logger.log(
        "Parallel settings: "
        f"requested_workers={max_workers_req}, effective_workers={max_workers_eff}, "
        f"cpu_cap={cpu_cap}, cpu_count={cpu_count}, max_in_flight={max_in_flight_eff}"
    )
    logger.log(
        f"Monitoring settings: monitor_interval={monitor_interval:.1f}s, "
        f"checkpoint_every={checkpoint_every}, state_path={state_path}"
    )

    built = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    graphlet_paths: list[str] = []
    submitted = 0
    completed_from_workers = 0
    aborted_due_to_fail_fast = False

    input_root = Path(input_dir)
    total = len(cif_paths)
    jobs: list[dict[str, str]] = []
    for cif_path in cif_paths:
        cif_obj = Path(cif_path)
        rel = cif_obj.relative_to(input_root)
        out_dir = Path(output_dir) / rel.parent
        out_path = out_dir / f"{cif_obj.stem}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            skipped += 1
            graphlet_paths.append(str(out_path))
        else:
            jobs.append(
                {
                    "cif_path": str(cif_obj),
                    "out_path": str(out_path),
                }
            )

    jobs_total = len(jobs)
    start_epoch = time.time()

    def _build_state_payload(
        *,
        status: str,
        running: int,
        queued_not_submitted: int,
    ) -> Dict[str, Any]:
        """
        Build the live state payload for a folder graphlet build.

        Parameters
        ----------
        status : str
            Current workflow status label.
        running : int
            Number of tasks currently submitted and unfinished.
        queued_not_submitted : int
            Number of discovered jobs not yet submitted to the process pool.

        Returns
        -------
        dict
            State payload suitable for writing to ``state_path``.
        """
        completed_total = built + skipped + failed
        pending_total = max(0, total - completed_total)
        elapsed_s = time.time() - start_epoch
        rate_items_per_min = (completed_total * 60.0 / elapsed_s) if elapsed_s > 0 else 0.0
        eta_s = (pending_total / (completed_total / elapsed_s)) if completed_total > 0 and elapsed_s > 0 else None
        return {
            "status": status,
            "last_update": timestamp(),
            "input_dir": input_dir,
            "output_dir": output_dir,
            "pattern": pattern,
            "recursive": bool(recursive),
            "suffix": suffix,
            "overwrite": bool(overwrite),
            "include_compact_features": bool(include_compact_features),
            "counts": {
                "num_input_cifs": total,
                "num_jobs_total": jobs_total,
                "num_graphlets_built": built,
                "num_graphlets_skipped": skipped,
                "num_graphlets_failed": failed,
                "num_completed_total": completed_total,
                "num_pending_total": pending_total,
                "num_submitted": submitted,
                "num_completed_from_workers": completed_from_workers,
            },
            "parallel": {
                "cpu_count": cpu_count,
                "cpu_cap": cpu_cap,
                "max_workers_requested": max_workers_req,
                "max_workers_effective": max_workers_eff,
                "max_in_flight": max_in_flight_eff,
                "running_now": int(running),
                "queued_not_submitted": int(queued_not_submitted),
            },
            "timing": {
                "elapsed_seconds": elapsed_s,
                "elapsed_human": _format_duration(elapsed_s),
                "rate_items_per_min": rate_items_per_min,
                "eta_seconds": eta_s,
                "eta_human": _format_duration(eta_s),
            },
            "manifest_path": manifest_path,
            "progress_log": progress_log,
            "state_path": state_path,
            "recent_failures": failures[-50:],
            "aborted_due_to_fail_fast": bool(aborted_due_to_fail_fast),
        }

    def _write_state(
        *,
        status: str,
        running: int,
        queued_not_submitted: int,
    ) -> None:
        """
        Write the current folder-build state to disk atomically.

        Parameters
        ----------
        status : str
            Current workflow status label.
        running : int
            Number of tasks currently submitted and unfinished.
        queued_not_submitted : int
            Number of discovered jobs not yet submitted to the process pool.

        Returns
        -------
        None
        """
        payload = _build_state_payload(
            status=status,
            running=running,
            queued_not_submitted=queued_not_submitted,
        )
        write_json_atomic(state_path, payload, indent=2)

    _write_state(status="starting", running=0, queued_not_submitted=jobs_total)

    if jobs_total == 0:
        logger.log("All requested graphlet outputs already exist; nothing to submit.")
    else:
        logger.log(f"Submitting {jobs_total} CIF jobs to process pool.")

        futures: dict[cf.Future, dict[str, str]] = {}
        next_job_idx = 0
        stop_submission = False

        def _submit_more(executor: cf.ProcessPoolExecutor) -> None:
            """
            Submit queued graphlet jobs until the in-flight limit is reached.

            Parameters
            ----------
            executor : concurrent.futures.ProcessPoolExecutor
                Process pool used to execute graphlet build workers.

            Returns
            -------
            None
            """
            nonlocal next_job_idx
            nonlocal submitted
            while (
                (not stop_submission)
                and next_job_idx < jobs_total
                and len(futures) < max_in_flight_eff
            ):
                job = jobs[next_job_idx]
                future = executor.submit(
                    _graphlet_build_worker,
                    cif_path=job["cif_path"],
                    out_path=job["out_path"],
                    atomic_radii_path=atomic_radii_path,
                    atomic_features_path=atomic_features_path,
                    include_compact_features=include_compact_features,
                )
                futures[future] = job
                next_job_idx += 1
                submitted += 1

        try:
            with cf.ProcessPoolExecutor(max_workers=max_workers_eff) as executor:
                _submit_more(executor)
                _write_state(
                    status="running",
                    running=len(futures),
                    queued_not_submitted=max(0, jobs_total - next_job_idx),
                )

                while futures:
                    done, _ = cf.wait(
                        list(futures.keys()),
                        timeout=monitor_interval,
                        return_when=cf.FIRST_COMPLETED,
                    )

                    if not done:
                        completed_total = built + skipped + failed
                        pending_total = max(0, total - completed_total)
                        elapsed_s = time.time() - start_epoch
                        rate = (completed_total * 60.0 / elapsed_s) if elapsed_s > 0 else 0.0
                        eta_s = (
                            pending_total / (completed_total / elapsed_s)
                            if completed_total > 0 and elapsed_s > 0
                            else None
                        )
                        logger.log(
                            "Heartbeat: "
                            f"completed={completed_total}/{total}, built={built}, skipped={skipped}, failed={failed}, "
                            f"running={len(futures)}, queued={max(0, jobs_total - next_job_idx)}, "
                            f"rate={rate:.2f}/min, eta={_format_duration(eta_s)}"
                        )
                        _write_state(
                            status="running",
                            running=len(futures),
                            queued_not_submitted=max(0, jobs_total - next_job_idx),
                        )
                        continue

                    for future in done:
                        job = futures.pop(future)
                        completed_from_workers += 1

                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover - worker transport failure
                            result = {
                                "ok": False,
                                "cif_path": job["cif_path"],
                                "out_path": job["out_path"],
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }

                        if result.get("ok"):
                            built += 1
                            graphlet_paths.append(str(result.get("out_path", job["out_path"])))
                        else:
                            failed += 1
                            failure = {
                                "cif_path": str(result.get("cif_path", job["cif_path"])),
                                "error_type": str(result.get("error_type", "RuntimeError")),
                                "error": str(result.get("error", "Unknown error")),
                            }
                            failures.append(failure)
                            logger.log(
                                "FAILED "
                                f"{failure['cif_path']}: {failure['error_type']}: {failure['error']}"
                            )
                            if fail_fast:
                                stop_submission = True
                                aborted_due_to_fail_fast = True

                        completed_total = built + skipped + failed
                        if (
                            completed_total == 1
                            or completed_total == total
                            or completed_total % max(1, progress_every) == 0
                        ):
                            logger.log(
                                f"{progress_step(completed_total, total, 'Graphlets')} "
                                f"(built={built}, skipped={skipped}, failed={failed}, "
                                f"running={len(futures)}, queued={max(0, jobs_total - next_job_idx)})"
                            )

                        if completed_from_workers % checkpoint_every == 0:
                            _write_state(
                                status="running",
                                running=len(futures),
                                queued_not_submitted=max(0, jobs_total - next_job_idx),
                            )

                    _submit_more(executor)

        except KeyboardInterrupt:  # pragma: no cover - interactive interruption path
            logger.log("Interrupted by user. Writing state checkpoint for resume.")
            _write_state(
                status="interrupted",
                running=0,
                queued_not_submitted=max(0, jobs_total - next_job_idx),
            )
            raise

    elapsed_total_s = time.time() - start_epoch
    completed_total = built + skipped + failed
    pending_total = max(0, total - completed_total)
    throughput = (completed_total * 60.0 / elapsed_total_s) if elapsed_total_s > 0 else 0.0
    graphlet_paths = sorted(set(graphlet_paths))

    manifest = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "pattern": pattern,
        "recursive": bool(recursive),
        "suffix": suffix,
        "overwrite": bool(overwrite),
        "include_compact_features": bool(include_compact_features),
        "atomic_radii_path": resolve_path(atomic_radii_path),
        "atomic_features_path": resolve_path(atomic_features_path),
        "num_input_cifs": total,
        "num_jobs_total": jobs_total,
        "num_graphlets_built": built,
        "num_graphlets_skipped": skipped,
        "num_graphlets_failed": failed,
        "num_completed_total": completed_total,
        "num_pending_total": pending_total,
        "throughput_items_per_min": throughput,
        "elapsed_seconds": elapsed_total_s,
        "elapsed_human": _format_duration(elapsed_total_s),
        "aborted_due_to_fail_fast": bool(aborted_due_to_fail_fast),
        "cpu_count": cpu_count,
        "cpu_cap": cpu_cap,
        "max_workers_requested": max_workers_req,
        "max_workers_effective": max_workers_eff,
        "max_in_flight": max_in_flight_eff,
        "monitor_interval": monitor_interval,
        "checkpoint_every": checkpoint_every,
        "graphlet_paths": graphlet_paths,
        "failures": failures,
        "manifest_path": manifest_path,
        "progress_log": progress_log,
        "state_path": state_path,
    }

    write_json_atomic(manifest_path, manifest, indent=2)
    final_status = "completed" if pending_total == 0 else "partial"
    _write_state(status=final_status, running=0, queued_not_submitted=0)
    logger.log(f"Wrote manifest to {manifest_path}")
    logger.log(
        "Graphlet folder build completed "
        f"(built={built}, skipped={skipped}, failed={failed}, pending={pending_total})"
    )

    if fail_fast and aborted_due_to_fail_fast:
        raise RuntimeError(
            "Fail-fast triggered after at least one CIF failure. "
            f"See manifest/state for details: {manifest_path}"
        )

    return manifest


def _collect_cif_paths_from_csv(csv_path: str, cif_column: str = "cif") -> tuple[list[str], list[str]]:
    """
    Collect existing and missing CIF paths from a CSV column.

    Parameters
    ----------
    csv_path : str
        CSV file containing CIF path rows.
    cif_column : str, optional
        Column name containing CIF paths. Default is ``"cif"``.

    Returns
    -------
    existing : list of str
        Unique CIF paths that exist on disk.
    missing : list of str
        Unique CIF paths that were present in the CSV but missing on disk.
    """
    existing: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    with open(resolve_path(csv_path), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cif_path = (row.get(cif_column) or "").strip()
            if not cif_path or cif_path in seen:
                continue
            seen.add(cif_path)
            if Path(cif_path).exists():
                existing.append(cif_path)
            else:
                missing.append(cif_path)
    return existing, missing


def run_csv_graphlet_histogram_build(
    *,
    csv_path: str,
    cif_column: str = "cif",
    out_root: str,
    graphlet_dir_name: str = "CSV_Graphlets",
    histogram_dir_name: str = "Classification_Histograms",
    bin_centers_name: str = "csv_dynamic_bin_centers.json",
    manifest_name: str = "classification_histogram_manifest.json",
    progress_log_name: str = "classification_histogram_progress.log",
    missing_name: str = "csv_missing_cifs.txt",
    num_bins: int = 20,
    hist_density: bool = False,
    recompute_bins: bool = False,
) -> Dict[str, Any]:
    """
    Run a CSV-driven graphlet and histogram workflow.

    Parameters
    ----------
    csv_path : str
        CSV file containing a column of CIF paths.
    cif_column : str, optional
        Column containing CIF paths. Default is ``"cif"``.
    out_root : str
        Root output directory for graphlets, histograms, bin centers, logs, and
        manifests.
    graphlet_dir_name : str, optional
        Subdirectory name for graphlet JSON files.
    histogram_dir_name : str, optional
        Subdirectory name for histogram JSON files.
    bin_centers_name : str, optional
        Filename for the dynamic bin-center JSON.
    manifest_name : str, optional
        Filename for the workflow manifest.
    progress_log_name : str, optional
        Filename for the progress log.
    missing_name : str, optional
        Filename for the missing-CIF list.
    num_bins : int, optional
        Number of bins to derive for each feature. Default is 20.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    recompute_bins : bool, optional
        If True, recompute bin centers even when the bin-center file already
        exists. Default is False.

    Returns
    -------
    dict
        Manifest payload summarizing input counts, output paths, binning
        settings, and generated artifact counts.

    Raises
    ------
    SystemExit
        If the CSV contains no existing CIF paths in ``cif_column``.
    """
    csv_path = resolve_path(csv_path)
    out_root = ensure_dir(out_root)
    graphlet_dir = ensure_dir(os.path.join(out_root, graphlet_dir_name))
    histogram_dir = ensure_dir(os.path.join(out_root, histogram_dir_name))
    bin_centers_path = resolve_path(os.path.join(out_root, bin_centers_name))
    manifest_path = resolve_path(os.path.join(out_root, manifest_name))
    missing_path = resolve_path(os.path.join(out_root, missing_name))
    progress_log_path = resolve_path(os.path.join(out_root, progress_log_name))
    logger = ProgressLogger(progress_log_path)

    cif_paths, missing_paths = _collect_cif_paths_from_csv(csv_path, cif_column=cif_column)
    if not cif_paths:
        raise SystemExit(
            f"No existing CIF paths were found in column '{cif_column}' of {csv_path}"
        )

    logger.log(f"Starting CSV graphlet workflow from {csv_path}")
    logger.log(f"Found {len(cif_paths)} existing CIFs from CSV")
    logger.log(f"Missing CIF paths: {len(missing_paths)}")

    graphlet_paths = []
    total_cifs = len(cif_paths)
    for idx, cif_path in enumerate(cif_paths, start=1):
        stem = Path(cif_path).stem
        out_path = os.path.join(graphlet_dir, f"{stem}_graphlet.json")
        build_and_save_graphlet_json(
            cif_path,
            out_path,
            include_compact_features=True,
        )
        graphlet_paths.append(out_path)
        if idx == 1 or idx == total_cifs or idx % 25 == 0:
            logger.log(progress_step(idx, total_cifs, "Graphlets"))
    logger.log(f"Wrote {len(graphlet_paths)} graphlet JSON files to {graphlet_dir}")

    if recompute_bins or not os.path.exists(bin_centers_path):
        logger.log("Deriving dynamic bin centers")
        bin_payload = derive_dynamic_bin_centers(
            graphlet_paths,
            out_path=bin_centers_path,
            num_bins=num_bins,
        )
        logger.log(f"Wrote dynamic bin centers to {bin_centers_path}")
    else:
        with open(bin_centers_path, "r") as f:
            bin_payload = json.load(f)
        logger.log(f"Reused existing bin centers from {bin_centers_path}")

    logger.log("Building histogram JSON files")
    bin_center_payload = load_predefined_bin_centers(bin_centers_path, fmt="auto")
    histogram_paths = []
    total_graphlets = len(graphlet_paths)
    for idx, graphlet_path in enumerate(graphlet_paths, start=1):
        stem = Path(graphlet_path).stem
        out_path = os.path.join(histogram_dir, f"{stem}_histogram.json")
        histogram_compact_feature_json(
            graphlet_path,
            bin_center_payload,
            out_path=out_path,
            hist_density=hist_density,
            strict=False,
        )
        histogram_paths.append(out_path)
        if idx == 1 or idx == total_graphlets or idx % 25 == 0:
            logger.log(progress_step(idx, total_graphlets, "Histograms"))
    logger.log(f"Wrote {len(histogram_paths)} histogram JSON files to {histogram_dir}")

    if missing_paths:
        with open(missing_path, "w") as f:
            f.write("\n".join(missing_paths) + "\n")
        logger.log(f"Wrote missing CIF list to {missing_path}")

    manifest = {
        "csv_path": csv_path,
        "cif_column": cif_column,
        "out_root": out_root,
        "graphlet_dir": graphlet_dir,
        "histogram_dir": histogram_dir,
        "bin_centers_path": bin_centers_path,
        "progress_log_path": progress_log_path,
        "num_existing_cifs_used": len(cif_paths),
        "num_missing_cifs": len(missing_paths),
        "num_graphlet_jsons": len(graphlet_paths),
        "num_histogram_jsons": len(histogram_paths),
        "num_bins": int(num_bins),
        "hist_density": bool(hist_density),
        "num_features": len(bin_payload.get("feature_names", [])),
    }
    write_json(manifest_path, manifest, indent=2)

    logger.log(f"Wrote manifest to {manifest_path}")
    logger.log("Workflow completed successfully")
    return manifest


def run_graphlet_pipeline(
    *,
    input_dir: str,
    out_root: str,
    pattern: str = "*.cif",
    recursive: bool = False,
    graphlet_dir_name: str = "graphlets",
    histogram_dir_name: str = "histograms",
    bin_centers_name: str = "bin_centers.json",
    manifest_name: str = "pipeline_manifest.json",
    reference_graphlet_paths: Sequence[str] | None = None,
    reference_limit: int | None = None,
    num_bins: int = 20,
    bin_width_factor: float = 1.0,
    hist_density: bool = False,
    recompute_bins: bool = False,
    overwrite_graphlets: bool = False,
    fail_fast: bool = False,
    max_workers: int = 20,
    cpu_cap: int = 20,
) -> Dict[str, Any]:
    """
    Run the user-facing CIF-to-histogram pipeline.

    This helper performs the recommended staged workflow in one call:
    graphlet JSON generation, shared bin-center creation or reuse, and
    histogram JSON generation.

    Parameters
    ----------
    input_dir : str
        Directory containing input CIF files.
    out_root : str
        Root directory for all generated artifacts.
    pattern : str, optional
        Glob pattern used to discover CIF files. Default is ``"*.cif"``.
    recursive : bool, optional
        If True, discover CIF files recursively. Default is False.
    graphlet_dir_name : str, optional
        Subdirectory name under ``out_root`` for graphlet JSON files.
    histogram_dir_name : str, optional
        Subdirectory name under ``out_root`` for histogram JSON files.
    bin_centers_name : str, optional
        Filename under ``out_root`` for the shared bin-center JSON.
    manifest_name : str, optional
        Filename under ``out_root`` for the pipeline manifest.
    reference_graphlet_paths : sequence of str or None, optional
        Optional graphlet JSON paths used to derive bin centers. If None, all
        graphlets from the current build are used, subject to
        ``reference_limit``.
    reference_limit : int or None, optional
        Optional deterministic limit on the number of graphlet files used for
        bin-center derivation. Applied only when ``reference_graphlet_paths`` is
        None.
    num_bins : int, optional
        Number of bins to derive for each feature. Default is 20.
    bin_width_factor : float, optional
        Scale factor for dynamic bin range estimation. Default is 1.0.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    recompute_bins : bool, optional
        If True, recompute bin centers even when the bin-center file exists.
        Default is False.
    overwrite_graphlets : bool, optional
        If True, rebuild graphlet JSON files that already exist. Default is
        False.
    fail_fast : bool, optional
        If True, stop after the first CIF failure during graphlet generation.
        Default is False.
    max_workers : int, optional
        Requested process-pool worker count for graphlet generation. Default
        is 20.
    cpu_cap : int, optional
        Hard upper bound on graphlet worker count. Default is 20.

    Returns
    -------
    dict
        Pipeline manifest with artifact paths, counts, and binning metadata.

    Raises
    ------
    SystemExit
        If no CIF files are found or no reference graphlets are available.
    """
    out_root = ensure_dir(out_root)
    graphlet_dir = ensure_dir(os.path.join(out_root, graphlet_dir_name))
    histogram_dir = ensure_dir(os.path.join(out_root, histogram_dir_name))
    bin_centers_path = resolve_path(os.path.join(out_root, bin_centers_name))
    manifest_path = resolve_path(os.path.join(out_root, manifest_name))

    graphlet_manifest = run_folder_graphlet_build(
        input_dir=input_dir,
        output_dir=graphlet_dir,
        pattern=pattern,
        recursive=recursive,
        overwrite=overwrite_graphlets,
        fail_fast=fail_fast,
        max_workers=max_workers,
        cpu_cap=cpu_cap,
    )
    graphlet_paths = list(graphlet_manifest["graphlet_paths"])

    if reference_graphlet_paths is not None:
        reference_paths = [resolve_path(path) for path in reference_graphlet_paths]
    else:
        reference_paths = list(graphlet_paths)
        if reference_limit is not None:
            reference_paths = reference_paths[: max(0, int(reference_limit))]

    if not reference_paths:
        raise SystemExit("No reference graphlet JSON files are available for bin-center derivation.")

    if recompute_bins or not os.path.exists(bin_centers_path):
        bin_payload = derive_dynamic_bin_centers(
            reference_paths,
            out_path=bin_centers_path,
            num_bins=num_bins,
            bin_width_factor=bin_width_factor,
        )
        bin_source = "derived"
    else:
        bin_payload = load_predefined_bin_centers(bin_centers_path, fmt="auto")
        bin_source = "reused"

    histogram_paths = batch_histogram_compact_feature_jsons(
        graphlet_paths,
        bin_centers_path=bin_centers_path,
        histogram_out_dir=histogram_dir,
        hist_density=hist_density,
        strict=False,
        prefer_existing_bins=True,
        num_bins=num_bins,
        bin_width_factor=bin_width_factor,
    )

    manifest = {
        "input_dir": resolve_path(input_dir),
        "out_root": out_root,
        "graphlet_dir": graphlet_dir,
        "histogram_dir": histogram_dir,
        "bin_centers_path": bin_centers_path,
        "manifest_path": manifest_path,
        "graphlet_manifest_path": graphlet_manifest["manifest_path"],
        "graphlet_state_path": graphlet_manifest["state_path"],
        "num_input_cifs": graphlet_manifest["num_input_cifs"],
        "num_graphlet_jsons": len(graphlet_paths),
        "num_reference_graphlets": len(reference_paths),
        "num_histogram_jsons": len(histogram_paths),
        "num_bins": int(num_bins),
        "num_histogram_channels": len(bin_payload.get("feature_names", [])),
        "hist_density": bool(hist_density),
        "bin_centers_source": bin_source,
        "graphlet_paths": graphlet_paths,
        "reference_graphlet_paths": reference_paths,
        "histogram_paths": histogram_paths,
    }
    write_json(manifest_path, manifest, indent=2)
    return manifest


__all__ = [
    # utils
    "resolve_path",
    "ensure_dir",
    "timestamp",
    "ProgressLogger",
    "progress_step",
    "collect_paths",
    "collect_cif_paths",
    "collect_graphlet_json_paths",
    "write_json",
    "write_json_atomic",
    # graphlet featurization
    "build_compact_feature_payload_from_cif",
    "build_graphlet_payload_from_cif",
    "build_and_save_graphlet_json",
    "batch_build_graphlet_jsons",
    # histogram workflow
    "derive_dynamic_bin_centers",
    "load_predefined_bin_centers",
    "ensure_bin_centers",
    "histogram_compact_feature_payload",
    "histogram_compact_feature_json",
    "batch_histogram_compact_feature_jsons",
    "run_graphlet_histogram_build",
    "run_full_graphlet_histogram_workflow",
    # batch runners
    "run_graphlet_pipeline",
    "run_folder_graphlet_build",
    "run_csv_graphlet_histogram_build",
]
