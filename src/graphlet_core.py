#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central graphlet/symmetry/histogram implementation.

This module is the single source-of-truth for:
- graphlet creation from CIF files
- symmetry feature extraction from CIF files
- dynamic/fixed-bin histogram creation from graphlet JSONs
- batch runner helpers used by CLI entrypoints
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from PYGraphlets import Create_Graphlets


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

DEFAULT_SPACEGROUP_XLS = os.path.join(CONFIG_DIR, "Space_group.xls")
DEFAULT_SPACEGROUP_SHEET = "Sheet3"

DEFAULT_ATOMIC_RADII_JSON = os.path.join(CONFIG_DIR, "atomic_radii.json")
DEFAULT_ATOMIC_FEATURES_JSON = os.path.join(CONFIG_DIR, "Filtered_atomic_features.json")

DEFAULT_CLASSIFICATION_BIN_PICKLE = os.path.join(CONFIG_DIR, "bin_centers_classification.pkl")
DEFAULT_REGRESSION_BIN_PICKLE = os.path.join(CONFIG_DIR, "bin_centers_regression.pkl")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def resolve_path(path: str | os.PathLike[str]) -> str:
    """Return absolute path with ~ and env vars expanded."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def ensure_dir(path: str | os.PathLike[str]) -> str:
    """Create directory and return absolute path."""
    out = resolve_path(path)
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(resolve_path(path)).parent.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Return local wall-clock timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ProgressLogger:
    """Minimal stdout + optional file logger for long-running jobs."""

    def __init__(self, log_path: str | os.PathLike[str] | None = None):
        self.log_path = resolve_path(log_path) if log_path else None
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")


def progress_step(index: int, total: int, label: str) -> str:
    return f"{label} {index}/{total}"


def collect_paths(
    root_dir: str | os.PathLike[str],
    *,
    pattern: str,
    recursive: bool = False,
    files_only: bool = True,
) -> list[str]:
    """Collect matching paths under root_dir, sorted deterministically."""
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
    """Collect CIF file paths under root_dir."""
    return collect_paths(root_dir, pattern=pattern, recursive=recursive, files_only=True)


def write_json(path: str | os.PathLike[str], payload: dict, *, indent: int = 2) -> str:
    """Write JSON payload and return absolute output path."""
    out_path = resolve_path(path)
    _ensure_parent(out_path)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=indent)
    return out_path


def write_json_atomic(path: str | os.PathLike[str], payload: dict, *, indent: int = 2) -> str:
    """Atomically write JSON payload and return absolute output path."""
    out_path = resolve_path(path)
    _ensure_parent(out_path)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp_path, out_path)
    return out_path


def _load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def _format_duration(seconds: float | None) -> str:
    """Render a compact human-readable duration string."""
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
# Symmetry extraction
# ---------------------------------------------------------------------------

class Config:
    """Default symmetry mapping configuration."""

    EXCEL_MAPPING_XLS: str = os.path.expanduser(
        os.getenv("EXCEL_MAPPING_XLS", DEFAULT_SPACEGROUP_XLS)
    )
    EXCEL_SHEET: str = os.getenv("EXCEL_SHEET", DEFAULT_SPACEGROUP_SHEET)


class _SymUtil:
    """Internal helper utilities for symmetry extraction."""

    @staticmethod
    def resolve_path(p: str | os.PathLike[str] | None) -> str | None:
        if p is None:
            return None
        return resolve_path(p)

    @staticmethod
    def sheet_arg(sheet: Union[str, int]) -> Union[str, int]:
        if isinstance(sheet, int):
            return sheet
        s = str(sheet).strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                return sheet
        return sheet


class SymmetryFeatureExtractor:
    """
    Compute averaged symmetry feature vectors from CIFs using a point-group map.
    """

    def __init__(
        self,
        excel_file: str = Config.EXCEL_MAPPING_XLS,
        sheet_name: Union[str, int] = Config.EXCEL_SHEET,
        symbol_col_index: int = 1,
        first_feat_col_index: int = 2,
    ):
        df = pd.read_excel(
            _SymUtil.resolve_path(excel_file),
            sheet_name=_SymUtil.sheet_arg(sheet_name),
            header=0,
        )

        self._feature_names: List[Any] = df.columns[first_feat_col_index:].tolist()
        symbols = [str(x) for x in df.iloc[:, symbol_col_index].tolist()]
        rows = df.iloc[:, first_feat_col_index:].values.tolist()
        self._pg_feature_map: Dict[str, List[float]] = {s: r for s, r in zip(symbols, rows)}

    def from_cif(
        self,
        cif_path: str,
        *,
        symprec: float = 1e-5,
        tol: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Compute averaged symmetry feature vector for a CIF.

        Returns
        -------
        dict
            {"feature_names": list, "feature_values": np.ndarray}
        """
        struct = PMGStructure.from_file(_SymUtil.resolve_path(cif_path))
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        sym_ops = sga.get_symmetry_operations()

        n_feat = len(self._feature_names)
        site_feature_list: List[List[float]] = []

        for site in struct:
            fc = site.frac_coords % 1.0
            ops_fix = [op for op in sym_ops if np.allclose(op.operate(fc) % 1.0, fc, atol=tol)]

            if not ops_fix:
                ptg_symbol = "1"
            else:
                rot_mats = [np.rint(op.rotation_matrix).astype(int) for op in ops_fix]
                ptg_symbol, _, _ = spglib.get_pointgroup(rot_mats)

            site_feature_list.append(self._pg_feature_map.get(ptg_symbol, [0.0] * n_feat))

        avg = np.asarray(site_feature_list, dtype=float).mean(axis=0) if site_feature_list else np.zeros(n_feat)
        return {"feature_names": self._feature_names, "feature_values": avg}


def build_symmetry_feature_payload_from_cif(
    cif_path: str,
    *,
    excel_file: str = DEFAULT_SPACEGROUP_XLS,
    sheet_name: str = DEFAULT_SPACEGROUP_SHEET,
) -> Dict[str, Any]:
    """Build JSON-safe symmetry features for one CIF file."""
    cif_path = resolve_path(cif_path)
    extractor = SymmetryFeatureExtractor(
        excel_file=resolve_path(excel_file),
        sheet_name=sheet_name,
    )
    payload = extractor.from_cif(cif_path)
    return {
        "symmetry_feature": np.asarray(payload["feature_values"], dtype=float).tolist(),
        "symmetry_feature_names": list(payload["feature_names"]),
    }


# ---------------------------------------------------------------------------
# Graphlet and histogram workflow
# ---------------------------------------------------------------------------

def _load_atomic_data(
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(resolve_path(atomic_radii_path), "r") as f:
        atomic_radii = json.load(f)
    with open(resolve_path(atomic_features_path), "r") as f:
        atomic_features_dict = json.load(f)
    return atomic_radii, atomic_features_dict


def _iter_feature_items(compact_payload: Dict[str, Any]) -> Iterable[Tuple[str, List[List[float]]]]:
    feature_groups = compact_payload.get("raw_features_counts", {})
    for _group_name, feat_dict in feature_groups.items():
        for feat_name, pairs in feat_dict.items():
            yield feat_name, pairs


def _pairs_to_arrays(pairs: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Each feature entry must be a list of [value, count] pairs.")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def _expand_pairs(pairs: Sequence[Sequence[float]]) -> np.ndarray:
    values, counts = _pairs_to_arrays(pairs)
    if values.size == 0:
        return np.array([], dtype=float)
    return np.repeat(values, counts.astype(int))


def _pygraphlets_bin_width(values: np.ndarray, bin_width_factor: float = 1.0) -> float:
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
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot derive edges from an empty feature value list.")

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    bin_width = _pygraphlets_bin_width(values, bin_width_factor=bin_width_factor)
    return np.arange(min_val - bin_width / 2, max_val + 3 * bin_width / 2, bin_width, dtype=float)


def _fixed_count_edges_from_outer_range(outer_edges: np.ndarray, num_bins: int = 20) -> np.ndarray:
    outer_edges = np.asarray(outer_edges, dtype=float)
    if outer_edges.size < 2:
        raise ValueError("At least two outer edges are required.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")
    return np.linspace(float(outer_edges[0]), float(outer_edges[-1]), num_bins + 1, dtype=float)


def _edges_to_centers(edges: np.ndarray) -> np.ndarray:
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
    """Build compact graphlet feature payload (counts) from one CIF file."""
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
    """Build full graphlet JSON payload from one CIF file."""
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
    """Build and write graphlet JSON payload for one CIF."""
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
    suffix: str = "_graphlets.json",
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
    include_compact_features: bool = True,
) -> List[str]:
    """Build graphlet JSON files for a list of CIF paths."""
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
) -> Dict[str, Any]:
    """Derive dynamic bin centers from compact graphlet JSON files."""
    aggregated: Dict[str, List[np.ndarray]] = {}
    for path in compact_json_paths:
        payload = _load_json(path)
        for feat_name, pairs in _iter_feature_items(payload):
            expanded = _expand_pairs(pairs)
            aggregated.setdefault(feat_name, []).append(expanded)

    feature_names = sorted(aggregated.keys())
    bin_centers: Dict[str, List[float]] = {}
    bin_edges: Dict[str, List[float]] = {}
    outer_ranges: Dict[str, List[float]] = {}

    for feat_name in feature_names:
        values = np.concatenate(aggregated[feat_name]) if aggregated[feat_name] else np.array([], dtype=float)
        outer_edges = _pygraphlets_outer_edges(values, bin_width_factor=bin_width_factor)
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
    return payload


def load_predefined_bin_centers(path: str, *, fmt: str = "auto") -> Dict[str, Any]:
    """Load bin centers from JSON or legacy pickle format."""
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
    """Load bin centers from disk or derive+save dynamic bin centers."""
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
    """Convert one compact graphlet payload into histogram features."""
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
    """Load one compact graphlet JSON and write/return histogram payload."""
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
    """End-to-end histogram workflow from graphlet JSON files."""
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
    """Run CIF -> graphlet JSON -> histogram JSON workflow."""
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
    """Worker for one CIF -> graphlet JSON task."""
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
    """CSV-driven graphlet + histogram workflow."""
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


# ---------------------------------------------------------------------------
# Compact workflow CLI (keeps previous behavior)
# ---------------------------------------------------------------------------

def _parse_compact_cli_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of bins per feature. Default: 20.",
    )
    parser.add_argument(
        "--bin-width-factor",
        type=float,
        default=1.0,
        help="Bin-width scaling factor for the PYGraphlets-style outer range. Default: 1.0.",
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
    return parser.parse_args()


def _expand_cif_inputs(inputs: Sequence[str]) -> List[str]:
    cif_paths: List[str] = []
    for item in inputs:
        path = Path(resolve_path(item))
        if path.is_dir():
            cif_paths.extend(sorted(str(p) for p in path.glob("*.cif")))
        else:
            cif_paths.append(str(path))

    seen = set()
    ordered: List[str] = []
    for path in cif_paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def compact_workflow_cli_main() -> None:
    args = _parse_compact_cli_args()
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


__all__ = [
    # utils
    "resolve_path",
    "ensure_dir",
    "timestamp",
    "ProgressLogger",
    "progress_step",
    "collect_paths",
    "collect_cif_paths",
    "write_json",
    "write_json_atomic",
    # symmetry
    "Config",
    "SymmetryFeatureExtractor",
    "build_symmetry_feature_payload_from_cif",
    # workflow
    "build_compact_feature_payload_from_cif",
    "build_graphlet_payload_from_cif",
    "build_and_save_graphlet_json",
    "batch_build_graphlet_jsons",
    "derive_dynamic_bin_centers",
    "load_predefined_bin_centers",
    "ensure_bin_centers",
    "histogram_compact_feature_payload",
    "histogram_compact_feature_json",
    "batch_histogram_compact_feature_jsons",
    "run_full_graphlet_histogram_workflow",
    # runners
    "run_folder_graphlet_build",
    "run_csv_graphlet_histogram_build",
    # cli
    "compact_workflow_cli_main",
]
