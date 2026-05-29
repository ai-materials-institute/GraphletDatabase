"""Adapters from FinalFeaturization histogram payloads into EMD-ready tensors."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from graphlet_core import (
    DEFAULT_ATOMIC_FEATURES_JSON,
    DEFAULT_ATOMIC_RADII_JSON,
    DEFAULT_CLASSIFICATION_BIN_PICKLE,
    build_compact_feature_payload_from_cif,
    histogram_compact_feature_payload,
    load_predefined_bin_centers,
    resolve_path,
    write_json,
)


def load_histogram_payload(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load one histogram payload from JSON or pickle."""
    path = resolve_path(path)
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def histogram_payload_to_tensor(payload: Dict[str, Any], *, key: str = "histogram_features") -> np.ndarray:
    """Extract the `(n_histograms, n_bins, 2)` histogram tensor from a payload."""
    hist = np.asarray(payload[key], dtype=float)
    if hist.ndim == 4:
        hist = hist[0]
    if hist.ndim != 3 or hist.shape[-1] != 2:
        raise ValueError("Histogram payload must resolve to shape `(n_histograms, n_bins, 2)`.")
    return hist


def build_emd_histogram_payload_from_cif(
    cif_path: str,
    *,
    bin_centers_path: str = DEFAULT_CLASSIFICATION_BIN_PICKLE,
    bin_centers_format: str = "auto",
    hist_density: bool = False,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
) -> Dict[str, Any]:
    """Build one histogram payload from a CIF in the format expected by the EMD utility."""
    cif_path = resolve_path(cif_path)
    compact_payload = build_compact_feature_payload_from_cif(
        cif_path,
        atomic_radii_path=atomic_radii_path,
        atomic_features_path=atomic_features_path,
    )
    bin_payload = load_predefined_bin_centers(bin_centers_path, fmt=bin_centers_format)
    hist_payload = histogram_compact_feature_payload(
        compact_payload,
        bin_payload,
        hist_density=hist_density,
    )
    hist_payload["cif_path"] = cif_path
    hist_payload["bin_centers_path"] = resolve_path(bin_centers_path)
    return hist_payload


def build_emd_histogram_payloads_for_cifs(
    cif_paths: Sequence[str] | Iterable[str],
    *,
    out_dir: str,
    suffix: str = "_emd_histogram.json",
    **kwargs: Any,
) -> list[str]:
    """Build EMD-ready histogram JSON payloads for many CIF files."""
    out_dir = resolve_path(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    for cif_path in cif_paths:
        cif_path = resolve_path(cif_path)
        stem = Path(cif_path).stem
        out_path = os.path.join(out_dir, f"{stem}{suffix}")
        payload = build_emd_histogram_payload_from_cif(cif_path, **kwargs)
        write_json(out_path, payload, indent=2)
        out_paths.append(out_path)
    return out_paths
