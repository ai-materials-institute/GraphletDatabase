"""
Earth Mover's Distance utilities for graphlet histogram descriptors.

This module converts graphlet histogram payloads into tensor form, computes
Wasserstein-1 distances per histogram channel, reduces per-channel distances
to scalar material distances, and builds summed exponential kernels from EMD
distance matrices.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import ot
import torch

from core import (
    DEFAULT_ATOMIC_FEATURES_JSON,
    DEFAULT_ATOMIC_RADII_JSON,
    DEFAULT_CLASSIFICATION_BIN_PICKLE,
    build_compact_feature_payload_from_cif,
    histogram_compact_feature_payload,
    load_predefined_bin_centers,
    resolve_path,
    write_json,
)


# ---------------------------------------------------------------------------
# Histogram tensor utilities
# ---------------------------------------------------------------------------

def _as_hist_tensor(
    values: torch.Tensor | np.ndarray | Sequence[float],
    *,
    data_shape: tuple[int, int, int] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Coerce input into histogram tensor form.

    Parameters
    ----------
    values : torch.Tensor, numpy.ndarray, or sequence of float
        Histogram values. Accepted shapes are ``(n_samples, n_histograms,
        n_bins, 2)`` or ``(n_histograms, n_bins, 2)``. Flattened values can be
        supplied with ``data_shape``.
    data_shape : tuple of int or None, optional
        Shape ``(n_histograms, n_bins, 2)`` used to reshape flattened input.
    dtype : torch.dtype or None, optional
        Desired tensor dtype.
    device : torch.device, str, or None, optional
        Desired tensor device.

    Returns
    -------
    torch.Tensor
        Tensor shaped ``(n_samples, n_histograms, n_bins, 2)``.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as histogram tensors with the last
        axis storing ``(bin_center, height)``.
    """
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if data_shape is not None:
        tensor = tensor.reshape(-1, *data_shape)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != 2:
        raise ValueError(
            "Histogram input must have shape `(n_samples, n_histograms, n_bins, 2)` "
            "or be flattenable to that shape via `data_shape`."
        )
    return tensor


def valid_bin_lengths(histograms: torch.Tensor | np.ndarray, *, data_shape: tuple[int, int, int] | None = None) -> list[int]:
    """
    Return the non-padded bin count for each histogram channel.

    Parameters
    ----------
    histograms : torch.Tensor or numpy.ndarray
        Histogram tensor or array with padded bins encoded as ``-1``.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.

    Returns
    -------
    list of int
        Maximum valid bin count for each histogram channel.
    """
    histograms = _as_hist_tensor(histograms, data_shape=data_shape)
    lengths: list[int] = []
    for hist_idx in range(histograms.shape[1]):
        valid_bins = torch.sum(
            (histograms[:, hist_idx, :, 0] != -1.0) | (histograms[:, hist_idx, :, 1] != -1.0),
            dim=1,
        )
        lengths.append(int(valid_bins.max().item()))
    return lengths


def _normalize_counts(counts: torch.Tensor) -> torch.Tensor:
    """
    Normalize count vectors row-wise.

    Parameters
    ----------
    counts : torch.Tensor
        Count or mass tensor with bins on the last dimension.

    Returns
    -------
    torch.Tensor
        Row-normalized counts. Zero-mass rows remain all zero.
    """
    totals = counts.sum(dim=-1, keepdim=True)
    safe_totals = torch.where(totals > 0, totals, torch.ones_like(totals))
    return counts / safe_totals


def _as_histogram_parameter(
    values: torch.Tensor | np.ndarray | Sequence[float] | float,
    *,
    n_histograms: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """
    Coerce a scalar or per-histogram parameter to vector form.

    Parameters
    ----------
    values : torch.Tensor, numpy.ndarray, sequence of float, or float
        Scalar value or one value per histogram.
    n_histograms : int
        Expected number of histogram channels.
    dtype : torch.dtype
        Output tensor dtype.
    device : torch.device
        Output tensor device.
    name : str
        Parameter name used in validation error messages.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor shaped ``(n_histograms,)``.

    Raises
    ------
    ValueError
        If ``values`` is neither scalar nor length ``n_histograms``.
    """
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(n_histograms)
    if tensor.numel() != n_histograms:
        raise ValueError(f"`{name}` must be scalar or length `n_histograms`.")
    return tensor.reshape(n_histograms)


def _normalize_emd_by_bin_width(
    distances: torch.Tensor,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None,
) -> torch.Tensor:
    """
    Optionally divide per-histogram EMD distances by bin widths.

    Parameters
    ----------
    distances : torch.Tensor
        Distance tensor with histogram channels on the first dimension.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None
        Scalar or per-histogram positive bin widths. If None, distances are
        returned unchanged.

    Returns
    -------
    torch.Tensor
        Distances normalized by bin widths when provided.

    Raises
    ------
    ValueError
        If any provided bin width is non-positive.
    """
    if bin_widths is None:
        return distances

    widths = _as_histogram_parameter(
        bin_widths,
        n_histograms=distances.shape[0],
        dtype=distances.dtype,
        device=distances.device,
        name="bin_widths",
    )
    if torch.any(widths <= 0):
        raise ValueError("`bin_widths` must be positive.")
    return distances / widths[:, None, None]


def reduce_emd_vectors(
    distances: torch.Tensor | np.ndarray,
    *,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    reduction: str = "mean",
    histograms_last: bool = True,
) -> torch.Tensor:
    """
    Reduce per-histogram EMD vectors to scalar distances.

    Parameters
    ----------
    distances : torch.Tensor or numpy.ndarray
        Per-histogram distances.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram weights.
    reduction : {"mean", "sum", "l2"}, optional
        Reduction to apply across histogram channels. Default is ``"mean"``.
    histograms_last : bool, optional
        If True, histogram channels are on the last axis. If False, they are
        on the first axis.

    Returns
    -------
    torch.Tensor
        Reduced scalar distances with the histogram axis removed.

    Raises
    ------
    ValueError
        If ``reduction`` is not supported or weights have an incompatible
        length.
    """
    distances_t = torch.as_tensor(distances)
    hist_dim = -1 if histograms_last else 0
    n_hist = distances_t.shape[hist_dim]

    if weights is not None:
        weights_t = _as_histogram_parameter(
            weights,
            n_histograms=n_hist,
            dtype=distances_t.dtype,
            device=distances_t.device,
            name="weights",
        )
        shape = [1] * distances_t.ndim
        shape[hist_dim] = n_hist
        weighted = distances_t * weights_t.reshape(shape)
    else:
        weighted = distances_t

    if reduction == "sum":
        return weighted.sum(dim=hist_dim)
    if reduction == "mean":
        if weights is None:
            return weighted.mean(dim=hist_dim)
        return weighted.sum(dim=hist_dim) / weights_t.sum()
    if reduction == "l2":
        return torch.linalg.vector_norm(weighted, ord=2, dim=hist_dim)

    raise ValueError("`reduction` must be one of: 'mean', 'sum', or 'l2'.")


def pairwise_histogram_emd(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    histogram_index: int,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
) -> torch.Tensor:
    """
    Compute pairwise Wasserstein-1 distances for one histogram channel.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First histogram tensor collection.
    x2 : torch.Tensor or numpy.ndarray
        Second histogram tensor collection.
    histogram_index : int
        Histogram channel to compare.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix shaped ``(n_x1_samples, n_x2_samples)``.

    Raises
    ------
    ValueError
        If inputs have inconsistent valid bin counts.
    """
    h1 = _as_hist_tensor(x1, data_shape=data_shape)
    h2 = _as_hist_tensor(x2, data_shape=data_shape, dtype=h1.dtype, device=h1.device)

    lengths_1 = valid_bin_lengths(h1)
    lengths_2 = valid_bin_lengths(h2)
    if lengths_1 != lengths_2:
        raise ValueError("Inconsistent number of histogram bins between inputs.")

    bin_len = lengths_1[histogram_index]
    x1_bins = h1[:, histogram_index, :bin_len, 0]
    x1_counts = _normalize_counts(h1[:, histogram_index, :bin_len, 1])
    x2_bins = h2[:, histogram_index, :bin_len, 0]
    x2_counts = _normalize_counts(h2[:, histogram_index, :bin_len, 1])

    n1_samples = x1_bins.shape[0]
    n2_samples = x2_bins.shape[0]

    x1_bins_rep = torch.cat([x1_bins] * n2_samples, dim=0)
    x2_bins_rep = x2_bins.repeat_interleave(n1_samples, dim=0)
    x1_counts_rep = torch.cat([x1_counts] * n2_samples, dim=0)
    x2_counts_rep = x2_counts.repeat_interleave(n1_samples, dim=0)

    if n_batches is None:
        emd = ot.wasserstein_1d(
            x1_bins_rep.T,
            x2_bins_rep.T,
            u_weights=x1_counts_rep.T,
            v_weights=x2_counts_rep.T,
        )
    else:
        n_inputs = x1_bins_rep.shape[0]
        batch_size = max(math.ceil(n_inputs / n_batches), 1)
        chunks = []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            stop = min((batch_idx + 1) * batch_size, n_inputs)
            if start >= n_inputs:
                break
            chunks.append(
                ot.wasserstein_1d(
                    x1_bins_rep[start:stop].T,
                    x2_bins_rep[start:stop].T,
                    u_weights=x1_counts_rep[start:stop].T,
                    v_weights=x2_counts_rep[start:stop].T,
                )
            )
        emd = torch.cat(chunks)

    pairwise = emd.reshape(n2_samples, n1_samples).T
    if torch.equal(h1, h2):
        diag_len = min(pairwise.shape[0], pairwise.shape[1])
        pairwise[torch.arange(diag_len), torch.arange(diag_len)] = 0.0
    return pairwise


def pairwise_emd_distances(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Compute pairwise Wasserstein-1 distances for every histogram channel.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First histogram tensor collection.
    x2 : torch.Tensor or numpy.ndarray
        Second histogram tensor collection.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.

    Returns
    -------
    torch.Tensor
        Distance tensor shaped ``(n_histograms, n_x1_samples, n_x2_samples)``.

    Raises
    ------
    ValueError
        If the two inputs have different histogram channel counts.
    """
    h1 = _as_hist_tensor(x1, data_shape=data_shape)
    h2 = _as_hist_tensor(x2, data_shape=data_shape, dtype=h1.dtype, device=h1.device)
    if h1.shape[1] != h2.shape[1]:
        raise ValueError("Histogram channel counts must match.")

    per_hist = [
        pairwise_histogram_emd(h1, h2, histogram_index=hist_idx, n_batches=n_batches)
        for hist_idx in range(h1.shape[1])
    ]
    distances = torch.stack(per_hist, dim=0)
    return _normalize_emd_by_bin_width(distances, bin_widths)


def material_emd_vector(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Compute one EMD value per histogram channel for two single materials.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        Histogram tensor for one material.
    x2 : torch.Tensor or numpy.ndarray
        Histogram tensor for one material.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.

    Returns
    -------
    torch.Tensor
        Per-channel EMD vector shaped ``(n_histograms,)``.

    Raises
    ------
    ValueError
        If either input contains more than one material sample.
    """
    h1 = _as_hist_tensor(x1, data_shape=data_shape)
    h2 = _as_hist_tensor(x2, data_shape=data_shape, dtype=h1.dtype, device=h1.device)
    if h1.shape[0] != 1 or h2.shape[0] != 1:
        raise ValueError("`material_emd_vector` expects exactly one material in each input.")

    distances = pairwise_emd_distances(h1, h2, bin_widths=bin_widths)
    return distances[:, 0, 0]


def material_emd_distance(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute a scalar EMD distance between two materials.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        Histogram tensor for one material.
    x2 : torch.Tensor or numpy.ndarray
        Histogram tensor for one material.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram weights for the reduction.
    reduction : {"mean", "sum", "l2"}, optional
        Reduction applied to the per-histogram EMD vector. Default is
        ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar distance tensor.
    """
    distances = material_emd_vector(x1, x2, data_shape=data_shape, bin_widths=bin_widths)
    return reduce_emd_vectors(distances, weights=weights, reduction=reduction)


def aligned_material_emd_vectors(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Compute per-histogram EMD vectors for matched material pairs.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First collection of histogram tensors.
    x2 : torch.Tensor or numpy.ndarray
        Second collection of histogram tensors. Sample ``i`` is compared with
        sample ``i`` from ``x1``.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.

    Returns
    -------
    torch.Tensor
        Per-pair EMD vectors shaped ``(n_samples, n_histograms)``.

    Raises
    ------
    ValueError
        If the two inputs contain different numbers of samples.
    """
    h1 = _as_hist_tensor(x1, data_shape=data_shape)
    h2 = _as_hist_tensor(x2, data_shape=data_shape, dtype=h1.dtype, device=h1.device)
    if h1.shape[0] != h2.shape[0]:
        raise ValueError("Aligned EMD requires the same number of samples in both inputs.")

    distances = pairwise_emd_distances(h1, h2, n_batches=n_batches, bin_widths=bin_widths)
    return distances.diagonal(dim1=1, dim2=2).T


def aligned_material_emd_distances(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute scalar EMD distances for matched material pairs.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First collection of histogram tensors.
    x2 : torch.Tensor or numpy.ndarray
        Second collection of histogram tensors.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram weights for the reduction.
    reduction : {"mean", "sum", "l2"}, optional
        Reduction applied to per-histogram vectors. Default is ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar distances shaped ``(n_samples,)``.
    """
    distances = aligned_material_emd_vectors(
        x1, x2, data_shape=data_shape, n_batches=n_batches, bin_widths=bin_widths,
    )
    return reduce_emd_vectors(distances, weights=weights, reduction=reduction)


def pairwise_material_emd_vectors(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Compute per-histogram EMD vectors for all material pairs.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First collection of histogram tensors.
    x2 : torch.Tensor or numpy.ndarray
        Second collection of histogram tensors.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.

    Returns
    -------
    torch.Tensor
        Per-pair EMD vectors shaped
        ``(n_x1_samples, n_x2_samples, n_histograms)``.
    """
    distances = pairwise_emd_distances(
        x1, x2, data_shape=data_shape, n_batches=n_batches, bin_widths=bin_widths,
    )
    return distances.permute(1, 2, 0)


def pairwise_material_emd_distances(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute scalar EMD distances for all material pairs.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First collection of histogram tensors.
    x2 : torch.Tensor or numpy.ndarray
        Second collection of histogram tensors.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram weights for the reduction.
    reduction : {"mean", "sum", "l2"}, optional
        Reduction applied to per-histogram vectors. Default is ``"mean"``.

    Returns
    -------
    torch.Tensor
        Pairwise scalar distance matrix shaped ``(n_x1_samples,
        n_x2_samples)``.
    """
    distances = pairwise_material_emd_vectors(
        x1, x2, data_shape=data_shape, n_batches=n_batches, bin_widths=bin_widths,
    )
    return reduce_emd_vectors(distances, weights=weights, reduction=reduction)


def exponential_kernel_from_emd(
    distances: torch.Tensor | np.ndarray,
    *,
    lengthscales: torch.Tensor | np.ndarray | Sequence[float] | float,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    histograms_last: bool = False,
) -> torch.Tensor:
    """
    Build a summed exponential kernel from per-histogram EMD distances.

    Parameters
    ----------
    distances : torch.Tensor or numpy.ndarray
        Per-histogram distance tensor. By default the expected shape is
        ``(n_histograms, n_x1, n_x2)``.
    lengthscales : torch.Tensor, numpy.ndarray, sequence, or float
        Positive scalar or per-histogram lengthscales.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram kernel weights.
    histograms_last : bool, optional
        If True, interpret ``distances`` as shaped ``(n_x1, n_x2,
        n_histograms)``. Default is False.

    Returns
    -------
    torch.Tensor
        Kernel matrix shaped ``(n_x1, n_x2)``.

    Raises
    ------
    ValueError
        If ``distances`` is not three-dimensional or lengthscales are
        non-positive.
    """
    distances_t = torch.as_tensor(distances)
    if histograms_last:
        distances_t = distances_t.permute(2, 0, 1)
    if distances_t.ndim != 3:
        raise ValueError("`distances` must be a 3D tensor.")

    n_hist = distances_t.shape[0]
    lengthscales_t = _as_histogram_parameter(
        lengthscales,
        n_histograms=n_hist,
        dtype=distances_t.dtype,
        device=distances_t.device,
        name="lengthscales",
    )
    if torch.any(lengthscales_t <= 0):
        raise ValueError("`lengthscales` must be positive.")

    if weights is None:
        weights_t = torch.ones(n_hist, dtype=distances_t.dtype, device=distances_t.device)
    else:
        weights_t = _as_histogram_parameter(
            weights,
            n_histograms=n_hist,
            dtype=distances_t.dtype,
            device=distances_t.device,
            name="weights",
        )

    kernel_terms = torch.exp(-(distances_t / lengthscales_t[:, None, None])) * weights_t[:, None, None]
    return kernel_terms.sum(dim=0)


def pairwise_emd_kernel(
    x1: torch.Tensor | np.ndarray,
    x2: torch.Tensor | np.ndarray,
    *,
    lengthscales: torch.Tensor | np.ndarray | Sequence[float] | float,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    data_shape: tuple[int, int, int] | None = None,
    n_batches: int | None = None,
    bin_widths: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Compute a pairwise summed exponential kernel from histogram tensors.

    Parameters
    ----------
    x1 : torch.Tensor or numpy.ndarray
        First collection of histogram tensors.
    x2 : torch.Tensor or numpy.ndarray
        Second collection of histogram tensors.
    lengthscales : torch.Tensor, numpy.ndarray, sequence, or float
        Positive scalar or per-histogram lengthscales.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram kernel weights.
    data_shape : tuple of int or None, optional
        Shape used to reshape flattened inputs.
    n_batches : int or None, optional
        Optional number of batches for POT's ``wasserstein_1d`` calls.
    bin_widths : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-histogram widths used to normalize distances.

    Returns
    -------
    torch.Tensor
        Kernel matrix shaped ``(n_x1_samples, n_x2_samples)``.
    """
    distances = pairwise_emd_distances(
        x1, x2, data_shape=data_shape, n_batches=n_batches, bin_widths=bin_widths,
    )
    return exponential_kernel_from_emd(distances, lengthscales=lengthscales, weights=weights)


# ---------------------------------------------------------------------------
# CIF-to-histogram adapters
# ---------------------------------------------------------------------------

def load_histogram_payload(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Load one histogram payload from JSON or pickle.

    Parameters
    ----------
    path : str or os.PathLike
        Histogram payload path. Files ending in ``.json`` are parsed as JSON;
        all other paths are read as pickle files.

    Returns
    -------
    dict
        Loaded histogram payload.
    """
    path = resolve_path(path)
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def histogram_payload_to_tensor(payload: Dict[str, Any], *, key: str = "histogram_features") -> np.ndarray:
    """
    Extract a histogram tensor from a payload.

    Parameters
    ----------
    payload : dict
        Histogram payload containing an array under ``key``.
    key : str, optional
        Payload key containing histogram data. Default is
        ``"histogram_features"``.

    Returns
    -------
    numpy.ndarray
        Histogram array shaped ``(n_histograms, n_bins, 2)``.

    Raises
    ------
    ValueError
        If the selected payload value does not resolve to a three-dimensional
        histogram array with two values per bin.
    """
    hist = np.asarray(payload[key], dtype=float)
    if hist.ndim == 4:
        hist = hist[0]
    if hist.ndim != 3 or hist.shape[-1] != 2:
        raise ValueError("Histogram payload must resolve to shape `(n_histograms, n_bins, 2)`.")
    return hist


def load_histogram_collection(
    histogram_paths: Sequence[str] | Iterable[str],
    *,
    key: str = "histogram_features",
) -> tuple[list[str], np.ndarray, list[Dict[str, Any]]]:
    """
    Load a collection of histogram payloads into one tensor.

    Parameters
    ----------
    histogram_paths : sequence or iterable of str
        Histogram JSON or pickle payload paths.
    key : str, optional
        Payload key containing histogram data. Default is
        ``"histogram_features"``.

    Returns
    -------
    hist_names : list of str
        Histogram channel names shared by every payload.
    histograms : numpy.ndarray
        Stacked histogram tensor shaped ``(n_materials, n_histograms, n_bins,
        2)``.
    payloads : list of dict
        Loaded payloads in the same order as ``histogram_paths``.

    Raises
    ------
    ValueError
        If no paths are provided or histogram channel names differ between
        payloads.
    """
    payloads = [load_histogram_payload(path) for path in histogram_paths]
    if not payloads:
        raise ValueError("At least one histogram payload path is required.")

    hist_names = list(payloads[0].get("histogram_feat_names") or payloads[0].get("hist_names") or [])
    if not hist_names:
        raise ValueError("Histogram payload does not contain histogram feature names.")

    tensors = []
    for payload in payloads:
        names = list(payload.get("histogram_feat_names") or payload.get("hist_names") or [])
        if names != hist_names:
            raise ValueError("Histogram payloads must share the same histogram feature-name order.")
        tensors.append(histogram_payload_to_tensor(payload, key=key))

    return hist_names, np.stack(tensors), payloads


def histogram_channel_indices(
    hist_names: Sequence[str],
    selected_names: Sequence[str] | None = None,
) -> list[int]:
    """
    Resolve histogram channel names to integer indices.

    Parameters
    ----------
    hist_names : sequence of str
        Available histogram channel names.
    selected_names : sequence of str or None, optional
        Channel names to select. If None, all channels are selected.

    Returns
    -------
    list of int
        Channel indices in selection order.

    Raises
    ------
    KeyError
        If any requested channel name is not present.
    """
    if selected_names is None:
        return list(range(len(hist_names)))

    name_to_index = {name: idx for idx, name in enumerate(hist_names)}
    missing = [name for name in selected_names if name not in name_to_index]
    if missing:
        raise KeyError(f"Unknown histogram channel names: {missing}")
    return [name_to_index[name] for name in selected_names]


def reduce_selected_emd_vectors(
    emd_vectors: torch.Tensor | np.ndarray,
    hist_names: Sequence[str],
    *,
    selected_names: Sequence[str] | None = None,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Reduce selected histogram channels from EMD vectors.

    Parameters
    ----------
    emd_vectors : torch.Tensor or numpy.ndarray
        EMD vectors with histogram channels on the last axis, such as
        ``(n_histograms,)`` or ``(n_materials, n_materials, n_histograms)``.
    hist_names : sequence of str
        Histogram channel names corresponding to the last axis.
    selected_names : sequence of str or None, optional
        Channel names to include. If None, all channels are used.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-selected-channel weights.
    reduction : {"mean", "sum", "l2"}, optional
        Reduction applied across selected channels. Default is ``"mean"``.

    Returns
    -------
    torch.Tensor
        Reduced scalar distance or distance matrix.
    """
    idx = histogram_channel_indices(hist_names, selected_names)
    selected = torch.as_tensor(emd_vectors)[..., idx]
    return reduce_emd_vectors(selected, weights=weights, reduction=reduction, histograms_last=True)


def kernel_from_selected_emd_vectors(
    emd_vectors: torch.Tensor | np.ndarray,
    hist_names: Sequence[str],
    *,
    selected_names: Sequence[str] | None = None,
    lengthscales: torch.Tensor | np.ndarray | Sequence[float] | float = 1.0,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
) -> torch.Tensor:
    """
    Build a kernel from selected all-pairs EMD channels.

    Parameters
    ----------
    emd_vectors : torch.Tensor or numpy.ndarray
        All-pairs EMD vectors shaped ``(n_x1, n_x2, n_histograms)``.
    hist_names : sequence of str
        Histogram channel names corresponding to the last axis.
    selected_names : sequence of str or None, optional
        Channel names to include. If None, all channels are used.
    lengthscales : torch.Tensor, numpy.ndarray, sequence, or float, optional
        Positive scalar or per-selected-channel lengthscales.
    weights : torch.Tensor, numpy.ndarray, sequence, float, or None, optional
        Optional scalar or per-selected-channel kernel weights.

    Returns
    -------
    torch.Tensor
        Kernel matrix shaped ``(n_x1, n_x2)``.

    Raises
    ------
    ValueError
        If ``emd_vectors`` is not shaped as all-pairs vectors.
    """
    idx = histogram_channel_indices(hist_names, selected_names)
    selected = torch.as_tensor(emd_vectors)[..., idx]
    if selected.ndim != 3:
        raise ValueError("`emd_vectors` must have shape `(n_x1, n_x2, n_histograms)`.")
    return exponential_kernel_from_emd(
        selected,
        lengthscales=lengthscales,
        weights=weights,
        histograms_last=True,
    )


def build_emd_histogram_payload_from_cif(
    cif_path: str,
    *,
    bin_centers_path: str = DEFAULT_CLASSIFICATION_BIN_PICKLE,
    bin_centers_format: str = "auto",
    hist_density: bool = False,
    atomic_radii_path: str = DEFAULT_ATOMIC_RADII_JSON,
    atomic_features_path: str = DEFAULT_ATOMIC_FEATURES_JSON,
) -> Dict[str, Any]:
    """
    Build one EMD-ready histogram payload from a CIF file.

    Parameters
    ----------
    cif_path : str
        Input CIF file path.
    bin_centers_path : str, optional
        Path to predefined bin centers in JSON or legacy pickle format.
    bin_centers_format : {"auto", "json", "pickle"}, optional
        Format passed to ``load_predefined_bin_centers``. Default is
        ``"auto"``.
    hist_density : bool, optional
        If True, normalize histogram heights. Default is False.
    atomic_radii_path : str, optional
        JSON file containing element radii.
    atomic_features_path : str, optional
        JSON file containing element-level feature values.

    Returns
    -------
    dict
        Histogram payload containing EMD-compatible histogram arrays and
        provenance paths.
    """
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
    """
    Build EMD-ready histogram JSON payloads for many CIF files.

    Parameters
    ----------
    cif_paths : sequence or iterable of str
        Input CIF file paths.
    out_dir : str
        Directory where histogram JSON files are written.
    suffix : str, optional
        Filename suffix appended to each CIF stem. Default is
        ``"_emd_histogram.json"``.
    **kwargs
        Additional keyword arguments forwarded to
        ``build_emd_histogram_payload_from_cif``.

    Returns
    -------
    list of str
        Output histogram JSON paths.
    """
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


__all__ = [
    # tensor utilities
    "valid_bin_lengths",
    "reduce_emd_vectors",
    # per-material EMD
    "material_emd_vector",
    "material_emd_distance",
    "aligned_material_emd_vectors",
    "aligned_material_emd_distances",
    "pairwise_material_emd_vectors",
    "pairwise_material_emd_distances",
    # low-level pairwise
    "pairwise_histogram_emd",
    "pairwise_emd_distances",
    # kernel
    "exponential_kernel_from_emd",
    "pairwise_emd_kernel",
    # adapters
    "load_histogram_payload",
    "histogram_payload_to_tensor",
    "load_histogram_collection",
    "histogram_channel_indices",
    "reduce_selected_emd_vectors",
    "kernel_from_selected_emd_vectors",
    "build_emd_histogram_payload_from_cif",
    "build_emd_histogram_payloads_for_cifs",
]
