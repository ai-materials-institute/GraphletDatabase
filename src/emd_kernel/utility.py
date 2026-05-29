"""Standalone EMD distance and kernel utilities for padded histogram tensors."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import ot
import torch


def _as_hist_tensor(
    values: torch.Tensor | np.ndarray | Sequence[float],
    *,
    data_shape: tuple[int, int, int] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Coerce input into `(n_samples, n_histograms, n_bins, 2)` tensor form."""
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
    """Return the non-padded bin count for each histogram channel."""
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
    """Normalize counts row-wise while handling zero-mass histograms safely."""
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
    """Coerce a scalar or per-histogram parameter to shape `(n_histograms,)`."""
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
    """Optionally divide per-histogram EMD distances by histogram bin widths."""
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
    """Reduce per-histogram EMD vectors to scalar distances.

    By default, `distances` is expected to have histograms on the last axis,
    such as `(n_histograms,)`, `(n_samples, n_histograms)`, or
    `(n_x1_samples, n_x2_samples, n_histograms)`.

    Supported reductions:
    - `"mean"`: average feature distance, optionally weighted
    - `"sum"`: summed feature distance, optionally weighted
    - `"l2"`: Euclidean norm across histogram features
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
    """Compute pairwise Wasserstein-1 distances for one histogram channel."""
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
    """Compute pairwise Wasserstein-1 distances for every histogram channel.

    Returns a tensor with shape `(n_histograms, n_x1_samples, n_x2_samples)`.
    If `bin_widths` is provided, each histogram channel is divided by its
    corresponding bin width, giving a dimensionless distance in bin units.
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
    """Return one EMD value per histogram channel for two materials.

    Inputs may be one material in shape `(n_histograms, n_bins, 2)` or
    `(1, n_histograms, n_bins, 2)`. The output has shape `(n_histograms,)`.
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
    """Return one scalar EMD distance between two materials.

    This reduces the per-histogram vector from `material_emd_vector` using
    `"mean"`, `"sum"`, or `"l2"`.
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
    """Return per-histogram EMD vectors for matched material pairs.

    `x1[i]` is compared only with `x2[i]`. The output has shape
    `(n_samples, n_histograms)`.
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
    """Return scalar EMD distances for matched material pairs."""
    distances = aligned_material_emd_vectors(
        x1,
        x2,
        data_shape=data_shape,
        n_batches=n_batches,
        bin_widths=bin_widths,
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
    """Return per-histogram EMD vectors for all material pairs.

    The output has shape `(n_x1_samples, n_x2_samples, n_histograms)`, so
    `out[i, j, k]` is the EMD between material `i` and material `j` for
    histogram channel `k`.
    """
    distances = pairwise_emd_distances(
        x1,
        x2,
        data_shape=data_shape,
        n_batches=n_batches,
        bin_widths=bin_widths,
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
    """Return scalar EMD distances for all material pairs.

    The output has shape `(n_x1_samples, n_x2_samples)`.
    """
    distances = pairwise_material_emd_vectors(
        x1,
        x2,
        data_shape=data_shape,
        n_batches=n_batches,
        bin_widths=bin_widths,
    )
    return reduce_emd_vectors(distances, weights=weights, reduction=reduction)


def exponential_kernel_from_emd(
    distances: torch.Tensor | np.ndarray,
    *,
    lengthscales: torch.Tensor | np.ndarray | Sequence[float] | float,
    weights: torch.Tensor | np.ndarray | Sequence[float] | float | None = None,
    histograms_last: bool = False,
) -> torch.Tensor:
    """Convert per-histogram EMD distances into a summed exponential kernel.

    By default, `distances` should have shape
    `(n_histograms, n_x1_samples, n_x2_samples)`, as returned by
    `pairwise_emd_distances`. Set `histograms_last=True` for distances shaped
    `(n_x1_samples, n_x2_samples, n_histograms)`.
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
    """Convert per-histogram EMD distances into a summed exponential kernel."""
    distances = pairwise_emd_distances(
        x1,
        x2,
        data_shape=data_shape,
        n_batches=n_batches,
        bin_widths=bin_widths,
    )
    return exponential_kernel_from_emd(distances, lengthscales=lengthscales, weights=weights)
