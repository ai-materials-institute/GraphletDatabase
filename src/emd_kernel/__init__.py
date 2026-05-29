"""Reusable EMD kernel utilities for FinalFeaturization histogram payloads."""

from .adapters import (
    build_emd_histogram_payload_from_cif,
    build_emd_histogram_payloads_for_cifs,
    histogram_payload_to_tensor,
    load_histogram_payload,
)
from .utility import (
    aligned_material_emd_distances,
    aligned_material_emd_vectors,
    exponential_kernel_from_emd,
    material_emd_distance,
    material_emd_vector,
    pairwise_emd_distances,
    pairwise_emd_kernel,
    pairwise_histogram_emd,
    pairwise_material_emd_distances,
    pairwise_material_emd_vectors,
    reduce_emd_vectors,
    valid_bin_lengths,
)

__all__ = [
    "load_histogram_payload",
    "histogram_payload_to_tensor",
    "build_emd_histogram_payload_from_cif",
    "build_emd_histogram_payloads_for_cifs",
    "valid_bin_lengths",
    "reduce_emd_vectors",
    "material_emd_vector",
    "material_emd_distance",
    "aligned_material_emd_vectors",
    "aligned_material_emd_distances",
    "pairwise_material_emd_vectors",
    "pairwise_material_emd_distances",
    "pairwise_histogram_emd",
    "pairwise_emd_distances",
    "exponential_kernel_from_emd",
    "pairwise_emd_kernel",
]
