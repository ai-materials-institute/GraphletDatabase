#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single CIF GP Prediction Module.

Takes a single CIF file and returns 4 GP model predictions:
- Classification probability (is this a superconductor?)
- Classification uncertainty (std)
- Regression mean (predicted Tc)
- Regression uncertainty (std)

Usage:
    python predict_single_cif.py /path/to/structure.cif

Author: Aaditya Panigrahi
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
import argparse
import numpy as np
import torch

# Add current directory to path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "GP_Models"))

from SplitGraphletSymmetryProcessor import _Util, load_bins_from_config
from CompactFeatureWorkflow import build_feature_payload_from_cif
from GP_Models.predict import (
    load_model,
    predict as gp_predict,
    REG_HIST_KEEP_IDXS,
)

warnings.filterwarnings("ignore")

# Legacy classification models use histogram indices 10:31 (21 histograms).
LEGACY_CLAS_HIST_SLICE = slice(10, 31)


def extract_features_from_cif(cif_path: str) -> dict:
    """
    Extract all features needed for GP prediction from a single CIF file.
    
    Parameters
    ----------
    cif_path : str
        Path to the CIF file.
    
    Returns
    -------
    dict with keys:
        - 'histograms': np.ndarray (67, 20, 2) full histogram tensor
        - 'symmetry_feature': np.ndarray (11,)
        - 'reduced_formula': str
    """
    cif_path = _Util.resolve_path(cif_path)
    payload = build_feature_payload_from_cif(cif_path)
    hist = np.asarray(payload["reg_histograms"], dtype=float)
    symm = np.asarray(payload.get("symmetry_feature", []), dtype=float)

    return {
        'histograms': hist,                                   # (67, 20, 2)
        'symmetry_feature': symm,                             # (11,)
        'reduced_formula': payload['reduced_formula'],
    }


def _infer_model_histogram_count(model_dir: str | None, default: int) -> int:
    """Infer histogram count from saved model artifacts."""
    if not model_dir:
        return default

    weights_path = os.path.join(model_dir, "emd_kernel_weights.npy")
    if os.path.exists(weights_path):
        try:
            weights = np.load(weights_path, allow_pickle=False)
            if weights.ndim == 0:
                return default
            return int(weights.shape[0])
        except Exception:
            pass

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            n_hist = getattr(scaler, "n_histograms", None)
            if n_hist:
                return int(n_hist)
        except Exception:
            pass

    return default


def _select_classification_histograms(full_hist: np.ndarray, expected_count: int) -> np.ndarray:
    """Match classifier histogram count to the saved model."""
    if full_hist.shape[0] == expected_count:
        return full_hist
    if expected_count == 21:
        return full_hist[LEGACY_CLAS_HIST_SLICE, :, :]
    raise ValueError(
        f"Cannot match classification histogram count {expected_count} "
        f"to available tensor with {full_hist.shape[0]} histograms."
    )


def _select_regression_histograms(
    full_hist: np.ndarray,
    expected_count: int,
    reg_hist_idx: list[int] | None,
) -> np.ndarray:
    """Match regressor histogram count to the saved model."""
    if reg_hist_idx is not None:
        reg_hist = full_hist[reg_hist_idx, :, :]
    elif expected_count == full_hist.shape[0]:
        reg_hist = full_hist
    else:
        reg_hist = full_hist[REG_HIST_KEEP_IDXS, :, :]

    if reg_hist.shape[0] != expected_count:
        raise ValueError(
            f"Selected {reg_hist.shape[0]} regression histograms but model expects {expected_count}."
        )
    return reg_hist


def _predict_classification(hist_feats, symm_feats, scaler, model, likelihood):
    """
    Run classification inference and return probability plus a Bernoulli std proxy.

    The older generic helper requests `stddev` from the Bernoulli likelihood,
    which is much slower for the large all-67 classifier. Here we use the
    predictive mean from the trained model and convert it to the standard
    deviation of a Bernoulli variable: sqrt(p * (1 - p)).
    """
    device = next(model.parameters()).device

    X = scaler.transform(hist_feats)
    X = torch.from_numpy(np.ascontiguousarray(X).copy()).float().to(device)
    S = torch.from_numpy(np.ascontiguousarray(symm_feats).copy()).float().to(device)

    X_flat = X.reshape(X.size(0), -1)
    X_cat = torch.cat([S, X_flat], dim=-1)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        probs = likelihood(model(X_cat)).mean.detach().cpu().numpy()

    stds = np.sqrt(np.clip(probs * (1.0 - probs), a_min=0.0, a_max=None))
    return probs, stds


def build_graphlet_outputs_from_cif(
    cif_path: str,
    feature_mode: str = "counts",
    include_graphlets: bool = True,
    histogram_mode: str | None = None,
    fixed_bin_set: str = "classification",
    hist_density: bool = True,
    num_bins: int | None = None,
) -> dict:
    """
    Build JSON-ready graphlet payloads from a CIF file.

    Parameters
    ----------
    cif_path : str
        Path to the CIF file.
    feature_mode : {"counts", "raw"}, optional
        Feature JSON style for the graphlet payload. Default is "counts".
    include_graphlets : bool, optional
        If True, include explicit graphlet records in the payload.
    histogram_mode : {"dynamic", "fixed", "fixed_2d"} or None, optional
        If provided, also include a histogram payload built from the same
        graphlet object.
    fixed_bin_set : {"classification", "regression"}, optional
        Which predefined bin-center config to use for fixed 2D histograms.
    hist_density : bool, optional
        Whether histogram values are densities instead of counts.
    num_bins : int or None, optional
        Optional override for dynamic histogram bin count.

    Returns
    -------
    dict
        JSON-ready graphlet payload, optionally with histogram payload.
    """
    cif_path = _Util.resolve_path(cif_path)
    config_dir = os.path.join(BASE_DIR, "../config")

    with open(os.path.join(config_dir, "atomic_radii.json"), "r") as f:
        atomic_radii = json.load(f)
    with open(os.path.join(config_dir, "Filtered_atomic_features.json"), "r") as f:
        atomic_features_dict = json.load(f)

    structure = PMGStructure.from_file(cif_path).get_primitive_structure()
    graphlet = Create_Graphlets(structure, atomic_radii)
    graphlet.Get_1_site_graphlets()
    graphlet.Get_2_site_graphlets()
    graphlet.Get_3_site_graphlets()
    graphlet.get_features(atomic_features_dict)

    payload = graphlet.get_json_payload(
        cif_path=cif_path,
        feature_mode=feature_mode,
        include_graphlets=include_graphlets,
        max_order=3,
    )

    if histogram_mode is not None:
        histogram_mode = histogram_mode.lower()
        if histogram_mode == "dynamic":
            payload["histogram"] = graphlet.get_histogram_payload(
                mode="dynamic",
                max_order=3,
                hist_density=hist_density,
                num_bins=num_bins,
                cif_path=cif_path,
            )
        elif histogram_mode in ("fixed", "fixed_2d"):
            bin_clas, bin_reg, feature_names = load_bins_from_config(
                bin_clas_path=os.path.join(config_dir, "bin_centers_classification.pkl"),
                bin_reg_path=os.path.join(config_dir, "bin_centers_regression.pkl"),
            )
            if fixed_bin_set == "classification":
                bin_centers_2d = bin_clas
            elif fixed_bin_set == "regression":
                bin_centers_2d = bin_reg
            else:
                raise ValueError("fixed_bin_set must be 'classification' or 'regression'.")

            payload["histogram"] = graphlet.get_histogram_payload(
                mode="fixed_2d",
                max_order=3,
                hist_density=hist_density,
                bin_centers_2d=bin_centers_2d,
                feature_names=feature_names,
                cif_path=cif_path,
            )
        else:
            raise ValueError("histogram_mode must be one of: None, 'dynamic', 'fixed', 'fixed_2d'.")

    return payload


def main(cif_path, reg_model=None, clas_model=None, reg_hist_idx=None):
    """
    Main entry point for command-line execution or programmatic use.
    
    Parameters
    ----------
    cif_path : str
        Path to the CIF file
    reg_model : str, optional
        Path to custom trained regression model directory
    clas_model : str, optional
        Path to custom trained classification model directory
    reg_hist_idx : list[int], optional
        Histogram indices to use for regression
    
    Returns
    -------
    dict
        Prediction results and formula
    """
    import logging
    
    # Configure logging to suppress noisy output from dependencies
    logging.basicConfig(level=logging.ERROR)
    
    # Build models directory paths if not provided
    GP_MODELS_DIR = os.path.join(BASE_DIR, "GP_Models", "Trained Models")
    reg_model = reg_model or os.path.join(GP_MODELS_DIR, "Regressor_4-2odr_all-sym")
    clas_model = clas_model or os.path.join(GP_MODELS_DIR, "Classifier_2odr_all-sym")
    
    print(f"{'='*50}")
    print(f"GP-Tc Prediction Results")
    print(f"{'='*50}")
    
    try:
        # Extract features from CIF
        features = extract_features_from_cif(cif_path)
        full_hist = features['histograms']
        clas_n_hist = _infer_model_histogram_count(clas_model, default=21)
        reg_n_hist = _infer_model_histogram_count(reg_model, default=len(REG_HIST_KEEP_IDXS))
        clas_h_selected = _select_classification_histograms(full_hist, clas_n_hist)
        reg_h_selected = _select_regression_histograms(full_hist, reg_n_hist, reg_hist_idx)
        
        # Load models (n_sg is auto-detected from saved weights)
        clas_scaler, clas_model_obj, clas_likelihood = load_model(
            clas_model,
            model_type="classification",
            n_histogram=clas_h_selected.shape[0],
            data_shape=(clas_h_selected.shape[0], 20, 2),
        )
        reg_scaler, reg_model_obj, reg_likelihood = load_model(
            reg_model,
            model_type="regression",
            n_histogram=reg_h_selected.shape[0],
            data_shape=(reg_h_selected.shape[0], 20, 2),
        )
        
        # Determine batch size processing needs
        min_batch_size = 10
        
        # Prepare input arrays - duplicate to meet min_batch_size
        clas_hist = np.tile(clas_h_selected[np.newaxis, :], (min_batch_size, 1, 1, 1))
        reg_hist = np.tile(reg_h_selected[np.newaxis, :], (min_batch_size, 1, 1, 1))
        
        symm_feat = np.tile(features['symmetry_feature'][np.newaxis, :], (min_batch_size, 1))
        empty_symm = np.zeros((min_batch_size, 0), dtype=np.float32)
        
        # Run classification prediction (pass symm features only if model uses them)
        clas_symm = symm_feat if clas_model_obj.covar_module.n_sg > 0 else empty_symm
        clas_mean, clas_std = _predict_classification(
            clas_hist,
            clas_symm,
            clas_scaler,
            clas_model_obj,
            clas_likelihood,
        )
        
        # Run regression prediction (pass symm features only if model uses them)
        reg_symm = symm_feat if reg_model_obj.covar_module.n_sg > 0 else empty_symm
        reg_mean, reg_std = gp_predict(reg_hist, reg_symm, reg_scaler, reg_model_obj, reg_likelihood)
        
        results = {
            'classification_prob': float(clas_mean[0]),
            'classification_std': float(clas_std[0]),
            'regression_mean': float(reg_mean[0]),
            'regression_std': float(reg_std[0]),
            'reduced_formula': features['reduced_formula'],
        }
        
        print(f"Formula:               {results['reduced_formula']}")
        print(f"{'='*50}")
        print(f"Classification (SC?):")
        print(f"  Probability:         {results['classification_prob']:.4f}")
        print(f"  Uncertainty (std):   {results['classification_std']:.4f}")
        print(f"{'='*50}")
        print(f"Regression (Tc):")
        print(f"  Predicted Tc:        {results['regression_mean']:.4f} K")
        print(f"  Uncertainty (std):   {results['regression_std']:.4f} K")
        print(f"{'='*50}\n")
        
        return results
        
    except Exception as e:
        print(f"\nError processing {cif_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict superconductivity properties from a CIF file."
    )
    parser.add_argument("cif_path", type=str, help="Path to the CIF file")
    parser.add_argument('--reg-model', type=str, default=None, help="Path to regression model directory")
    parser.add_argument('--clas-model', type=str, default=None, help="Path to classification model directory")
    parser.add_argument('--reg-hist-idx', nargs='+', type=int, default=None, help="Custom histogram indices for regression")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cif_path):
        print(f"Error: CIF file not found: {args.cif_path}", file=sys.stderr)
        sys.exit(1)
        
    main(args.cif_path, args.reg_model, args.clas_model, args.reg_hist_idx)
