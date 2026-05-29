#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GP Model Trainer.

Centralized training script for Exact GP Regression and Variational GP Classification models.
Replaces the old scattered scripts in sc_train_gp-main/scripts/ with a single CLI entrypoint.

It sets mathematically optimal feature indices by default, caches the fitted `Histo_Array_Scaler`,
and deposits all checkpointing states cleanly into a single output folder.

Usage:
    python train.py --model_type regression
    python train.py --model_type classification
"""

import sys, os, pickle, json, copy, argparse, time
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# --- Suppress extremely verbose warnings from PyTorch/gpytorch
import warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# --- Imports from flat modules
from data_loader import load_train_and_test_data
from kernels import SgEmdExactGPModelV2, EmdSgGpClassificationModel
from evaluation import (
    get_performance_stats,
    get_model_predictions,
    get_class_model_preds,
    get_performance_stats_classification,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _uses_json_feature_payloads(path_to_data_file):
    """
    Detect whether the training source is a folder/JSON of per-material features
    rather than a legacy dataset pickle.
    """
    if not path_to_data_file:
        return False
    resolved = os.path.abspath(os.path.expanduser(path_to_data_file))
    return os.path.isdir(resolved) or resolved.endswith(".json")


def _infer_hist_feature_count(path_to_data_file, preferred_keys):
    """
    Infer the number of histogram feature rows from the first JSON feature file.
    """
    resolved = os.path.abspath(os.path.expanduser(path_to_data_file))
    sample_path = None
    if os.path.isdir(resolved):
        for name in sorted(os.listdir(resolved)):
            if name.endswith(".json"):
                sample_path = os.path.join(resolved, name)
                break
    elif resolved.endswith(".json") and os.path.exists(resolved):
        sample_path = resolved

    if sample_path is None:
        return None

    with open(sample_path, "r") as f:
        payload = json.load(f)

    for key in preferred_keys:
        if key in payload:
            arr = np.asarray(payload[key], dtype=float)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim >= 3:
                return int(arr.shape[0])
    return None

def train_regression(args):
    """Executes the exact GP regression model training loop."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting GP Regression Training...")
    
    # 1. Load & split data. Scaler is fit and returned here.
    X_train, X_test, y_train, y_test, hist_data_shape, n_hist, n_symm, scaler = load_train_and_test_data(
        path_to_data_file=args.path_to_data_file,
        hist_features_key='histogram_features',
        labels_key='Tc',
        symm_features_key='symmetry_features',
        list_of_symm_features_to_use=args.symm_idx, # empty list defaults to all
        list_of_hist_features_to_use=args.hist_idx,
        random_split_seed=args.seed,
        test_size=args.test_size,
        use_oversampling_for_class_data=False,
        csv_split_file=args.csv_split_file,
        labels_csv_file=args.labels_csv_file,
    )
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    print(f"X_train shape: {X_train.shape}, n_hist: {n_hist}, n_symm: {n_symm}")
    model = SgEmdExactGPModelV2(
        train_x=X_train, 
        train_y=y_train, 
        likelihood=likelihood, 
        n_sg=n_symm,
        map_saas_tau=None,
        n_histogram=n_hist, 
        data_shape=hist_data_shape, 
        n_batches_emd_kernel=args.emd_batches,
        add_saas_ls_prior=False,
    ).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"[{time.strftime('%H:%M:%S')}] Commencing {args.epochs} optimization epochs...")
    for i in range(args.epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0 or i == 0:
            print(f"   Epoch {i+1}/{args.epochs} - Loss: {loss.item():.4f}")

    print(f"[{time.strftime('%H:%M:%S')}] Evaluating model performance...")
    train_preds, test_preds = get_model_predictions(model, likelihood, X_test, X_train)
    y_train_np = y_train.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    stats = get_performance_stats(
        train_preds=train_preds, y_train=y_train_np,
        test_preds=test_preds, y_test=y_test_np
    )
    stats = {k: float(v) for k, v in stats.items()}
    
    print(f"   Test R2: {stats['test_r2']:.3f} | Test MAE: {stats['test_mae']:.3f}")
    save_results(args.save_dir, model, likelihood, scaler, stats, train_preds, test_preds, y_train_np, y_test_np, n_hist, n_symm, "regression", X_train=X_train, y_train=y_train)


def train_classification(args):
    """Executes the variational ELBO GP classification model training loop."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting GP Variational Classification Training...")

    if _uses_json_feature_payloads(args.path_to_data_file):
        hist_features_key = 'histogram_features'
        symm_features_key = 'symmetry_features'
    else:
        hist_features_key = 'X_all'
        symm_features_key = 'symm_features'

    X_train, X_test, y_train, y_test, hist_data_shape, n_hist, n_symm, scaler = load_train_and_test_data(
        path_to_data_file=args.path_to_data_file,
        hist_features_key=hist_features_key,
        labels_key='label',
        symm_features_key=symm_features_key,
        list_of_symm_features_to_use=args.symm_idx,
        list_of_hist_features_to_use=args.hist_idx,
        random_split_seed=args.seed,
        test_size=args.test_size,
        use_oversampling_for_class_data=True, # Critical for imbalanced superconductor data
        csv_split_file=args.csv_split_file,
        labels_csv_file=args.labels_csv_file,
    )

    model = EmdSgGpClassificationModel(
        train_x=X_train[0:args.n_inducing], 
        n_histogram=n_hist,
        n_sg=n_symm,
        data_shape=hist_data_shape, 
        n_batches_emd_kernel=args.emd_batches
    ).to(device)

    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"[{time.strftime('%H:%M:%S')}] Commencing {args.epochs} optimization epochs (Batch Size: {args.batch_size})...")
    for i in range(args.epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (i+1) % 5 == 0 or i == 0:
            print(f"   Epoch {i+1}/{args.epochs} - Variational ELBO Loss: {epoch_loss:.4f}")

    print(f"[{time.strftime('%H:%M:%S')}] Evaluating model performance...")
    train_preds, test_preds = get_class_model_preds(model, likelihood, X_test, X_train, bsz=args.batch_size)
    y_test_np = y_test.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    
    stats = get_performance_stats_classification(
        y_test=y_test_np, y_train=y_train_np,
        test_preds=test_preds, train_preds=train_preds
    )
    
    print(f"   Test Acc: {stats.get('test_accuracy', 'N/A')} | Test F1: {stats.get('test_f1', 'N/A')}")
    save_results(args.save_dir, model, likelihood, scaler, stats, train_preds, test_preds, None, None, n_hist, n_symm, "classification")


def save_results(save_dir, model, likelihood, scaler, stats, train_preds, test_preds, y_train_np, y_test_np, n_hist, n_symm, model_type, X_train=None, y_train=None):
    """Consolidated utility to export state blocks to disk reliably."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 0. Training data (required for ExactGP posterior at inference time)
    if X_train is not None:
        torch.save(X_train.cpu(), os.path.join(save_dir, "train_x.pt"))
    if y_train is not None:
        torch.save(y_train.cpu(), os.path.join(save_dir, "train_y.pt"))
    
    # 1. State Dictionaries
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
    torch.save(likelihood.state_dict(), os.path.join(save_dir, "likelihood_state.pt"))
    if scaler is not None:
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
    # 2. Kernel Meta
    if n_hist > 0:
        np.save(os.path.join(save_dir, "emd_kernel_weights.npy"), model.covar_module.weights.detach().cpu().numpy())
        np.save(os.path.join(save_dir, "emd_kernel_lengthscales.npy"), model.covar_module.lengthscales.detach().cpu().numpy())
    if n_symm > 0:
        sg_ls = model.covar_module.sg_kernel.base_kernel.lengthscale.squeeze().detach().cpu().numpy()
        np.save(os.path.join(save_dir, "sg_kernel_lengthscale.npy"), np.array(sg_ls))
        
    # 3. Analytics
    np.save(os.path.join(save_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(save_dir, "test_preds.npy"), test_preds)
    with open(os.path.join(save_dir, "performance_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)
        
    # 4. Plots (Regression Only)
    if model_type == "regression" and y_test_np is not None:
        plot_regression(save_dir, "test_result.png", y_test_np, test_preds, stats["test_r2"], stats["test_mae"], "Test")
        plot_regression(save_dir, "train_result.png", y_train_np, train_preds, stats["train_r2"], stats["train_mae"], "Train")

    print(f"[{time.strftime('%H:%M:%S')}] Saved all models, scalers, and artifacts to {save_dir}")

def plot_regression(save_dir, filename, y_true, y_pred, r2, mae, split_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel(f"True {split_name} Y Value")
    plt.ylabel(f"Predicted Y Value")
    plt.title(f"{split_name} R2: {r2:.3f}, MAE: {mae:.3f}")
    plt.savefig(os.path.join(save_dir, filename))
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Training Interface for GP-Tc Models")
    
    # Required architecture
    parser.add_argument('model_type', choices=['regression', 'classification'], help='Model type to train')
    
    # Path/Data overrides (Defaults fallback to project defaults if not provided)
    parser.add_argument('--path_to_data_file', type=str, default=None, help='Overrides default dataloader .pkl path')
    parser.add_argument('--save_dir', type=str, default=None, help='Overrides default save directory')
    
    # Shared Optimization Hyperparameters
    parser.add_argument('--epochs', type=int, default=32, help='Number of training epochs (Default: 32)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (Regression Default=0.1, Classification Default=0.05)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train/Test split fraction (Default: 0.2, ignored if --csv_split_file provided)')
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility (Default: 2, ignored if --csv_split_file provided)')
    parser.add_argument('--csv_split_file', type=str, default=None, help='Path to CSV file with pre-defined train/test splits. If provided, splits from CSV instead of random splitting.')
    parser.add_argument('--labels_csv_file', type=str, default=None, help='Optional labels CSV when training directly from a folder of per-material JSON/pickle feature files.')
    parser.add_argument('--emd_batches', type=int, default=10, help='Chunks for EMD Kernel to prevent OOM (Default: 10)')
    parser.add_argument('--use-symm', action='store_true', help='Use 11D symmetry features alongside histograms')
    parser.add_argument('--hist-idx', nargs='+', type=int, default=None, help='Indices of histogram features to use (e.g., 12 18 26 30)')
    
    # Classification-specific
    parser.add_argument('--n_inducing', type=int, default=1024, help='Variational Inducing points (Classification Only)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size (Classification Only)')
    
    args = parser.parse_args()
    
    # Map default optimal configurations depending on the requested route.
    if args.model_type == "regression":
        args.path_to_data_file = args.path_to_data_file or os.path.join(BASE_DIR, "../../data/regression_data_histogram&symmetry.pkl")
        args.save_dir = args.save_dir or os.path.join(BASE_DIR, "Trained Models/Regressor_Custom") if args.hist_idx else os.path.join(BASE_DIR, "Trained Models/Regressor_4-2odr_all-sym")
        args.lr = args.lr if args.lr is not None else 0.1
        args.hist_idx = args.hist_idx if args.hist_idx is not None else [13, 18, 26, 30] # Optimal features for exact Tc regression
        args.symm_idx = list(range(11)) if args.use_symm else []
        train_regression(args)
        
    elif args.model_type == "classification":
        args.path_to_data_file = args.path_to_data_file or os.path.join(BASE_DIR, "../../data/classification_data_3DSCnonsc_labeled.pkl")
        args.save_dir = args.save_dir or os.path.join(BASE_DIR, "Trained Models/Classifier_2odr_all-sym")
        args.lr = args.lr if args.lr is not None else 0.05
        if args.hist_idx is None:
            if _uses_json_feature_payloads(args.path_to_data_file):
                inferred_hist_count = _infer_hist_feature_count(
                    args.path_to_data_file,
                    preferred_keys=("histogram_features", "reg_histograms", "clas_histograms", "X_all"),
                )
                args.hist_idx = list(range(inferred_hist_count if inferred_hist_count is not None else 67))
            else:
                # The legacy pre-packaged classification dataset `X_all`
                # contains the historical 21-feature slice.
                args.hist_idx = list(range(21))
        args.symm_idx = list(range(11)) if args.use_symm else []
        train_classification(args)
