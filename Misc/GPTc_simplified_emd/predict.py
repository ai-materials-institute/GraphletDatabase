"""
Unified GP Prediction Module.

Provides model loading (with auto-detected n_sg), single-sample inference,
and batch-over-folder prediction for both regression and classification.

Author: Aaditya Panigrahi, Natalie Maus, Yanjun Liu
"""

import sys, os, re, time, pickle, warnings, gc, math, json
import numpy as np
import torch
import gpytorch

warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from scalers import Histo_Array_Scaler
from kernels import SgEmdExactGPModelV2, EmdSgGpClassificationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Defaults ─────────────────────────────────────────────────────────────────
N_BATCHES_EMD_KERNEL = 10
N_INDUCING_PTS = 1024
REG_HIST_KEEP_IDXS = [13, 18, 26, 30]


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers  (formerly Predictor_Utils.py)
# ═════════════════════════════════════════════════════════════════════════════

def merge_dicts(dict1, dict2):
    """Merge two dicts whose values are lists (concatenate on collision)."""
    return {k: dict1.get(k, []) + dict2.get(k, []) for k in set(dict1) | set(dict2)}


def chunk_list(lst, chunk_size=2000):
    """Split a list into batches of length `chunk_size`."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def extract_valid_id(filename: str):
    """Extract an identifier from a filename, stripping extensions and known suffixes."""
    filename = os.path.basename(filename)
    filename = re.sub(r'_histogram\.json$', '', filename)
    filename = re.sub(r'_graphlet\.json$', '', filename)
    filename = re.sub(r'_histogram$', '', filename)
    filename = re.sub(r'_histogram\.pkl$', '', filename)
    filename = re.sub(r'\.json$', '', filename)
    filename = re.sub(r'\.pkl$', '', filename)
    match = re.search(r"icsd_(\d+)", filename)
    return match.group(1) if match else filename


def _load_feature_payload(path):
    """Load one JSON or pickle feature payload."""
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_histograms_from_payload(payload, hist_key, keep_idx):
    """
    Extract one histogram tensor from a single feature payload.
    """
    if hist_key in payload:
        hist = np.asarray(payload[hist_key], dtype=float)
    elif hist_key == "reg_histograms":
        for key in ("histogram_features", "hist_array"):
            if key in payload:
                hist = np.asarray(payload[key], dtype=float)
                break
        else:
            raise KeyError("No regression histogram tensor found in payload.")
    elif hist_key == "clas_histograms":
        for key in ("clas_histograms", "histogram_features", "hist_array"):
            if key in payload:
                hist = np.asarray(payload[key], dtype=float)
                break
        else:
            raise KeyError("No classification histogram tensor found in payload.")
    else:
        raise KeyError(f"Unsupported histogram key '{hist_key}'.")

    if hist.ndim == 4:
        hist = hist[0]
    if keep_idx is not None:
        hist = hist[keep_idx]
    return hist


def _extract_formula_from_payload(payload):
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return payload.get("reduced_formula") or payload.get("formula") or metadata.get("reduced_formula", "")


def _extract_symmetry_from_payload(payload):
    if "symmetry_feature" in payload:
        arr = np.asarray(payload["symmetry_feature"], dtype=float)
    elif "symmetry_features" in payload:
        arr = np.asarray(payload["symmetry_features"], dtype=float)
        if arr.ndim > 1:
            arr = arr[0]
    elif "symm_features" in payload:
        arr = np.asarray(payload["symm_features"], dtype=float)
        if arr.ndim > 1:
            arr = arr[0]
    else:
        arr = np.zeros(0, dtype=float)
    return arr


def clear_prediction_memory():
    """Force GC and clear CUDA cache to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# Unified model loader
# ═════════════════════════════════════════════════════════════════════════════

def _auto_detect_n_sg(state_dict):
    """Peek at the state dict to determine n_sg from saved weights."""
    sg_key = "covar_module.sg_kernel.base_kernel.raw_lengthscale"
    if sg_key in state_dict:
        n = state_dict[sg_key].shape[-1]
        print(f"Auto-detected n_sg={n} from saved model weights.")
        return n
    print("No symmetry kernel found in saved model. Using n_sg=0.")
    return 0


def load_model(model_path, model_type="regression",
               n_sg=None, n_histogram=None, data_shape=None):
    """
    Load a trained GP model, its likelihood, and the histogram scaler.

    Parameters
    ----------
    model_path : str
        Directory containing model_state.pt, likelihood_state.pt, scaler.pkl.
    model_type : str
        'regression' or 'classification'.
    n_sg : int or None
        Number of symmetry features.  None = auto-detect from saved weights.
    n_histogram : int or None
        Number of histogram features.  None = use default (4 for reg, 21 for cls).
    data_shape : tuple or None
        Shape of histogram data.  None = derive from n_histogram.

    Returns
    -------
    scaler, model, likelihood
    """
    # ---- defaults
    if n_histogram is None:
        n_histogram = 4 if model_type == "regression" else 21
    if data_shape is None:
        data_shape = (n_histogram, 20, 2)

    # ---- scaler
    scaler_path = os.path.join(model_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print(f"Warning: {scaler_path} not found. Instantiating unfitted scaler.")
        scaler = Histo_Array_Scaler()

    # ---- load state dict + auto-detect n_sg
    sd_path = os.path.join(model_path, "model_state.pt")
    lik_path = os.path.join(model_path, "likelihood_state.pt")
    state_dict = torch.load(sd_path, map_location=device)

    if n_sg is None:
        n_sg = _auto_detect_n_sg(state_dict)

    # ---- build model skeleton
    feat_width = n_sg + n_histogram * 20 * 2

    if model_type == "regression":
        # ExactGP requires the training data to compute posterior predictions
        train_x_path = os.path.join(model_path, "train_x.pt")
        train_y_path = os.path.join(model_path, "train_y.pt")
        if os.path.exists(train_x_path) and os.path.exists(train_y_path):
            train_x = torch.load(train_x_path, map_location=device)
            train_y = torch.load(train_y_path, map_location=device)
        else:
            # Fallback for old models without saved training data
            print("Warning: train_x.pt/train_y.pt not found. Predictions may be NaN.")
            train_x = torch.zeros(1, feat_width).to(device)
            train_y = torch.zeros(1).to(device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = SgEmdExactGPModelV2(
            train_x=train_x, train_y=train_y, likelihood=likelihood,
            n_sg=n_sg, map_saas_tau=None,
            n_histogram=n_histogram, data_shape=data_shape,
            n_batches_emd_kernel=N_BATCHES_EMD_KERNEL,
            add_saas_ls_prior=False,
        ).to(device)
        likelihood = model.likelihood
    else:
        dummy_x = torch.zeros(N_INDUCING_PTS, feat_width).to(device)
        model = EmdSgGpClassificationModel(
            train_x=dummy_x, n_histogram=n_histogram, n_sg=n_sg,
            data_shape=data_shape, n_batches_emd_kernel=N_BATCHES_EMD_KERNEL,
        ).to(device)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    model.load_state_dict(state_dict)
    likelihood.load_state_dict(torch.load(lik_path, map_location=device))
    model.eval()
    likelihood.eval()

    return scaler, model, likelihood


# ═════════════════════════════════════════════════════════════════════════════
# Unified single-sample prediction
# ═════════════════════════════════════════════════════════════════════════════

def predict(hist_feats, symm_feats, scaler, model, likelihood):
    """
    Run GP inference on prepared feature arrays.

    Parameters
    ----------
    hist_feats : np.ndarray  (B, H, 20, 2)
    symm_feats : np.ndarray  (B, n_sg)  — pass shape (B, 0) to omit.
    scaler : Histo_Array_Scaler
    model : GP model (regression or classification)
    likelihood : gpytorch likelihood

    Returns
    -------
    mean, std : np.ndarray, np.ndarray
    """
    model.eval()
    likelihood.eval()

    X = scaler.transform(hist_feats)
    X = torch.from_numpy(np.ascontiguousarray(X).copy()).float().to(device)
    S = torch.from_numpy(np.ascontiguousarray(symm_feats).copy()).float().to(device)

    X_flat = X.reshape(X.size(0), -1)
    X_cat = torch.cat([S, X_flat], dim=-1)

    with torch.no_grad():
        out = likelihood(model(X_cat))
        return out.mean.detach().cpu().numpy(), out.stddev.detach().cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# Unified batch-over-folder prediction
# ═════════════════════════════════════════════════════════════════════════════

def batch_predict(files_chunk, folder_path, scaler, model, likelihood,
                  model_type="regression", hist_key="reg_histograms",
                  keep_idx=None):
    """
    Predict on a batch of .pkl feature files.

    Parameters
    ----------
    files_chunk : list[str]
    folder_path : str
    scaler, model, likelihood : from load_model()
    model_type : 'regression' or 'classification'
    hist_key : key used to read histograms from each .pkl
    keep_idx : list[int] or None — histogram indices to keep

    Returns
    -------
    dict with keys: icsd_id, reduced_formula, pred, std
    """
    ids, formulas, hists, symms = [], [], [], []

    for fname in files_chunk:
        data = _load_feature_payload(os.path.join(folder_path, fname))
        hists.append(_extract_histograms_from_payload(data, hist_key, keep_idx))
        symms.append(_extract_symmetry_from_payload(data))
        ids.append(extract_valid_id(fname))
        formulas.append(_extract_formula_from_payload(data))

    hists = np.array(hists)
    symms = np.array(symms)

    # If model has no symmetry kernel, zero-out symm features
    if model.covar_module.n_sg == 0:
        symms = np.zeros((len(hists), 0), dtype=np.float32)

    mean, std = predict(hists, symms, scaler, model, likelihood)

    pred_key = "reg_pred" if model_type == "regression" else "clas_pred"
    std_key = "reg_std" if model_type == "regression" else "clas_std"

    return {
        'icsd_id': ids,
        'reduced_formula': formulas,
        pred_key: mean.tolist(),
        std_key: std.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI — run batch prediction on a folder of .pkl files
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch GP prediction over a folder of .json or .pkl feature files")
    parser.add_argument('model_type', choices=['regression', 'classification'])
    parser.add_argument('--folder', type=str,
                        default=os.path.join(BASE_DIR, "../ICSD_Features/Histogram_Features"),
                        help="Folder containing JSON or pickle feature files")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--hist-idx', nargs='+', type=int, default=None, help='Indices of histogram features to use (e.g., 12 18 26 30)')
    args = parser.parse_args()

    if args.model_path is None:
        if args.model_type == "regression":
            args.model_path = os.path.join(BASE_DIR, "Trained Models/Regressor_4-2odr_all-sym")
        else:
            args.model_path = os.path.join(BASE_DIR, "Trained Models/Classifier_2odr_all-sym")

    scaler, model, likelihood = load_model(args.model_path, args.model_type)

    if not os.path.exists(args.folder):
        print(f"Warning: Data folder not found at {args.folder}")
        sys.exit(1)

    files = [name for name in os.listdir(args.folder) if name.endswith(".json") or name.endswith(".pkl")]
    chunks = chunk_list(files)
    results = {}

    hist_key = "reg_histograms" if args.model_type == "regression" else "clas_histograms"
    
    if args.hist_idx is not None:
        keep_idx = args.hist_idx
    else:
        keep_idx = REG_HIST_KEEP_IDXS if args.model_type == "regression" else list(range(10, 31))

    start = time.time()
    for i, chunk in enumerate(chunks):
        print(f"\n Batch: {i+1}/{len(chunks)}")
        batch = batch_predict(chunk, args.folder, scaler, model, likelihood,
                              model_type=args.model_type, hist_key=hist_key, keep_idx=keep_idx)
        results = merge_dicts(results, batch)

    elapsed = time.time() - start
    print(f"\nFinished {len(files)} files in {elapsed:.1f}s")
""" Completed."""
