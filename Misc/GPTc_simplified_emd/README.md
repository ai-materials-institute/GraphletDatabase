Copied EMD-related code from `GPTc_simplified`.

Included files:
- `kernels.py`
- `train.py`
- `predict.py`
- `predict_single_cif.py`
- `Run_All67_GP_Training.sh`

Selection rule:
- files containing the EMD/Wasserstein implementation itself,
- or EMD-specific model loading, prediction, and training wiring.

Primary implementation:
- `kernels.py`
  - `SgEMDKernelV2`
  - `compute_emd_more_memory_efficient(...)`
  - calls `ot.wasserstein_1d(...)`

Original source locations:
- `/data/ICSD_graphlet/GPTc_simplified/src/GP_Models/kernels.py`
- `/data/ICSD_graphlet/GPTc_simplified/src/GP_Models/train.py`
- `/data/ICSD_graphlet/GPTc_simplified/src/GP_Models/predict.py`
- `/data/ICSD_graphlet/GPTc_simplified/src/predict_single_cif.py`
- `/data/ICSD_graphlet/GPTc_simplified/main/Run_All67_GP_Training.sh`
