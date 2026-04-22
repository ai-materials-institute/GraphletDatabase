# GraphletDatabase

GraphletDatabase is a focused toolkit for generating structure-derived features from CIF files.

It supports three main workflows:
- CIF -> graphlet JSON
- CIF -> symmetry feature payload
- graphlet JSON -> histogram JSON

The implementation is centralized in `src/graphlet_core.py` with thin compatibility wrappers for older entrypoints.

## Requirements

This repository is designed to run with the conda environment `torchgpu`.

Use this canonical command pattern for Python commands:

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 <args>
```

Why this matters:
- The base shell may be missing required packages.
- `scripts/run_torchgpu.sh` runs commands inside `torchgpu`.

## Quick Start

### 1) Verify interpreter and core dependencies

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "import sys; print(sys.executable)"
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "import numpy, pymatgen; print('ok')"
```

### 2) Build graphlet JSON for one CIF

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "from CompactFeatureWorkflow import build_and_save_graphlet_json; print(build_and_save_graphlet_json('data/MP_cifs/mp-2716482.cif', 'data/mp-2716482_graphlet.json'))"
```

## Data and Output Policy

This project intentionally does not version large datasets or generated artifacts.

- Raw inputs live in `data/` (for example `data/MP_cifs/*.cif`).
- Generated graphlets and runtime logs live in `Graphlets/`.
- CSV workflow outputs default to `ICSD_Features/`.

Git rules in this repo:
- `data/*`, `Graphlets/*`, and `ICSD_Features/` are ignored.
- `*.cif` and `*.CIF` are globally ignored.
- Folder docs/placeholders are tracked (`.gitkeep` and README files).

## Main Workflows

### Build graphlet JSONs from a CIF folder

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 main/Build_Folder_Graphlets.py \
  --input-dir data/MP_cifs \
  --output-dir Graphlets/MP_cifs \
  --pattern '*.cif' \
  --suffix '_graphlets.json' \
  --max-workers 20 \
  --cpu-cap 20 \
  --monitor-interval 30 \
  --checkpoint-every 25
```

Resume behavior:
- Re-run the same command without `--overwrite`.
- Existing graphlet outputs are skipped.

Runtime monitoring files:
- `<output-dir>/graphlet_build_state.json`
- `<output-dir>/graphlet_build_manifest.json`
- `<output-dir>/graphlet_build_progress.log`

### Long-run full `MP_cifs` build (supervised + resumable)

Start supervisor:

```sh
nohup ./scripts/run_full_mp_cifs_graphlets.sh > Graphlets/MP_cifs/nohup_supervisor.out 2>&1 &
```

Monitor live:

```sh
./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs
```

One-shot status snapshot:

```sh
ONCE=1 ./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs
```

Supervisor retry controls (optional):
- `MAX_RESTARTS=10` by default (`-1` for unlimited)
- `STALL_LIMIT=3` by default (`-1` to disable)

### Build graphlet + histogram JSONs from a CSV

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 main/Build_CSV_Graphlet_Histograms.py \
  --csv-path <path-to-csv> \
  --cif-column cif \
  --out-root /tmp/graphlet_hist_out
```

### Run full workflow from files/directories directly

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 src/CompactFeatureWorkflow.py \
  data/MP_cifs \
  --graphlet-out-dir /tmp/graphlets_out \
  --bin-centers-path /tmp/graphlet_bins.json \
  --histogram-out-dir /tmp/histograms_out
```

## CLI Entrypoints

- `main/Build_Folder_Graphlets.py`
  - Parallel CIF folder -> graphlet JSON build.
- `main/Build_CSV_Graphlet_Histograms.py`
  - CSV CIF paths -> graphlets + dynamic bin centers + histograms.
- `src/CompactFeatureWorkflow.py`
  - Compatibility CLI for full CIF -> graphlet -> histogram workflow.

## Project Layout

- `src/graphlet_core.py`: central implementation.
- `src/PYGraphlets.py`: low-level graphlet construction.
- `src/CompactFeatureWorkflow.py`: compatibility wrapper.
- `src/SplitGraphletSymmetryProcessor.py`: symmetry wrapper.
- `src/cli_common.py`: CLI helper wrapper.
- `config/`: runtime configs and default bin-center pickles.
- `scripts/`: environment helper, long-run supervisor, live monitor.
- `main/`: top-level CLI entry scripts.
- `data/`: local input data (ignored in git).
- `Graphlets/`: local outputs/logs (ignored in git).

## Troubleshooting

### `Conda binary not found`

Set `CONDA_BIN` before running:

```sh
CONDA_BIN=/path/to/conda PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -V
```

### No CIF files discovered

Check:
- `--input-dir` points to the actual folder.
- `--pattern` matches your filenames.
- Use `--recursive` if CIF files are nested in subfolders.

## Notes on Scope

This repository is intentionally scoped to feature generation workflows.
Training-only and legacy entrypoints are excluded to keep maintenance and runtime behavior focused.
