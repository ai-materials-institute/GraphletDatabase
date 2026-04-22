# Agent Run Instructions

This repository should be run with the conda environment `torchgpu` by default.

## One Command Pattern (Required)
- For Python commands in this repo, use:
  - `PYTHONPATH=src ./scripts/run_torchgpu.sh python3 <args>`
- This is the canonical command pattern and should match README examples.

## Why
- The base shell environment may not contain required packages (for example `numpy`).
- `torchgpu` is the environment with project dependencies.
- `scripts/run_torchgpu.sh` now sets writable cache defaults:
  - `MPLCONFIGDIR=/tmp/mpl`
  - `XDG_CACHE_HOME=/tmp/xdg-cache`
- Those defaults avoid repeated matplotlib/fontconfig cache permission warnings.

## Quick Checks
- Verify interpreter:
  - `PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "import sys; print(sys.executable)"`
- Verify core deps:
  - `PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "import numpy, pymatgen; print('ok')"`

## Example: Build Graphlet JSON For One CIF
- `PYTHONPATH=src ./scripts/run_torchgpu.sh python3 -c "from CompactFeatureWorkflow import build_and_save_graphlet_json; print(build_and_save_graphlet_json('data/MP_cifs/mp-2716482.cif', 'data/mp-2716482_graphlet.json'))"`

## Example: Long-Run Parallel Folder Build (Monitored + Resumable)
- `PYTHONPATH=src ./scripts/run_torchgpu.sh python3 main/Build_Folder_Graphlets.py --input-dir <cif-folder> --output-dir <graphlet-output-folder> --pattern '*.cif' --max-workers 20 --cpu-cap 20 --monitor-interval 30 --checkpoint-every 25`
- Resume behavior: rerun the same command without `--overwrite` to skip completed outputs.
- Monitoring files (default):
  - `<output-dir>/graphlet_build_state.json`
  - `<output-dir>/graphlet_build_manifest.json`

## Example: Unattended Full `MP_cifs` Run (`nohup`)
- Start supervisor:
  - `nohup ./scripts/run_full_mp_cifs_graphlets.sh > Graphlets/MP_cifs/nohup_supervisor.out 2>&1 &`
- Live monitor:
  - `./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs`
- One snapshot:
  - `ONCE=1 ./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs`
