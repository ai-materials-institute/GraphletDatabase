# data/

This directory is for local input datasets and is intentionally not versioned in Git.

Expected layout for the main workflow:

```text
data/
  MP_cifs/
    *.cif
```

Notes:
- CIF files are globally ignored by this repository (`*.cif`, `*.CIF`).
- Large archives (for example `MP_cifs.zip`) are also kept local only.
- To process a different dataset location, pass `--input-dir` to the CLI scripts.

Canonical run pattern:

```sh
PYTHONPATH=src ./scripts/run_torchgpu.sh python3 main/Build_Folder_Graphlets.py --input-dir <cif-dir> --output-dir <out-dir>
```
