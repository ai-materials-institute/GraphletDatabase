# graphlet-featurization

Graphlet featurization utilities for crystalline materials represented as CIF
files.

This repository accompanies work on graphlet-based featurization of crystalline
materials. It converts crystal structures into local 1-site, 2-site, and
3-site graphlet descriptors, aggregates those descriptors into shared-bin
histograms, and compares materials with Earth Mover's Distance (EMD) vectors,
scalar distances, and exponential kernels.

The code is designed around reproducible intermediate artifacts: graphlet JSON
files, shared bin-center JSON files, histogram JSON files, manifest files, and
live state files for long-running batch jobs.

## What This Repository Provides

- **Graphlet construction** from CIF files using `pymatgen` structures,
  Voronoi-neighbor geometry, atomic radii, and elemental feature tables.
- **Histogram descriptors** from graphlet feature distributions using one
  shared bin-center definition.
- **One-command pipeline** for the recommended CIF directory to graphlets to
  histograms workflow.
- **Collection-loading helpers** for stacking histogram JSON files into tensor
  form and selecting histogram channels by name.
- **EMD vectors** with one Wasserstein-1 distance per histogram channel.
- **Material distances** by reducing selected EMD channels with mean, sum, or
  L2 reductions and optional channel weights.
- **Kernel matrices** from selected per-channel EMD distances using a summed
  exponential kernel.
- **Batch execution utilities** with resumable folder processing, progress
  logs, state checkpoints, and manifest files.

## Repository Layout

```text
src/
    graphlets.py    Graphlet construction and graphlet-to-histogram analyzers
    core.py         Graphlet, histogram, bin-center, and batch workflow utilities
    emd.py          EMD vectors, scalar distances, kernels, and tensor adapters
    cli.py          Command-line entrypoints

config/
    atomic_radii.json
    Filtered_atomic_features.json
    Space_group.xls

pyproject.toml      Package metadata, dependencies, and console scripts
requirements.txt    Unpinned dependency list for pip users (uv sync is preferred)
```

## Input Data

The repository expects the user to provide a collection of CIF files. Atomic
radii and elemental feature tables used by the featurization workflow are
included under `config/`.

## Installation

The recommended setup uses `uv`.

```bash
uv sync
```

`uv sync` creates or updates `.venv`, resolves dependencies from
`pyproject.toml`, and installs this project in editable mode by default. The
editable install is useful because workflows resolve configuration files
relative to the repository root.

The package requires Python ≥ 3.10. To pin explicitly:

```bash
uv python install 3.10
uv sync --python 3.10
```

Check the environment:

```bash
uv run graphlet-pipeline --help
uv run graphlet-build-folder --help
uv run python -c "from core import run_graphlet_pipeline; from emd import load_histogram_collection; print('ok')"
```

## Quickstart

For a directory of CIF files, the most user-friendly path is the one-command
pipeline:

```bash
uv run graphlet-pipeline \
  --input-dir /path/to/cifs \
  --out-root /path/to/run_outputs \
  --recursive
```

This writes:

```text
/path/to/run_outputs/
    graphlets/                 Graphlet JSON files
    histograms/                Histogram JSON files
    bin_centers.json           Shared histogram bin definitions
    pipeline_manifest.json     Summary of generated artifacts
```

Then load the histograms and compute distances:

```python
import json
from pathlib import Path
from emd import (
    load_histogram_collection,
    pairwise_material_emd_vectors,
    reduce_selected_emd_vectors,
)

manifest = json.loads(Path("/path/to/run_outputs/pipeline_manifest.json").read_text())
hist_names, histograms, payloads = load_histogram_collection(
    manifest["histogram_paths"]
)

emd_vectors = pairwise_material_emd_vectors(histograms, histograms)
D = reduce_selected_emd_vectors(emd_vectors, hist_names, reduction="mean")
```

## Conceptual Workflow

The recommended workflow is:

1. **Create graphlets from a CIF collection.**
   Build one graphlet JSON per material. This is the expensive
   structure-parsing step and is resumable.

2. **Choose a reference set for histogram bins.**
   Use either the full graphlet collection or a representative subset, such as
   a training set, to derive one shared `bin_centers.json`.

3. **Build histograms with the shared bins.**
   Convert every material to be compared into a histogram JSON using the same
   bin-center file. This aligns histogram channel names and bin centers across
   materials.

4. **Compute EMD vectors.**
   For two materials, compute one EMD value per histogram channel. For a
   collection, compute all pairwise EMD vectors.

5. **Reduce or kernelize selected channels.**
   Select histogram channels by name or index. Reduce selected EMD channels to
   scalar distances, or use selected per-channel EMD matrices to construct
   kernels.

The most important invariant is: **all materials being compared must be
histogrammed with the same bin-center file**.

The reference set defines the available histogram channels. If a feature never
appears in the reference graphlets, it will not appear in later histogram
payloads produced from that bin-center file.

## Command-Line Usage

All commands are also available through the unified `graphlet-featurization`
dispatcher, which supports the same flags as the individual commands:

```bash
uv run graphlet-featurization pipeline --input-dir ... --out-root ...
uv run graphlet-featurization build-folder --input-dir ... --output-dir ...
uv run graphlet-featurization build-csv --csv-path ...
uv run graphlet-featurization compact-workflow /path/to/a.cif ...
```

### Recommended one-command pipeline

```bash
uv run graphlet-pipeline \
  --input-dir /path/to/cifs \
  --out-root /path/to/run_outputs \
  --recursive \
  --max-workers 20 \
  --cpu-cap 20
```

Useful options:

| Flag | Default | Meaning |
|---|---|---|
| `--pattern` | `*.cif` | Glob pattern for CIF discovery (e.g. `**/*.CIF`). |
| `--reference-list` | none | Newline-delimited graphlet JSON paths used to derive bin centers. Useful for train/test splits: derive bins once from the training set and pass that file here for subsequent runs. |
| `--reference-limit` | none | Use only the first N generated graphlets as the bin-center reference set. |
| `--num-bins` | `20` | Number of bins per histogram channel. |
| `--bin-width-factor` | `1.0` | Scaling factor for the dynamic bin-range estimation. |
| `--recompute-bins` | off | Recompute `bin_centers.json` if it already exists. |
| `--overwrite-graphlets` | off | Recompute graphlet JSON files that already exist. |
| `--hist-density` | off | Store normalized histogram heights. |

The pipeline command is a convenience wrapper around the staged workflow below.

### Build graphlet JSONs from a CIF directory

```bash
uv run graphlet-build-folder \
  --input-dir /path/to/cifs \
  --output-dir /path/to/graphlets \
  --recursive \
  --max-workers 20 \
  --cpu-cap 20
```

Useful options:

| Flag | Default | Meaning |
|---|---|---|
| `--pattern` | `*.cif` | Glob pattern for CIF discovery (e.g. `**/*.CIF`). |
| `--recursive` | off | Discover CIF files in subdirectories. |
| `--max-workers` | `20` | Requested process-pool worker count. |
| `--cpu-cap` | `20` | Hard upper bound on effective worker count. |
| `--max-in-flight` | `3×workers` | Max submitted-but-unfinished tasks. Reduce to limit memory use. |
| `--overwrite` | off | Recompute graphlet JSON files that already exist. |
| `--fail-fast` | off | Stop after the first failed CIF. |
| `--progress-every` | `100` | Print progress after every N completed CIFs. |
| `--progress-log` | none | Optional path for a timestamped progress log file (required by `tail -f` monitoring below). |
| `--checkpoint-every` | `25` | Write live state after every N finished worker tasks. |
| `--monitor-interval` | `30` | Heartbeat interval in seconds. |

The folder build writes:

- `graphlet_build_state.json` for live progress.
- `graphlet_build_manifest.json` for the final build summary.

Existing graphlet JSON files are skipped by default, so the command can be
re-run after interruption.

### Monitor a running build

The progress log is only written when `--progress-log` is passed to
`graphlet-build-folder`. If you started the build with that flag:

```bash
tail -f /path/to/graphlets/graphlet_build_progress.log
```

One-shot state-file summary (always available; the state file is written
regardless of `--progress-log`):

```bash
uv run python - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("/path/to/graphlets/graphlet_build_state.json").read_text())
counts = state["counts"]
timing = state["timing"]

print(
    f"built={counts['num_graphlets_built']} "
    f"skipped={counts['num_graphlets_skipped']} "
    f"failed={counts['num_graphlets_failed']}"
)
print(
    f"elapsed={timing['elapsed_human']} "
    f"eta={timing['eta_human']} "
    f"rate={timing['rate_items_per_min']:.1f}/min"
)
PY
```

### Small end-to-end CIF workflow

For small experiments, graphlets, bins, and histograms can be produced in one
command. Inputs can be individual CIF files or directories (searched
non-recursively for `*.cif`):

```bash
uv run graphlet-compact-workflow \
  /path/to/a.cif /path/to/b.cif \
  --graphlet-out-dir /tmp/graphlets \
  --bin-centers-path /tmp/bin_centers.json \
  --histogram-out-dir /tmp/histograms
```

For larger studies, prefer the staged workflow below so graphlets are built
once and reused.

### CSV-driven workflow

If CIF paths are stored in a CSV column:

```bash
uv run graphlet-build-csv \
  --csv-path /path/to/materials.csv \
  --cif-column cif \
  --out-root /path/to/output
```

## Python Workflow

The Python API exposes the same staged workflow used by the command-line tools.

### One-call pipeline helper

```python
from core import run_graphlet_pipeline

manifest = run_graphlet_pipeline(
    input_dir="/path/to/cifs",
    out_root="/path/to/run_outputs",
    recursive=True,
)
```

The returned manifest contains `graphlet_paths`, `histogram_paths`,
`bin_centers_path`, and all artifact counts.

### 1. Build graphlets for a CIF collection

```python
from core import run_folder_graphlet_build

manifest = run_folder_graphlet_build(
    input_dir="/path/to/cifs",
    output_dir="/path/to/graphlets",
    recursive=True,
    max_workers=20,
    cpu_cap=20,
)

graphlet_paths = manifest["graphlet_paths"]
```

### 2. Derive shared histogram bins from all graphlets or a subset

```python
from core import derive_dynamic_bin_centers

# Use all graphlets, or replace this with a training/reference subset.
reference_graphlets = graphlet_paths

bin_payload = derive_dynamic_bin_centers(
    reference_graphlets,
    out_path="/path/to/bin_centers.json",
    num_bins=20,
)
```

For train/test workflows, derive bins from the training set and reuse the saved
`bin_centers.json` for validation, test, and future materials.

### 3. Build histogram JSONs with the shared bins

```python
from core import batch_histogram_compact_feature_jsons

histogram_paths = batch_histogram_compact_feature_jsons(
    graphlet_paths,
    bin_centers_path="/path/to/bin_centers.json",
    histogram_out_dir="/path/to/histograms",
    prefer_existing_bins=True,
)
```

Each histogram payload contains:

- `histogram_feat_names`: histogram channel names in order.
- `histogram_features`: array shaped `(n_histograms, n_bins, 2)`.

The last axis stores `[bin_center, height]`.

### 4. Load histogram tensors

```python
from emd import load_histogram_collection

hist_names, histograms, payloads = load_histogram_collection(histogram_paths)

# histograms.shape == (n_materials, n_histograms, n_bins, 2)
```

### 5. Compute an EMD vector between two materials

```python
from emd import material_emd_vector

i, j = 0, 1
emd_vec = material_emd_vector(histograms[i], histograms[j])

# emd_vec.shape == (n_histograms,)
```

The entry `emd_vec[k]` is the EMD for histogram channel `hist_names[k]`.

### 6. Compute all-pairs EMD vectors

```python
from emd import pairwise_material_emd_vectors

all_emd_vecs = pairwise_material_emd_vectors(histograms, histograms)

# all_emd_vecs.shape == (n_materials, n_materials, n_histograms)
```

This tensor is useful to cache when experimenting with channel subsets,
reductions, weights, or kernels.

### 7. Reduce selected channels into material distances

```python
from emd import reduce_selected_emd_vectors

selected_names = [
    "bond_len_2_ord",
    "angle_mean_3_ord",
    "angle_std_3_ord",
]

# Distance between two materials from selected channels.
dist_ij = reduce_selected_emd_vectors(
    emd_vec,
    hist_names,
    selected_names=selected_names,
    reduction="mean",
)

# All-pairs distance matrix from selected channels.
D = reduce_selected_emd_vectors(
    all_emd_vecs,
    hist_names,
    selected_names=selected_names,
    reduction="mean",
)

# D.shape == (n_materials, n_materials)
```

Supported reductions are `mean`, `sum`, and `l2`. Optional per-channel weights
can be passed with `weights=...`.

### 8. Build a kernel from selected EMD channels

```python
from emd import kernel_from_selected_emd_vectors, histogram_channel_indices, pairwise_emd_kernel

# Option A: kernel from cached all-pairs EMD vectors.
K = kernel_from_selected_emd_vectors(
    all_emd_vecs,
    hist_names,
    selected_names=selected_names,
    lengthscales=1.0,
)

# K.shape == (n_materials, n_materials)

# Option B: compute the kernel directly from selected histogram channels.
selected_idx = histogram_channel_indices(hist_names, selected_names)
K_direct = pairwise_emd_kernel(
    histograms[:, selected_idx],
    histograms[:, selected_idx],
    lengthscales=1.0,
)
```

If one lengthscale is used per selected channel, pass a vector of length
`len(selected_idx)`.

## Artifact Summary

| Artifact | Producer | Purpose |
|---|---|---|
| Graphlet JSON | `graphlet-build-folder`, `build_graphlet_payload_from_cif` | Stores per-material graphlets and compact value-count feature distributions. |
| Bin-center JSON | `derive_dynamic_bin_centers` | Stores shared histogram channel names, bin centers, and bin edges. |
| Histogram JSON | `batch_histogram_compact_feature_jsons` | Stores fixed-bin histogram descriptors for each material. |
| Manifest JSON | batch workflows | Records paths, counts, timing, worker settings, and failures. |
| State JSON | `run_folder_graphlet_build` | Tracks live progress for long-running graphlet builds. |

Generated output directories such as `Graphlets/` and `ICSD_Features/` are
excluded from git.

## Public Modules

```python
from graphlets import Create_Graphlets, Graphlet_Analyzer, Graphlet_AnalyzerFixedBins2D
from core import (
    build_graphlet_payload_from_cif,
    batch_build_graphlet_jsons,
    derive_dynamic_bin_centers,
    batch_histogram_compact_feature_jsons,
    run_graphlet_pipeline,
    run_folder_graphlet_build,
)
from emd import (
    load_histogram_collection,
    histogram_payload_to_tensor,
    histogram_channel_indices,
    material_emd_vector,
    pairwise_material_emd_vectors,
    reduce_selected_emd_vectors,
    kernel_from_selected_emd_vectors,
    pairwise_emd_kernel,
)
```

## Reproducibility Notes

- Keep the `bin_centers.json` used for a study with the generated histogram
  JSON files.
- Do not compare histograms generated from different bin-center files.
- For machine-learning splits, derive bins on the training/reference set only,
  then reuse the same bins for validation and test materials. With the pipeline
  command, write the training graphlet paths to a text file (one path per line)
  and pass it via `--reference-list` for subsequent runs on new materials.
- Cache all-pairs EMD vectors when exploring multiple feature subsets or
  kernel hyperparameters.
- Inspect `graphlet_build_manifest.json` and `graphlet_build_state.json` for
  completed, skipped, and failed CIF counts.

## Citing

The accompanying paper is in preparation. Once published, a full citation will
be added here. In the meantime, if you use this repository in published work,
please include the repository URL and the commit hash or version tag used to
generate your results.

## License

This project is distributed under the [MIT License](LICENSE).
