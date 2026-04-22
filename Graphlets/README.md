# Graphlets/

This directory stores generated graphlet outputs and run metadata.

Typical contents during/after runs:
- `*_graphlets.json` output files
- `graphlet_build_state.json`
- `graphlet_build_manifest.json`
- `graphlet_build_progress.log`
- `full_run_supervisor.log`

This directory is intentionally not versioned in Git except for:
- `Graphlets/.gitkeep`
- `Graphlets/README.md`

Recommended monitor command:

```sh
./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs
```

One-shot status snapshot:

```sh
ONCE=1 ./scripts/monitor_graphlet_build.sh Graphlets/MP_cifs
```
