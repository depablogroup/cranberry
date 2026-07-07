# Prepare And Run MD

`cranberry md` currently expects a canonical CRANBERRY coarse-grained PDB with virtual-site atoms and `CONECT` records. Use packaged examples or inspect your own input first:

```bash
cranberry inspect input cranberry/data/examples/2ntCG_cg_vs_conect.pdb
```

Run a short CPU simulation by giving an explicit step count:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --report-interval 100 \
  --output-dir md-out \
  --platform CPU
```

The output directory contains:

```text
output.dcd
log
detailed.log
args.json
checkpoint.chk
final.pdb
```

Restart from the checkpoint. Restart appends to `output.dcd`, `log`, and `detailed.log` in the output directory; missing output files are created with a warning:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --restart-from md-out/checkpoint.chk \
  --output-dir md-out \
  --platform CPU
```

`log` is the standard OpenMM state log. `detailed.log` adds force-group decomposition columns, including total potential energy and named CRANBERRY components such as `bond`, `pucker`, `stacking35`, `pairing`, `spline`, and `electrostatic`. `final.pdb` includes `CONECT` records so it can be inspected as a canonical Cranberry PDB.

`args.json` is the latest run metadata. If a later run changes any metadata, Cranberry archives the previous file in `args_history/`. During restart, Cranberry errors on model, PDB hash, temperature, salt, timestep, or run-kind mismatches; platform and version differences are warnings.

The Phase 3 CLI accepts `--steps` directly. A friendlier `--time` plus `--timestep` interface is deferred until the basic MD path is stable.
