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

`log` is the standard OpenMM state log. `detailed.log` adds force-group decomposition columns, including total potential energy and named CRANBERRY components such as `bond`, `pucker`, `stacking35`, `pairing`, `spline`, and `electrostatic`.

The Phase 3 CLI accepts `--steps` directly. A friendlier `--time` plus `--timestep` interface is deferred until the basic MD path is stable.
