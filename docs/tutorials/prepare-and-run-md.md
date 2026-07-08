# Prepare And Run MD

`cranberry prepare` is the Phase 4 entry point for canonicalizing coarse-grained inputs, and `cranberry cg` is the short alias. At the moment they operate on prepared CRANBERRY CG PDBs, not full atomistic structures. In the default path they validate the canonical CG file, report that nothing needs to be changed, and do not write a new file. When requested, they can add a missing 5'-terminal phosphate bead and the matching `P-S3` bond before writing the output PDB. This phosphate option is mainly for restoring end-local phosphate context when you want Cranberry's coarse-grained sugar-puckering estimate near a chain end. The example filename pattern `*_cg_vs_conect.pdb` means coarse-grained + virtual-site + CONECT records:

```bash
cranberry prepare cranberry/data/examples/2ntCG_cg_vs_conect.pdb --add-terminal-phosphate
```

`cranberry md` still expects a canonical CRANBERRY coarse-grained PDB with virtual-site atoms and `CONECT` records. Use packaged examples or inspect your own input first:

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
