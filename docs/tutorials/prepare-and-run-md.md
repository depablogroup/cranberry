# Prepare And Run MD

`cranberry cg` is the Phase 4 entry point for coarse-graining atomistic RNA inputs into canonical CRANBERRY CG PDBs, and `cranberry prepare` is the command for canonicalizing an already coarse-grained input. In the default path, `prepare` validates the canonical CG file, reports that nothing needs to be changed, and does not write a new file. When requested, it can add a missing 5'-terminal phosphate bead and the matching `P-S3` bond before writing the output PDB. That placement is a fixed deterministic heuristic in the current model contract. The example filename pattern `*_cg_vs_conect.pdb` means coarse-grained + virtual-site + CONECT records:

```bash
cranberry cg atomistic_input.pdb --output atomistic_input_cg_vs_conect.pdb
cranberry prepare cranberry/data/examples/2ntCG_cg_vs_conect.pdb --add-terminal-phosphate
```

`cranberry md` still expects a canonical CRANBERRY coarse-grained PDB with virtual-site atoms and `CONECT` records. Use packaged examples or inspect your own input first:

```bash
cranberry inspect input cranberry/data/examples/2ntCG_cg_vs_conect.pdb
```

Run a short CPU simulation by giving an explicit step count:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb   --steps 1000   --n-record 10   --output-dir md-out   --platform CPU
```

The output directory contains these default files. Add `--write-minimization-report` to also write `minimization_report.json` with pre/post minimization total and force-group energies:

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
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb   --steps 1000   --restart-from md-out/checkpoint.chk   --output-dir md-out   --platform CPU
```

`checkpoint.chk` is refreshed at each report interval during the run and again when the run completes. `log` is the standard OpenMM state log with the legacy Cranberry fields: step, time, potential energy, kinetic energy, total energy, temperature, elapsed time, speed, and estimated time remaining. Add `--log-progress` to include OpenMM `Progress (%)` in the log header and rows. `detailed.log` adds force-group decomposition columns, including total potential energy and named CRANBERRY components such as `bond`, `pucker`, `stacking35`, `pairing`, `spline`, and `electrostatic`. `final.pdb` includes `CONECT` records so it can be inspected as a canonical Cranberry PDB.

`args.json` is the latest run metadata. If a later run changes any metadata, Cranberry archives the previous file in `args_history/`. During restart, Cranberry errors on model, PDB hash, temperature, salt, timestep, or run-kind mismatches; platform and version differences are warnings.

The Phase 3 CLI accepts `--steps` directly. A friendlier `--time` plus `--timestep` interface is deferred until the basic MD path is stable.
