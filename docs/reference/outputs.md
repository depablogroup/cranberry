# Output Files

`cranberry md` writes these files by default into `--output-dir`, which defaults to the current directory:

- `output.dcd`: trajectory written by OpenMM `DCDReporter`
- `log`: OpenMM state log with step, time, potential energy, temperature, and speed
- `detailed.log`: total potential energy plus named force-group components
- `args.json`: machine-readable record of the MD command settings
- `checkpoint.chk`: OpenMM checkpoint written after the run
- `final.pdb`: final coordinates after virtual-site positions are recomputed, including `CONECT` records from the topology

`detailed.log` uses OpenMM-style quoted headers with units:

```text
#"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)",...
```

MD overwrites these default outputs unless `--no-overwrite` is passed. Restarting with `--restart-from checkpoint.chk` loads the checkpoint state, then appends to `output.dcd`, `log`, and `detailed.log` in `--output-dir`. If any of those files are missing, Cranberry warns and creates them starting from the checkpoint step. `checkpoint.chk`, `args.json`, and `final.pdb` are updated to the latest run state. REMD output conventions are planned separately because REMD will follow OpenMMTools-style no-overwrite behavior.

`args.json` stores the latest MD run metadata, including model, input PDB SHA256, temperature, salt, timestep, platform, package versions, restart path, and append flags. Before replacing `args.json`, Cranberry archives any distinct previous metadata file under `args_history/000001_args.json`, `args_history/000002_args.json`, and so on. If the new metadata is identical, no history copy is written.

On restart, Cranberry reads the existing `args.json` in `--output-dir` when present. Model, input PDB hash, temperature, salt, timestep, and run kind mismatches are errors. Platform, OpenMM version, and Cranberry version mismatches are warnings. If `args.json` is missing, Cranberry warns and relies on OpenMM's checkpoint compatibility check.
