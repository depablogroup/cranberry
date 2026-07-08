# CLI Reference

Implemented commands:

```bash
cranberry inspect
cranberry inspect data
cranberry inspect forcefield [MODEL]
cranberry inspect input PDB
cranberry prepare PDB [--output OUT] [--add-terminal-phosphate]
cranberry cg PDB [--output OUT] [--add-terminal-phosphate]
cranberry energy PDB
cranberry md PDB --steps STEPS
```

Run a short CPU MD simulation from a canonical coarse-grained PDB:

The example filename pattern `*_cg_vs_conect.pdb` means coarse-grained + virtual-site + CONECT records.

```bash
cranberry md input_cg_vs_conect.pdb --steps 1000 --output-dir md-out --platform CPU
```

Restart from a previous MD checkpoint. Restart appends to `output.dcd`, `log`, and `detailed.log` in the output directory; missing output files are created with a warning. Cranberry also checks restart metadata from `args.json` when available and archives distinct previous metadata under `args_history/`:

```bash
cranberry md input_cg_vs_conect.pdb --steps 1000 --restart-from md-out/checkpoint.chk --output-dir md-out --platform CPU
```

`cranberry prepare` is the primary command name, and `cranberry cg` is its short alias. They currently operate on already coarse-grained CRANBERRY inputs rather than full atomistic structures. By default they validate the canonical CG PDB, print that nothing needs to be changed, and do not write a new file. When `--add-terminal-phosphate` is enabled, Cranberry rewrites the PDB to insert a `P` bead at each chain start missing one and adds the corresponding `P-S3` `CONECT` bond. This is mainly useful when you want phosphate context near a chain end for the coarse-grained sugar-puckering estimate.

Common `prepare` options:

```text
--output PATH              output PDB path, used only with --add-terminal-phosphate
--add-terminal-phosphate   insert a terminal phosphate at chain starts missing P
--no-overwrite             fail if the output file already exists
```

Common `md` options:

```text
--steps N              number of MD integration steps; required
--output-dir DIR       output directory, default current directory
--model MODEL          force-field model, default resolves to cranberry-v1-alpha.1
--temperature K        temperature in kelvin, default 298
--salt MM              salt concentration in millimolar, default 150
--timestep FS          integration timestep in femtoseconds, default 5
--report-interval N    steps between output reports, default min(steps, 1000)
--platform NAME        OpenMM platform name, default CPU; use default to let OpenMM choose
--restart-from CHK     OpenMM checkpoint to restart from
--no-overwrite         fail if default MD output files already exist
```

Compute total and decomposed energies on CPU:

```bash
cranberry energy input_cg_vs_conect.pdb --platform CPU
```

Write machine-readable JSON:

```bash
cranberry energy input_cg_vs_conect.pdb --json
```

Common `energy` options:

```text
--model MODEL          force-field model, default resolves to cranberry-v1-alpha.1
--temperature K        temperature in kelvin, default 298
--salt MM              salt concentration in millimolar, default 150
--platform NAME        OpenMM platform name, default CPU; use default to let OpenMM choose
--json                 print JSON instead of a text table
```

Planned commands:

- `cranberry remd`
