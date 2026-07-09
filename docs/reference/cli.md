# CLI Reference

Implemented commands:

```bash
cranberry inspect
cranberry inspect data
cranberry inspect forcefield [MODEL]
cranberry inspect input PDB
cranberry cg PDB [--output OUT] [--add-terminal-phosphate]
cranberry prepare PDB [--output OUT] [--add-terminal-phosphate]
cranberry energy PDB
cranberry md PDB --steps STEPS
```

`cranberry cg` coarse-grains an atomistic RNA PDB into canonical CRANBERRY CG form. It emits a prepared `*_cg_vs_conect.pdb`-style output by default, with coarse-grained beads, BC/BN virtual sites, and `CONECT` records:

```bash
cranberry cg atomistic_input.pdb --output atomistic_input_cg_vs_conect.pdb
```

`cranberry prepare` is the canonicalization step for an already coarse-grained CRANBERRY PDB. By default it validates the canonical CG file, reports that nothing needs to be changed, and does not write a new file. Use `--add-terminal-phosphate` only when you want Cranberry to add missing 5'-terminal phosphate context near a chain end for sugar-puckering analysis in the coarse-grained model.

Common `cg` and `prepare` options:

```text
--output PATH              output PDB path; default depends on the command
--add-terminal-phosphate   insert a terminal phosphate at chain starts missing P
--no-overwrite             fail if the output file already exists
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
