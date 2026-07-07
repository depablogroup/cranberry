# CLI Reference

Implemented commands:

```bash
cranberry inspect
cranberry inspect data
cranberry inspect forcefield [MODEL]
cranberry inspect input PDB
cranberry energy PDB
cranberry md PDB --steps STEPS
```

Run a short CPU MD simulation from a canonical coarse-grained PDB:

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
--timestep FS          integration timestep in femtoseconds, default 10
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

- `cranberry prepare`
- `cranberry cg`
- `cranberry remd`
