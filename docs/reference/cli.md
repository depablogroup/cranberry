# CLI Reference

Implemented commands:

```bash
cranberry inspect
cranberry inspect data
cranberry inspect forcefield [MODEL]
cranberry inspect input PDB
cranberry energy PDB
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
- `cranberry md`
- `cranberry remd`
