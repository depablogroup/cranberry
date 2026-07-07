# Quickstart

After installation, first confirm whether your OpenMM environment is visible and, if you plan to use GPU acceleration, verify the backend before running Cranberry:

```bash
python -m openmm.testInstallation
```

If that passes, inspect the installed package and bundled force-field assets:

```bash
cranberry inspect
cranberry inspect forcefield
cranberry inspect data
```

Validate a canonical coarse-grained input PDB:

```bash
cranberry inspect input cranberry/data/examples/1zih_cg_vs_conect.pdb
```

Compute an energy decomposition:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CPU
```

Run a tiny CPU MD smoke simulation:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb --steps 10 --output-dir md-out --platform CPU
```

For GPU validation, reuse the same command shape with an explicit platform such as `CUDA`:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CUDA
```

Restart from the generated checkpoint:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb --steps 10 --restart-from md-out/checkpoint.chk --output-dir md-out --platform CPU
```
