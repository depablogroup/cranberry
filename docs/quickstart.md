# Quickstart

Inspect the installed package and bundled force-field assets:

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
