# CLI Reference

The parser help below is generated from the `argparse` definitions in `cranberry.cli.main` and `cranberry.remd`. The generated sections are the authoritative source for option names and defaults.

## Generated Help

Top-level CLI help:

```{argparse-help}
:module: cranberry.cli.main
:function: build_parser
```

REMD command help:

```{argparse-help}
:module: cranberry.cli.main
:function: build_parser
:command: remd
```

Implemented commands:

- `cranberry inspect`
- `cranberry prepare`
- `cranberry cg`
- `cranberry energy`
- `cranberry md`
- `cranberry remd`

`cranberry cg` coarse-grains an atomistic RNA PDB into canonical CRANBERRY CG form. It emits a prepared `*_cg_vs_conect.pdb`-style output by default, with coarse-grained beads, BC/BN virtual sites, and `CONECT` records.

`cranberry prepare` is the canonicalization step for an already coarse-grained CRANBERRY PDB. By default it validates the canonical CG file, reports that nothing needs to be changed, and does not write a new file. Use `--add-terminal-phosphate` when you want Cranberry to add missing 5'-terminal phosphate context near a chain end for sugar-puckering analysis in the coarse-grained model. This terminal-phosphate placement is a fixed deterministic heuristic in the v1 contract.

`cranberry md` still expects a canonical CRANBERRY coarse-grained PDB with virtual-site atoms and `CONECT` records.

The REMD implementation uses NetCDF as the restart artifact, with optional DCD translation for inspection. It follows OpenMMTools-style no-overwrite behavior by default, and `--overwrite` is the explicit opt-in if you want to replace existing REMD outputs.
