1zih atomistic fixture

Source:
- Copied from `../OpenMM-CGRNA/data/pdb/aa/1zih.pdb`

Command used to generate the packaged coarse-grained reference:

```bash
cranberry cg cranberry/data/examples/aa/1zih/1zih.pdb --output cranberry/data/examples/1zih_cg_vs_conect.pdb
```

Notes:
- This fixture is used for the `cg` regression test.
- The coarse-grained reference is intentionally checked in separately so the test can compare the generated output against a stable packaged baseline.
