# Development Environment

Use the dedicated conda environment for developing the new `cranberry-rna` package:

```bash
conda activate cranberry-dev
cd <workspace>/cranberry
python -m pip install -e ".[dev]"
```

The environment was created with Python 3.11. This is the supported development and runtime Python for the current release.

Verify the editable install:

```bash
python -c "import cranberry; print(cranberry.__file__); print(cranberry.__version__)"
cranberry --help
python -m pytest -q
sphinx-build -b html docs docs/_build/html
```

Expected Phase 1 status:

- Import resolves to the source tree under `CRANBERRY/cranberry/cranberry`.
- `cranberry --help` shows planned v1 subcommands.
- Minimal pytest suite passes.
- Sphinx builds public docs while excluding `docs/dev/`.

Do not use `base` for package development. Do not install the new package into `cranberry-workshop-gpu`; that environment is reserved for the temporary workshop reference runner.
