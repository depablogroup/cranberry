# FAQ

## Installation

### Pip printed dependency-conflict warnings about `ndfes` or `edgembar`. Is Cranberry installed?

During `python -m pip install -e ".[dev]"`, pip may print a warning like:

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
ndfes 3.0 requires matplotlib, which is not installed.
ndfes 3.0 requires scipy, which is not installed.
edgembar 3.0 requires matplotlib, which is not installed.
```

This warning is about packages that are already present in the active Python environment.
It is not a Cranberry installation error: Cranberry's required runtime dependencies are
`openmm` and `h5py`, plus the development tools installed by the `[dev]` extra.

Verify Cranberry first:

```bash
python -c "import cranberry; print(cranberry.__file__)"
cranberry --help
```

If those commands work, Cranberry is installed and you can continue. If you also need
the packages named in the warning, install their missing dependencies in the same
activated environment:

```bash
conda activate cranberry-dev
python -m pip install matplotlib scipy
```

If `cranberry --help` still says `command not found`, or the import check prints
`None`, return to the repository checkout and rerun the editable install:

```bash
conda activate cranberry-dev
cd /absolute/path/to/your/cranberry
python -m pip install -e ".[dev]"
```

For the cleanest developer setup, create a fresh `cranberry-dev` environment and avoid
installing unrelated packages such as `ndfes` or `edgembar` there unless you need them.
