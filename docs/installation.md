# Installation

During development, use an isolated environment and an editable install:

```bash
conda create -n cranberry-dev python=3.11
conda activate cranberry-dev
cd cranberry
python -m pip install -e ".[dev]"
```

Verify the import path and CLI:

```bash
python -c "import cranberry; print(cranberry.__file__)"
cranberry inspect
```

If you plan to use GPU acceleration, run the OpenMM installation self-test first to confirm that your environment can see the available platform(s):

```bash
python -m openmm.testInstallation
```

After that, try a short Cranberry GPU smoke run with an explicit platform, for example:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CUDA
```
