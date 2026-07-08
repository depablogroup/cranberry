# Installation

During development, use an isolated environment and install Cranberry from the repository checkout. The placeholder `/path/to/cranberry` means "the folder you cloned from GitHub"; do not type that path literally.

## Fresh editable install from GitHub

For CPU-only development, create a clean environment and let pip install Cranberry's Python dependencies:

```bash
conda create -n cranberry-dev python=3.11
conda activate cranberry-dev
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install -e ".[dev]"
```

For GPU development, install OpenMM with conda first, then install Cranberry. This keeps OpenMM's platform plugins under conda's dependency resolver instead of letting pip choose an OpenMM build during the editable install:

```bash
conda create -n cranberry-dev -c conda-forge python=3.11 openmm
conda activate cranberry-dev
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install -e ".[dev]"
```

If you already cloned the repository, skip `git clone` and change into your existing checkout instead:

```bash
conda activate cranberry-dev
cd /absolute/path/to/your/cranberry
python -m pip install -e ".[dev]"
```

For example, if your clone is in `~/code/cranberry`, run `cd ~/code/cranberry` before the `pip install` command.

## Verify the install

Run the verification commands from inside the activated `cranberry-dev` environment:

```bash
python -c "import cranberry; print(cranberry.__file__)"
cranberry --help
cranberry inspect
```

A successful editable install should print a path inside your repository checkout, such as `/absolute/path/to/your/cranberry/cranberry/__init__.py`. If `python -c "import cranberry; print(cranberry.__file__)"` prints `None`, Python is not importing the installed package from this repository. The most common fix is to return to the cloned repository folder and rerun the editable install:

```bash
conda activate cranberry-dev
cd /absolute/path/to/your/cranberry
python -m pip install -e ".[dev]"
```

For common installation warnings and troubleshooting, see the [FAQ](faq.md).

## GPU smoke test

If you plan to use GPU acceleration, run the OpenMM installation self-test first to confirm that your environment can see the available platform(s):

```bash
python -m openmm.testInstallation
```

You can also print the platforms registered in the active Python environment:

```bash
python - <<'PY'
from openmm import Platform
print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
PY
```

A CUDA-capable environment must include `CUDA` in that list. `nvidia-smi` and `which nvcc` are useful system checks, but they are not enough by themselves: Cranberry can only use CUDA if OpenMM's Python package has registered the CUDA platform plugin in the same environment.

If the platform list only contains entries such as `Reference`, `CPU`, and `OpenCL`, recreate the environment with OpenMM installed by conda first:

```bash
conda create -n cranberry-dev -c conda-forge python=3.11 openmm
conda activate cranberry-dev
cd /absolute/path/to/your/cranberry
python -m pip install -e ".[dev]"
python -m openmm.testInstallation
```

After OpenMM lists `CUDA`, try a short Cranberry GPU smoke run with an explicit platform:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CUDA
```
