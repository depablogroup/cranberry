# Installation

During development, use an isolated environment and an editable install:

```bash
cd cranberry
python -m pip install -e ".[dev]"
```

Verify the import path:

```bash
python -c "import cranberry; print(cranberry.__file__)"
```
