# Benchmarks

Benchmarks are separate from tests. Continuous integration should only run tiny benchmark smoke checks; full CPU/GPU benchmarks should be manual or scheduled and published here as snapshots.

The benchmark runner writes JSON snapshots under `benchmarks/results/` and regenerates this docs section. That means a new benchmark on another machine can update the published plots by committing the new JSON plus the generated benchmark docs artifacts.

```{toctree}
:maxdepth: 1

current
```

## Available snapshots

| Series | Generated | Platform | MPS | GPU | Raw JSON |
| --- | --- | --- | ---: | --- | --- |
| `nvidia-geforce-rtx-2060` | `2026-07-08T21:17:19.546056+00:00` | `CUDA` | `False` | `NVIDIA GeForce RTX 2060` | `nvidia-geforce-rtx-2060.json` |

## Planned expansion

- Add explicit CPU, CUDA, and later MPS series side by side for the same bundled canonical systems.
- Add multi-process MPS demonstrations later as separate benchmark kinds rather than mixing them into the first MD baseline.
- Add REMD benchmark series after the REMD workflow exists.
