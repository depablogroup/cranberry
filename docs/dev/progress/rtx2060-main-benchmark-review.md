# RTX 2060 main benchmark review

Date: 2026-09-06  
Target: `depablogroup/cranberry` `main`  
Commit tested: `2c879943` (`Timestep validation`)  
Tag present: `v1.0.0a1-pre-review`

## Scope

This is a refreshed speed snapshot for Cedric's benchmarking comparison. It
uses the existing `benchmarks/benchmark_md.py` seven-fixture CUDA suite and
does not change package or force-field behavior.

## Command and environment

```text
PYTHONPATH=/tmp/cranberry-main-benchmark \
conda run --no-capture-output -n cranberry-dev \
python benchmarks/benchmark_md.py \
  --platform CUDA \
  --series-name nvidia-geforce-rtx-2060 \
  --steps 1000 --warmup-steps 10 \
  --timestep-fs 5.0 --temperature-kelvin 298.0 \
  --salt-millimolar 150.0 --model default \
  --results-json benchmarks/results/nvidia-geforce-rtx-2060.json
```

- GPU: NVIDIA GeForce RTX 2060
- Driver: 595.84
- OpenMM: 8.5.2
- Cranberry: 1.0.0a1
- Platform: CUDA
- MPS: disabled

The benchmark runner completed all seven systems and generated the JSON
snapshot plus `docs/benchmarks/current.md`, `docs/benchmarks/current.svg`, and
`docs/benchmarks/index.md`.

## Results

| System | Nucleotides | Atoms | ns/day |
| --- | ---: | ---: | ---: |
| `2ntCG` | 2 | 15 | 7967.09 |
| `1zih` | 12 | 95 | 3370.32 |
| `157d` | 24 | 190 | 2875.54 |
| `1l2x` | 27 | 216 | 2882.87 |
| `rU40` | 40 | 319 | 3435.94 |
| `2mi0` | 43 | 342 | 2725.57 |
| `5ml7` | 95 | 760 | 2137.88 |

## Checks and risks

- `nvidia-smi` confirmed the RTX 2060 and driver before and after the run.
- The generated snapshot has schema version 1 and records the exact platform,
  driver, OpenMM version, model, timestep, and run settings.
- The existing publisher initially encountered an unrelated schema collision
  from `timestep-validation-rtx2060-summary.json`; that file was preserved and
  temporarily excluded only while regenerating the benchmark docs.
- The short 1,000-step protocol is suitable for comparison with the existing
  snapshot but is not a precision throughput estimate. Longer timed runs would
  be a separate benchmark series.
