# Benchmarks

Benchmarks are separate from tests. Continuous integration should only run tiny benchmark smoke checks; full CPU/GPU benchmarks should be manual or scheduled and published here as snapshots.

The current benchmark tab is the latest published MD snapshot for the bundled canonical systems. It includes a speed-vs-system-size plot and a table of per-system throughput.

```{toctree}
:maxdepth: 1

current
```

The first benchmark slice is MD only. The next planned snapshots should add CPU vs GPU comparison rows, MPS on/off runs, and later REMD once that workflow exists.
