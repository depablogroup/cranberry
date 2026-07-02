# Reference Output Generation

This developer-only note records how to generate temporary reference outputs while the new `cranberry-rna` package is being built.

## Policy

Use the GPU workshop copy as the temporary executable reference implementation:

```text
CRANBERRY_workshop/cranberry-workshop-919771f4-gpu/
```

Treat the full legacy repo as code reference only:

```text
OpenMM-CGRNA/
```

Do not use the full legacy repo as the normal source of generated regression data unless there is a specific reason. The workshop copy is smaller, carries the relevant CRANBERRY parameter and XML assets, and has runnable scripts with a known conda environment.

## Conda Environment

Use the workshop GPU environment:

```bash
conda activate cranberry-workshop-gpu
```

or, from automation:

```bash
conda run -n cranberry-workshop-gpu <command>
```

Check the environment exists:

```bash
conda env list
```

## GPU Check

From the GPU workshop root:

```bash
cd /home/yihengwu/code/projects/RNA/CRANBERRY/CRANBERRY_workshop/cranberry-workshop-919771f4-gpu
conda run -n cranberry-workshop-gpu python workshop/check_gpu.py
```

Expected successful output includes:

```text
OpenMM platforms: ['Reference', 'CPU', 'CUDA', 'OpenCL']
Requested GPU platform: CUDA
GPU platform check OK.
```

## Short 1zih Smoke Run

Use a separate `RUN_NAME` so reference generation does not overwrite existing workshop outputs:

```bash
cd /home/yihengwu/code/projects/RNA/CRANBERRY/CRANBERRY_workshop/cranberry-workshop-919771f4-gpu
conda run -n cranberry-workshop-gpu bash -lc 'NSTEPS=10 NRECORD=2 RUN_NAME=1zih_gpu_smoke bash workshop/run_1zih_gpu.sh'
```

This was verified on 2026-07-02. The run used the CUDA platform and wrote outputs to:

```text
CRANBERRY_workshop/cranberry-workshop-919771f4-gpu/runs/1zih_gpu_smoke/
```

Expected files include:

```text
args0.txt
checkpnt.chk
log
detailed.log
output.dcd
sugar.log
```

Note: the workshop script currently writes the legacy checkpoint typo `checkpnt.chk`. The new package should write `checkpoint.chk`.

## Full Workshop Run

The default workshop GPU run is:

```bash
cd /home/yihengwu/code/projects/RNA/CRANBERRY/CRANBERRY_workshop/cranberry-workshop-919771f4-gpu
conda run -n cranberry-workshop-gpu bash workshop/run_1zih_gpu.sh
```

Useful environment overrides:

```bash
NSTEPS=1000 NRECORD=50 RUN_NAME=1zih_gpu_ref bash workshop/run_1zih_gpu.sh
OPENMM_GPU_PLATFORM=OpenCL bash workshop/run_1zih_gpu.sh
```

The script sets:

```text
CRANBERRY_WORKSHOP_CGRNA=<workshop-root>/OpenMM-CGRNA
PYTHONPATH=$CRANBERRY_WORKSHOP_CGRNA
JAX_PLATFORM_NAME=cpu
```

and runs OpenMM on `OPENMM_GPU_PLATFORM`, default `CUDA`.

## Reference Data Use

Use workshop outputs for temporary regression targets while porting:

- initial potential energy and force-group components before minimization
- potential energy and force-group components after minimization
- `detailed.log` column behavior
- `log` reporter behavior
- `output.dcd` existence and basic trajectory shape

Do not treat short stochastic MD trajectories as strict numerical regression data. Prefer fixed-coordinate energy decomposition tests for stable package tests.
