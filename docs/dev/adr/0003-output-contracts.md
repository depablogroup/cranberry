# ADR 0003: Output Contracts

## Status

Accepted.

## Context

The legacy project already has output conventions used by scripts and tests. The new package should preserve useful OpenMM/OpenMMTools-compatible names while cleaning up obvious mistakes.

## Decision

Default MD outputs:

- `output.dcd`
- `log`
- `detailed.log`
- `args.json`
- `checkpoint.chk`
- `final.pdb`

Use `checkpoint.chk`, not the legacy typo `checkpnt.chk`.

`detailed.log` is written by default at the same interval as `log` and `output.dcd`. It includes total potential energy and force-group component energies, with OpenMM-style quoted headers and units.

Default REMD outputs:

- `output.nc`
- `output_checkpoint.nc`

Overwrite policy:

- `cranberry md` follows OpenMM-style overwrite behavior.
- `cranberry remd` follows OpenMMTools-style no-overwrite behavior by default; `--overwrite` should be an explicit opt-in.

## Consequences

- Existing analysis habits around `output.dcd`, `log`, and `output.nc` remain familiar.
- `detailed.log` becomes self-contained for energy analysis.
- Restart and output handling need to account for different overwrite expectations between MD and REMD.
