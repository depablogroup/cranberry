# Force Field

The v1 force field is the canonical CRANBERRY model. 5SPN and RACER are not public v1 APIs.

Current packaged model:

```text
cranberry-v1-alpha.1
```

The package resolves `default` to the current alpha model during development.

The alpha model is stored as one packaged HDF5 file:

```text
cranberry/data/forcefields/cranberry-v1-alpha.1.h5
```

This file contains the canonical bonded, sugar-pucker, WCA, spline, stacking, and pairing parameters needed by `CranberryForceField`. The legacy `run_rna.py --angle-scaling 0.1` behavior is baked into the packaged angle spring constants rather than exposed as a public option.

Named energy components are:

```text
bond
angle
dihedral
pucker
stacking35
stacking55
stacking33
pairing
wca
spline
electrostatic
```

Force-group integer IDs are internal and should not be treated as public API.

## Pairing execution

Cranberry may represent the `pairing` component with either an OpenMM
`CustomCompoundBondForce` or donor-specialized `CustomHbondForce` objects. This is an execution
optimization only: both paths use the same pairing parameters, guarded geometry, 0.8 nm dynamic
cutoff, sequence-neighbor exclusions, force group, and reported energy component.

`CranberryForceField.createSystem()` selects the representation once from the fixed topology:

- Systems with at most 8,192 valid directed pairing candidates use the pair-parallel compound
  path when more than one donor channel is active or a type pair requires multiple geometry slots.
- Larger systems use the H-bond neighbor-list path to bound construction cost, storage, and
  all-pairs runtime work.
- One-channel, one-slot systems use the H-bond path because explicit pair traversal provides no
  repeatable performance benefit.

The compound path lists every chemically valid candidate, including candidates initially outside
the cutoff. Their current coordinates are evaluated on every force calculation, and
`step(0.8-r)` makes candidates outside the cutoff contribute zero energy and force. Consequently,
pairs can enter and leave the cutoff normally during MD without rebuilding the `System`.

This is not a runtime switch: the representation does not change during a trajectory. It is also
not platform-aware because OpenMM chooses the CPU or GPU platform later, when a `Context` is
created. The threshold and representation are internal implementation details rather than public
model parameters, so callers should use the named `pairing` energy component instead of depending
on a particular underlying OpenMM force class.
