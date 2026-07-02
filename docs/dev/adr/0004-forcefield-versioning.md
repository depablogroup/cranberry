# ADR 0004: Force-Field Versioning

## Status

Accepted.

## Context

The CRANBERRY force field is tied to a paper under review. The project needs a way to test and package the v1 model before final publication without creating reproducibility problems.

## Decision

Use alpha, beta, and release-candidate versions before the final paper-ready release:

- `cranberry-rna 1.0.0a1` with `cranberry-v1-alpha.1`
- `cranberry-rna 1.0.0b1` with `cranberry-v1-beta.1`
- `cranberry-rna 1.0.0rc1` with `cranberry-v1-rc.1`
- `cranberry-rna 1.0.0` with `cranberry-v1`

Do not change the default scientific model in patch releases. Patch releases are for software fixes only. Scientific parameter/model changes require a minor or major release.

## Consequences

- Users can trust that `1.0.x` does not silently change scientific behavior.
- Review-driven parameter changes can still happen before the final `1.0.0`.
- Release notes must clearly distinguish software fixes from force-field changes.
