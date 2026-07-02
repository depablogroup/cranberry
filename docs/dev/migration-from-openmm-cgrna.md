# Migration From OpenMM-CGRNA

This developer-only document will track how legacy `OpenMM-CGRNA` behavior maps into the new `cranberry-rna` package.

Initial decisions:

- Temporary reference outputs should be generated from `CRANBERRY_workshop/cranberry-workshop-919771f4-gpu/`, not from the full legacy repo by default. See `docs/dev/reference-output-generation.md`.

- `D_CGRNA` is removed.
- Old `core.*` imports are not supported.
- v1 supports canonical CRANBERRY only, not public 5SPN/RACER APIs.
- Migration guidance should document important old commands and their new equivalents.
- Changes to legacy parameter files during construction of `cranberry-v1.h5` should be recorded here.
