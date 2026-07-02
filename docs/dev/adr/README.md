# Architecture Decision Records

Architecture Decision Records, or ADRs, are short developer notes that capture important project decisions after they are made.

Use ADRs when a decision is important enough that future contributors will ask "why did we do it this way?" Examples for this project include package layout, force-field versioning, output file contracts, dependency extras, and public API shape.

Keep each ADR focused:

- Context: what problem or constraint forced a decision?
- Decision: what did we choose?
- Consequences: what gets easier, what gets harder, and what must remain true?

ADRs are not public user documentation by default. They should stay in the GitHub repository under `docs/dev/adr/` and be excluded from the public Sphinx build unless we intentionally publish developer documentation later.
