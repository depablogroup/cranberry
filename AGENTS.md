# Cranberry Package Instructions

This is the new `cranberry-rna` distribution. The Python import package is `cranberry`.

Before changing package architecture, force-field behavior, CLI contracts, output formats, or tests, read `docs/dev/program.md` and `docs/dev/cranberry-v1-plan.md`.

Keep this file concise. Put long design notes, migration notes, release decisions, and detailed operating rules in `docs/dev/`.

## Commit Review Workflow

Before committing non-trivial code, package, CLI, validation, or documentation-architecture changes, create or update a developer-facing code review report under `docs/dev/progress/`. The report should summarize the important code changes, relevant command outputs, tests, and risks. Wait for explicit user approval before committing the report and related changes.
