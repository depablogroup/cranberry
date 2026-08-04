#!/usr/bin/env bash
set -euo pipefail

common_excludes=(
  --glob '!.git/**'
  --glob '!docs/_build/**'
  --glob '!build/**'
  --glob '!dist/**'
  --glob '!*.egg-info/**'
)

if path_matches="$(
  rg --hidden -n '/home/[^<[:space:]]|/Users/[^<[:space:]]' .     "${common_excludes[@]}"     --glob '!tests/**'     --glob '!scripts/check-publication-privacy.sh'
)"; then
  printf '%s\n' "Machine-specific home paths found:" "$path_matches" >&2
  exit 1
elif [[ $? -ne 1 ]]; then
  exit 2
fi

if host_matches="$(
  rg -n '^host:' benchmarks/results     | grep -v 'host: <host>$'
)"; then
  printf '%s\n' "Unsanitized benchmark hostnames found:" "$host_matches" >&2
  exit 1
elif [[ ${PIPESTATUS[0]} -gt 1 ]]; then
  exit 2
fi
