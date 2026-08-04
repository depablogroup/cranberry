#!/usr/bin/env bash
set -euo pipefail

path_matches="$(
  git grep -nE '/home/[^<[:space:]]|/Users/[^<[:space:]]' -- .     ':(exclude)tests/**'     ':(exclude)scripts/check-publication-privacy.sh'     || true
)"
if [[ -n "$path_matches" ]]; then
  printf '%s\n' "Machine-specific home paths found:" "$path_matches" >&2
  exit 1
fi

host_matches="$(
  git grep -n '^host:' -- benchmarks/results || true
)"
unsanitized_hosts=""
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  case "$line" in
    *"host: <host>") ;;
    *) unsanitized_hosts+="$line"$'\n' ;;
  esac
done <<< "$host_matches"
if [[ -n "$unsanitized_hosts" ]]; then
  printf '%s\n' "Unsanitized benchmark hostnames found:" "$unsanitized_hosts" >&2
  exit 1
fi
