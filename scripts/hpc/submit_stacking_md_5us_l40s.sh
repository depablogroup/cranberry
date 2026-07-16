#!/usr/bin/env bash
# Compatibility wrapper. The current stacking production target is 3 us.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/submit_stacking_md_3us_l40s.sh"
