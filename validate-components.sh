#!/usr/bin/env bash
# Validate every component descriptor (component.yaml) in the repository against the
# videoflow component schema. Run locally before pushing, or in CI.
#
#   ./validate-components.sh
#
# Requires the `videoflow` package to be importable (pip install videoflow). Each
# sub-package that ships a component.yaml is a marketplace-distributable component.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Portable across macOS (bash 3.2) and Linux — no mapfile.
descriptors=()
while IFS= read -r f; do
  descriptors+=("$f")
done < <(find . -maxdepth 2 -name component.yaml | sort)

if [ "${#descriptors[@]}" -eq 0 ]; then
  echo "No component.yaml descriptors found."
  exit 0
fi

echo "Validating ${#descriptors[@]} component descriptor(s)..."
python -m videoflow.cli component validate "${descriptors[@]}"
