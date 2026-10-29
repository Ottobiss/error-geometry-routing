#!/usr/bin/env bash
# Run from this extracted delivery directory with GitHub write access.
set -euo pipefail
payload_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
branch='results/spatial-router-acdc-20261028'
checkout_dir="${1:-$HOME/spatial-routing-sae-publication}"
if [[ -e "$checkout_dir" ]]; then
  echo "Destination exists: $checkout_dir. Choose a new checkout directory." >&2
  exit 1
fi
git clone --branch "$branch" --single-branch https://github.com/Ottobiss/error-geometry-routing.git "$checkout_dir"
destination="$checkout_dir/experiments/spatial_routing_sae"
mkdir -p "$destination"
cp -R "$payload_dir"/. "$destination"/
git -C "$checkout_dir" add --force experiments/spatial_routing_sae
git -C "$checkout_dir" commit -m 'Document SpatialRouter and AE experiments with reproducible ACDC metrics and checkpoints'
git -C "$checkout_dir" push --set-upstream origin "$branch"
