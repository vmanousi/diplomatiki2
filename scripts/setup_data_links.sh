#!/bin/bash
# Symlinks the datasets used by configs/*.yaml into data/, so the same
# repo-relative paths (data/Labeled_Images_GastroHun, data/official_splits_GastroHun,
# data/Gastrovision_webdataset) work on any machine, pointing at wherever
# that machine actually stores the datasets.
#
# Usage: scripts/setup_data_links.sh <path-to-datasets-root>
# Example (cluster): scripts/setup_data_links.sh /home/v/vmanousi/Datasets

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <path-to-datasets-root>"
  echo "Example: $0 /home/v/vmanousi/Datasets"
  exit 1
fi

DATASETS_ROOT="$1"
mkdir -p data

ln -sfn "$DATASETS_ROOT/GastroHun/Labeled_Images_GastroHun" data/Labeled_Images_GastroHun
ln -sfn "$DATASETS_ROOT/GastroHun/official_splits_GastroHun" data/official_splits_GastroHun
ln -sfn "$DATASETS_ROOT/Gastrovision_webdataset" data/Gastrovision_webdataset

echo "Linked datasets from $DATASETS_ROOT into data/"
ls -la data/
