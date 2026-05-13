#!/usr/bin/env bash
set -euo pipefail

# Defaults
INPUT="/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
OUTDIR="/mnt/datasets/mscoco_converted"
DATASET_NAME="mscoco"
FORMATS="webdataset,tfrecord,ffrecord,tdp"
VARIANTS="B,F"
SHARD_SIZE=1000
PYTHON="/home/llm/experiment/dataset_Analysis_Modeling/.env/bin/python"
YES=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: regen_mscoco.sh [options]

Options:
  --input PATH         Input FFSlim .ffr (default: /mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr)
  --outdir PATH        Output root (default: /mnt/datasets/mscoco_converted)
  --dataset-name NAME  Dataset name (default: mscoco)
  --formats LIST       Comma-separated formats (default: webdataset,tfrecord,ffrecord,tdp)
  --variants LIST      Comma-separated variants (default: B,F)
  --shard-size N       Shard size for webdataset (default: 1000)
  --python PATH        Python executable to use
  --yes                Confirm deletion and regeneration
  --dry-run            Print actions without executing
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --formats) FORMATS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --yes) YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
 done

if [[ $YES -ne 1 ]]; then
  echo "Refusing to delete outputs without --yes"
  exit 2
fi

IFS=',' read -r -a FORMAT_LIST <<< "$FORMATS"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"

run_cmd() {
  local cmd=("$@")
  echo "${cmd[*]}"
  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi
  "${cmd[@]}"
}

for variant in "${VARIANT_LIST[@]}"; do
  for fmt in "${FORMAT_LIST[@]}"; do
    out_dir="$OUTDIR/${DATASET_NAME}_${fmt}_${variant}"
    if [[ -d "$out_dir" ]]; then
      run_cmd rm -rf "$out_dir"
    fi
  done
 done

run_cmd "$PYTHON" "/home/llm/experiment/dataset_Analysis_Modeling/generaters/run_convert_mscoco.py" \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --dataset-name "$DATASET_NAME" \
  --formats "$FORMATS" \
  --variants "$VARIANTS" \
  --shard-size "$SHARD_SIZE"

run_cmd "$PYTHON" "/home/llm/experiment/dataset_Analysis_Modeling/scripts/verify_mscoco_outputs.py" \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --dataset-name "$DATASET_NAME"
