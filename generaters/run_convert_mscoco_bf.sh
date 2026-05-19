#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/datasets/mscoco_converted}"
DATASET="${DATASET:-mscoco}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
LOG_FILE="${LOG_FILE:-$ROOT/${DATASET}_bf_convert.log}"
ONLY_FORMATS=""
ONLY_MODES=""
F2B_FROM_B2F=0

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_convert_mscoco_bf.sh [options]

Options:
  --root PATH           Converted root (default: env ROOT)
  --dataset NAME        Dataset name (default: env DATASET)
  --python PATH         Python executable (default: env PYTHON_BIN)
  --shard-size N        WebDataset shard size (default: env SHARD_SIZE)
  --log-file PATH       Log file path (default: env LOG_FILE)
  --only-format LIST    Comma-separated formats to run (tdp,webdataset,tfrecord,ffrecord)
  --only-mode LIST      Comma-separated modes to run (B2F,F2B)
  --f2b-from-b2f        Use *_B2F outputs as F2B inputs
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --only-format) ONLY_FORMATS="$2"; shift 2 ;;
    --only-mode) ONLY_MODES="$2"; shift 2 ;;
    --f2b-from-b2f) F2B_FROM_B2F=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

run_ffrecord() {
  local mode="$1"
  local in_file="$2"
  local out_file="$3"
  mkdir -p "$(dirname "$out_file")"
  "$PYTHON_BIN" "$BASE_DIR/generaters/convert_ffrecord_bf.py" \
    --mode "$mode" \
    --input "$in_file" \
    --output "$out_file" \
    --dataset "$DATASET" \
    --format "ffrecord" \
    --log-file "$LOG_FILE"
}

run_tfrecord() {
  local mode="$1"
  local in_file="$2"
  local out_file="$3"
  mkdir -p "$(dirname "$out_file")"
  "$PYTHON_BIN" "$BASE_DIR/generaters/convert_tfrecord_bf.py" \
    --mode "$mode" \
    --input "$in_file" \
    --output "$out_file" \
    --dataset "$DATASET" \
    --format "tfrecord" \
    --log-file "$LOG_FILE"
}

run_webdataset() {
  local mode="$1"
  local in_dir="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" "$BASE_DIR/generaters/convert_webdataset_bf.py" \
    --mode "$mode" \
    --input-dir "$in_dir" \
    --output-dir "$out_dir" \
    --shard-size "$SHARD_SIZE" \
    --dataset "$DATASET" \
    --format "webdataset" \
    --log-file "$LOG_FILE"
}

run_tdp() {
  local mode="$1"
  local in_dir="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" "$BASE_DIR/generaters/convert_tdp_bf.py" \
    --mode "$mode" \
    --input-dir "$in_dir" \
    --output-dir "$out_dir" \
    --dataset "$DATASET" \
    --format "tdp" \
    --log-file "$LOG_FILE"
}

FFRECORD_B="$ROOT/${DATASET}_ffrecord_B/${DATASET}_ffrecord_B.ffr"
FFRECORD_F="$ROOT/${DATASET}_ffrecord_F/${DATASET}_ffrecord_F.ffr"
FFRECORD_B2F="$ROOT/${DATASET}_ffrecord_B2F/${DATASET}_ffrecord_B2F.ffr"
TFRECORD_B="$ROOT/${DATASET}_tfrecord_B/${DATASET}_tfrecord_B.tfrecord"
TFRECORD_F="$ROOT/${DATASET}_tfrecord_F/${DATASET}_tfrecord_F.tfrecord"
TFRECORD_B2F="$ROOT/${DATASET}_tfrecord_B2F/${DATASET}_tfrecord_B2F.tfrecord"
WEBDATASET_B="$ROOT/${DATASET}_webdataset_B"
WEBDATASET_F="$ROOT/${DATASET}_webdataset_F"
WEBDATASET_B2F="$ROOT/${DATASET}_webdataset_B2F"
TDP_B="$ROOT/${DATASET}_tdp_B"
TDP_F="$ROOT/${DATASET}_tdp_F"
TDP_B2F="$ROOT/${DATASET}_tdp_B2F"

FORMATS=("ffrecord" "tfrecord" "webdataset" "tdp")
MODES=("B2F" "F2B")
if [[ -n "$ONLY_FORMATS" ]]; then
  IFS=',' read -r -a FORMATS <<< "$ONLY_FORMATS"
fi
if [[ -n "$ONLY_MODES" ]]; then
  IFS=',' read -r -a MODES <<< "$ONLY_MODES"
fi

for fmt in "${FORMATS[@]}"; do
  case "$fmt" in
    ffrecord)
      for mode in "${MODES[@]}"; do
        if [[ "$mode" = "B2F" ]]; then
          run_ffrecord B2F "$FFRECORD_B" "$FFRECORD_B2F"
        else
          if [[ "$F2B_FROM_B2F" -eq 1 ]]; then
            run_ffrecord F2B "$FFRECORD_B2F" "$ROOT/${DATASET}_ffrecord_F2B/${DATASET}_ffrecord_F2B.ffr"
          else
            run_ffrecord F2B "$FFRECORD_F" "$ROOT/${DATASET}_ffrecord_F2B/${DATASET}_ffrecord_F2B.ffr"
          fi
        fi
      done
      ;;
    tfrecord)
      for mode in "${MODES[@]}"; do
        if [[ "$mode" = "B2F" ]]; then
          run_tfrecord B2F "$TFRECORD_B" "$TFRECORD_B2F"
        else
          if [[ "$F2B_FROM_B2F" -eq 1 ]]; then
            run_tfrecord F2B "$TFRECORD_B2F" "$ROOT/${DATASET}_tfrecord_F2B/${DATASET}_tfrecord_F2B.tfrecord"
          else
            run_tfrecord F2B "$TFRECORD_F" "$ROOT/${DATASET}_tfrecord_F2B/${DATASET}_tfrecord_F2B.tfrecord"
          fi
        fi
      done
      ;;
    webdataset)
      for mode in "${MODES[@]}"; do
        if [[ "$mode" = "B2F" ]]; then
          run_webdataset B2F "$WEBDATASET_B" "$WEBDATASET_B2F"
        else
          if [[ "$F2B_FROM_B2F" -eq 1 ]]; then
            run_webdataset F2B "$WEBDATASET_B2F" "$ROOT/${DATASET}_webdataset_F2B"
          else
            run_webdataset F2B "$WEBDATASET_F" "$ROOT/${DATASET}_webdataset_F2B"
          fi
        fi
      done
      ;;
    tdp)
      for mode in "${MODES[@]}"; do
        if [[ "$mode" = "B2F" ]]; then
          run_tdp B2F "$TDP_B" "$TDP_B2F"
        else
          if [[ "$F2B_FROM_B2F" -eq 1 ]]; then
            run_tdp F2B "$TDP_B2F" "$ROOT/${DATASET}_tdp_F2B"
          else
            run_tdp F2B "$TDP_F" "$ROOT/${DATASET}_tdp_F2B"
          fi
        fi
      done
      ;;
    *)
      echo "Unknown format: $fmt" >&2
      exit 2
      ;;
  esac
done
