#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.env/bin/python"
MANIFEST_ROOT="/mnt/datasets/generate_datasets"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

# 仅示例默认算法列表；后续可按实验需求扩展
FFSLIM_ALGOS=(lru quiver coordl dgim_lru dgim_quiver dgim_coordl dgim_iacma dgim_iacma_admission dgim_iacma_with_memory dgim_iacma_with_memory_admission)
TDP_ALGOS=(lru quiver coordl iacma iacma_admission iacma_with_memory iacma_with_memory_admission)

CACHE_SIZE="${1:-0}"
REQUEST_COUNT="${2:-0}"

find "$MANIFEST_ROOT" -maxdepth 2 -name generated_manifest.json | sort | while read -r manifest; do
  dataset_dir="$(dirname "$manifest")"
  dataset_name="$(basename "$dataset_dir")"

  if [[ "$dataset_name" == *"ffslim"* ]]; then
    for algo in "${FFSLIM_ALGOS[@]}"; do
      echo "[SIM][FFSlim] dataset=$dataset_name algo=$algo cache=$CACHE_SIZE req=$REQUEST_COUNT"
      "$PYTHON_BIN" "$ROOT_DIR/loaders/random_FFSlim_loader_sim.py" \
        --manifest "$manifest" \
        --num "$CACHE_SIZE" \
        --cache-algorithm "$algo" \
        --index-structure hash_index \
        --request-count "$REQUEST_COUNT"
    done
  elif [[ "$dataset_name" == *"tdp"* ]]; then
    for algo in "${TDP_ALGOS[@]}"; do
      echo "[SIM][TDP] dataset=$dataset_name algo=$algo cache=$CACHE_SIZE req=$REQUEST_COUNT"
      "$PYTHON_BIN" "$ROOT_DIR/loaders/random_TDP_loader_sim.py" \
        --manifest "$manifest" \
        --num "$CACHE_SIZE" \
        --cache-algorithm "$algo" \
        --request-count "$REQUEST_COUNT"
    done
  fi
done

echo "Sim benchmark finished. See $ROOT_DIR/output_sim.log"
