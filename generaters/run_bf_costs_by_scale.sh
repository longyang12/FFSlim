#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/llm/experiment/dataset_Analysis_Modeling"
PY="$ROOT_DIR/.env/bin/python"
CONFIG_BASE="$ROOT_DIR/generaters/config/base_yaml/mscoco.yaml"

BASE_MEDIA=100000
MULTS=(1 2 4 8 16)
FORMATS=(tdp webdataset tfrecord ffrecord)

OUT_ROOT="/mnt/datasets/mscoco_variable"
SHARD_SIZE=1000

# Drop filesystem caches before each conversion measurement
_drop_caches() {
  sudo bash -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

_log_dataset_size() {
  local dataset="$1"
  local fmt="$2"
  local mode="$3"
  local path="$4"
  local size_bytes=0

  if [[ -e "$path" ]]; then
    size_bytes=$(du -sb "$path" 2>/dev/null | cut -f1 || echo 0)
  fi

  printf "%s %s %s %s\n" "$dataset" "$fmt" "$mode" "$size_bytes" >> "$log_file"
}

# 生成临时 config：只改 media_num 和 output_root
_make_config() {
  local media_num="$1"
  local out_root="$2"
  local tmp_cfg="$3"
  awk -v m="$media_num" -v o="$out_root" '
    /^media_num:/ {$0="media_num: " m}
    /^output_root:/ {$0="output_root: " o}
    {print}
  ' "$CONFIG_BASE" > "$tmp_cfg"
}

for mult in "${MULTS[@]}"; do
  media_num=$((BASE_MEDIA * mult))
  dataset_name="mscoco_${media_num}"
  ffslim_root="$OUT_ROOT/${dataset_name}_ffslim"
  conv_root="$OUT_ROOT/${dataset_name}_converted"
  log_file="$OUT_ROOT/bf_convert.log"

  echo "==> SCALE x${mult} (media_num=${media_num})"

  tmp_cfg="$(mktemp /tmp/mscoco_${media_num}_XXXX.yaml)"
  _make_config "$media_num" "$ffslim_root" "$tmp_cfg"

  echo "==> Generate FFSlim"
  "$PY" "$ROOT_DIR/generaters/generate_FFSlim_by_config.py" --config "$tmp_cfg"
  rm -f "$tmp_cfg"

  for fmt in "${FORMATS[@]}"; do
    echo "==> Convert format: $fmt (B only)"
    "$PY" "$ROOT_DIR/generaters/run_convert_mscoco.py" \
      --input "$ffslim_root/mscoco_caption.ffr" \
      --outdir "$conv_root" \
      --dataset-name "$dataset_name" \
      --formats "$fmt" \
      --variants "B"

    echo "==> Measure $fmt B2F"
    _drop_caches
    ROOT="$conv_root" DATASET="$dataset_name" PYTHON_BIN="$PY" SHARD_SIZE="$SHARD_SIZE" LOG_FILE="$log_file" \
      bash "$ROOT_DIR/generaters/run_convert_mscoco_bf.sh" \
      --only-format "$fmt" \
      --only-mode "B2F"

    _log_dataset_size "$dataset_name" "$fmt" "B2F" "$conv_root/${dataset_name}_${fmt}_B2F"

    # B2F 输出即为 F，删除 B
    rm -rf "$conv_root/${dataset_name}_${fmt}_B"

    echo "==> Measure $fmt F2B"
    _drop_caches
    ROOT="$conv_root" DATASET="$dataset_name" PYTHON_BIN="$PY" SHARD_SIZE="$SHARD_SIZE" LOG_FILE="$log_file" \
      bash "$ROOT_DIR/generaters/run_convert_mscoco_bf.sh" \
      --only-format "$fmt" \
      --only-mode "F2B" \
      --f2b-from-b2f

    _log_dataset_size "$dataset_name" "$fmt" "F2B" "$conv_root/${dataset_name}_${fmt}_F2B"

    # 清理 B2F / F2B
    rm -rf "$conv_root/${dataset_name}_${fmt}_B2F"
    rm -rf "$conv_root/${dataset_name}_${fmt}_F2B"
  done

  echo "==> Clean FFSlim + converted root"
  rm -rf "$ffslim_root" "$conv_root"
done
