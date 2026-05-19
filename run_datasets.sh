#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llm/experiment/dataset_Analysis_Modeling"
PY="$ROOT/.env/bin/python"

DATASETS=(flickr30k flickr30k_entities mscoco vqa2 gqa Clotho_E MSRVTT)
FORMATS=(tdp webdataset tfrecord ffrecord)
VARIANTS=(B F)

OUT_ROOT="/mnt/datasets/converted_tmp"
MONITOR_DIR="$ROOT/monitor"
DATASET_INFO_LOG="$ROOT/dataset-info.log"
MEMORY_PROFILE_TXT="$ROOT/datasets-memory.txt"

# cgroup memory control
# 1=enable, 0=disable globally
USE_CGROUP_LIMIT="${USE_CGROUP_LIMIT:-0}"
# Comma-separated skip list. Supports two key forms:
#   1) fmt_variant           e.g. tfrecord_F
#   2) dataset|fmt|variant   e.g. vqa2|tdp|B
NO_CGROUP_KEYS="${NO_CGROUP_KEYS:-}"

# Guard fallback chain for retry when memory limit fails.
# Example override: MEMORY_GUARD_STEPS="16,24,32,48,64"
MEMORY_GUARD_STEPS="${MEMORY_GUARD_STEPS:-16,32,48,64,80,96,128,160,192,256}"

declare -A ANON_PEAK_MB

normalize_dataset_name() {
  local raw="$1"
  local lower
  lower=$(echo "$raw" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    flickr30k) echo "flickr30k" ;;
    flickr30k_entities) echo "flickr30k_entities" ;;
    mscoco) echo "mscoco" ;;
    vqa2) echo "vqa2" ;;
    gqa) echo "gqa" ;;
    clotho_e) echo "Clotho_E" ;;
    msrvtt) echo "MSRVTT" ;;
    *) echo "" ;;
  esac
}

should_use_cgroup() {
  local ds="$1"
  local fmt="$2"
  local variant="$3"

  if [[ "$USE_CGROUP_LIMIT" != "1" ]]; then
    return 1
  fi

  local key1="${fmt}_${variant}"
  local key2="${ds}|${fmt}|${variant}"

  if [[ -n "$NO_CGROUP_KEYS" ]]; then
    IFS=',' read -r -a arr <<< "$NO_CGROUP_KEYS"
    for raw in "${arr[@]}"; do
      local item
      item=$(echo "$raw" | tr -d '[:space:]')
      if [[ -z "$item" ]]; then
        continue
      fi
      if [[ "$item" == "$key1" || "$item" == "$key2" ]]; then
        return 1
      fi
    done
  fi

  return 0
}

load_memory_profile() {
  local profile="$1"
  if [ ! -f "$profile" ]; then
    echo "Warning: memory profile not found: $profile" >&2
    return 0
  fi

  while IFS=$'\t' read -r ds mode fmt mb; do
    [ -z "$ds" ] && continue
    ds=$(normalize_dataset_name "$ds")
    [ -z "$ds" ] && continue
    ANON_PEAK_MB["${ds}|${fmt}|${mode}"]="$mb"
  done < <(
    awk '
      BEGIN { mode = ""; OFS = "\t" }
      { gsub(/\r/, "", $0) }
      /进程匿名页开销-B/ { mode = "B"; next }
      /进程匿名页开销-F/ { mode = "F"; next }
      /^Workload/ { next }
      mode != "" {
        n = split($0, a, /[[:space:]]+/)
        if (n >= 5) {
          ds = tolower(a[1])
          gsub(/[^a-z0-9_]/, "", ds)
          print ds, mode, "tdp", a[2]
          print ds, mode, "webdataset", a[3]
          print ds, mode, "tfrecord", a[4]
          print ds, mode, "ffrecord", a[5]
        }
      }
    ' "$profile"
  )
}

compute_limit_mb_with_guard() {
  local anon_peak_mb="$1"
  local guard_mb="$2"
  awk -v v="$anon_peak_mb" -v guard="$guard_mb" 'BEGIN {
    limit = int(v + guard + 0.999999)
    if (limit < 1) limit = 1
    print limit
  }'
}

run_loader_direct() {
  local cmd=("$@")
  "${cmd[@]}"
}

run_loader_with_limit() {
  local limit_mb="$1"
  shift
  local cmd=("$@")

  if command -v systemd-run >/dev/null 2>&1; then
    local unit="loader_${dataset}_${fmt}_${variant}_$$"
    sudo systemd-run --quiet --wait --collect \
      --unit "$unit" \
      --property "MemoryAccounting=yes" \
      --property "MemoryMax=${limit_mb}M" \
      --property "MemorySwapMax=0" \
      "${cmd[@]}"
  else
    echo "Warning: systemd-run not found, fallback without cgroup MemoryMax." >&2
    "${cmd[@]}"
  fi
}

run_loader_with_guard_retry() {
  local anon_peak_mb="$1"
  shift
  local cmd=("$@")

  local last_rc=1
  IFS=',' read -r -a guard_steps <<< "$MEMORY_GUARD_STEPS"

  for guard in "${guard_steps[@]}"; do
    guard=$(echo "$guard" | tr -d '[:space:]')
    if [[ -z "$guard" || ! "$guard" =~ ^[0-9]+$ ]]; then
      continue
    fi

    limit_mb=$(compute_limit_mb_with_guard "$anon_peak_mb" "$guard")
    echo "Applying cgroup memory limit: ${limit_mb}MB (dataset=$dataset, format=$fmt, mode=$variant, guard=${guard}MB)"

    if run_loader_with_limit "$limit_mb" \
      "${cmd[@]}" \
      --memory "$limit_mb"; then
      export LAST_SUCCESS_GUARD_MB="$guard"
      return 0
    fi

    last_rc=$?
    echo "Loader failed at guard=${guard}MB (exit=$last_rc), retrying with larger guard..." >&2
  done

  return "$last_rc"
}

load_memory_profile "$MEMORY_PROFILE_TXT"

for dataset in "${DATASETS[@]}"; do
  ffslim="/mnt/datasets/${dataset}_FFSlim/mscoco_caption.ffr"
  conv_root="$OUT_ROOT/${dataset}_converted"

  for fmt in "${FORMATS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      echo "==> $dataset $fmt $variant"

      # 转换
      "$PY" "$ROOT/generaters/run_convert_mscoco.py" \
        --input "$ffslim" \
        --outdir "$conv_root" \
        --dataset-name "$dataset" \
        --formats "$fmt" \
        --variants "$variant"

      # 记录数据集与索引文件占用
      conv_dir="$conv_root/${dataset}_${fmt}_${variant}"
      dataset_du_bytes=0
      index_du_bytes=0
      if [ -d "$conv_dir" ]; then
        dataset_du_bytes=$(du -sb "$conv_dir" 2>/dev/null | cut -f1 || echo 0)
      fi

      if [ "$fmt" = "ffrecord" ]; then
        index_du_bytes=0
      else
        if [ -d "$conv_dir" ]; then
          index_sum=0
          case "$fmt" in
            tdp) index_glob="*.csv" ;;
            tfrecord|webdataset) index_glob="*.jsonl" ;;
            *) index_glob="*index*" ;;
          esac
          while IFS= read -r -d '' f; do
            size=$(du -sb "$f" 2>/dev/null | cut -f1 || echo 0)
            index_sum=$((index_sum + size))
          done < <(find "$conv_dir" -type f -iname "$index_glob" -print0)
          index_du_bytes=$index_sum
        fi
      fi
      printf "%s_%s_%s %s %s\n" "$dataset" "${fmt}" "${variant}" "$dataset_du_bytes" "$index_du_bytes" >> "$DATASET_INFO_LOG"

      loader_cmd=(
        "$ROOT/loaders/run_all_loaders.sh"
        --datasets "$dataset"
        --root "$conv_root"
        --only "${fmt}_${variant}"
        --monitor 1
        --monitor-dir "$MONITOR_DIR"
      )

      if should_use_cgroup "$dataset" "$fmt" "$variant"; then
        profile_key="${dataset}|${fmt}|${variant}"
        anon_peak_mb="${ANON_PEAK_MB[$profile_key]:-}"
        if [ -z "$anon_peak_mb" ]; then
          echo "Warning: no memory profile for $profile_key, fallback fixed limit 256MB" >&2
          run_loader_with_limit 256 "${loader_cmd[@]}" --memory 256
        else
          run_loader_with_guard_retry "$anon_peak_mb" "${loader_cmd[@]}"
        fi
      else
        echo "cgroup limit disabled for ${dataset}|${fmt}|${variant}; running without MemoryMax"
        run_loader_direct "${loader_cmd[@]}" --memory 0
      fi

      # 使用 run_all_loaders.sh 自带的监控，调用 analyze/plot_monitor.py 汇总
      echo "==> Summarize monitor"
      "$PY" "$ROOT/analyze/plot_monitor.py" --input "$MONITOR_DIR"

      # 删除本轮转换产物
      rm -rf "$conv_root/${dataset}_${fmt}_${variant}"
    done
  done

  rm -rf "$conv_root"
done
