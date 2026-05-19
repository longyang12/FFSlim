#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llm/experiment/dataset_Analysis_Modeling"
PY="$ROOT/.env/bin/python"
CONFIG_BASE="$ROOT/generaters/config/base_yaml/mscoco.yaml"

BASE_MEDIA=100000
MULTS=(1 2 4 8 16)
ORDER=(tdp_F  webdataset_F  tfrecord_F  ffrecord_F tdp_B webdataset_B tfrecord_B ffrecord_B)

OUT_ROOT="/mnt/datasets/mscoco_variable"
MONITOR_DIR="$ROOT/monitor"

# 生成临时 config：只改 media_num 和 output_root
make_config() {
  local media_num="$1"
  local out_root="$2"
  local tmp_cfg="$3"
  # 只替换 media_num 和 output_root
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

  echo "==> SCALE x${mult} (media_num=${media_num})"

  tmp_cfg="$(mktemp /tmp/mscoco_${media_num}_XXXX.yaml)"
  make_config "$media_num" "$ffslim_root" "$tmp_cfg"

  echo "==> Generate FFSlim"
  "$PY" "$ROOT/generaters/generate_FFSlim_by_config.py" --config "$tmp_cfg"
  rm -f "$tmp_cfg"

  for key in "${ORDER[@]}"; do
    fmt="${key%_*}"
    variant="${key#*_}"

    echo "==> Convert format: $fmt ($variant)"
    "$PY" "$ROOT/generaters/run_convert_mscoco.py" \
      --input "$ffslim_root/mscoco_caption.ffr" \
      --outdir "$conv_root" \
      --dataset-name "$dataset_name" \
      --formats "$fmt" \
      --variants "$variant"

    # 记录数据集磁盘占用与索引文件占用到 dataset-info.log
    conv_dir="$conv_root/${dataset_name}_${fmt}_${variant}"
    dataset_du_bytes=0
    index_du_bytes=0
    if [ -d "$conv_dir" ]; then
      dataset_du_bytes=$(du -sb "$conv_dir" 2>/dev/null | cut -f1 || echo 0)
    fi

    # ffrecord 没有单独索引文件，直接记为 0；其它格式按约定的扩展名统计
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
      else
        index_du_bytes=0
      fi
    fi

    # 输出格式： 数据集_规模_数据格式_F/B 数据集字节数 索引字节数
    printf "%s_%s_%s %s %s\n" "$dataset_name" "${fmt}" "${variant}" "$dataset_du_bytes" "$index_du_bytes" >> "$ROOT/dataset-info.log"

    echo "==> Run loader for $key"
    "$ROOT/loaders/run_all_loaders.sh" \
      --datasets "$dataset_name" \
      --root "$conv_root" \
      --only "$key" \
      --monitor 1 \
      --monitor-dir "$MONITOR_DIR"

    echo "==> Summarize monitor"
    "$PY" "$ROOT/analyze/plot_monitor.py" --input "$MONITOR_DIR"

    echo "==> Clean converted format dir"
    rm -rf "$conv_root/${dataset_name}_${fmt}_${variant}"
  done

  echo "==> Clean FFSlim + converted root"
  rm -rf "$ffslim_root" "$conv_root"
done