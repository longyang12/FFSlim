#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_CONFIG="$ROOT_DIR/generaters/config/base_yaml/gqa.yaml"
GEN_CONFIG_DIR="$ROOT_DIR/generaters/config/generate_yaml"
FFSLIM_GENERATOR="$ROOT_DIR/generaters/generate_FFSlim_by_config.py"
TDP_GENERATOR="$ROOT_DIR/generaters/generate_TDP_by_config.py"
PYTHON_BIN="$ROOT_DIR/.env/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

mkdir -p "$GEN_CONFIG_DIR"

media_num=$(awk -F': ' '/^media_num:/ {print $2}' "$BASE_CONFIG")
avg_media_size=$(awk -F': ' '/^avg_media_size:/ {print $2}' "$BASE_CONFIG")
std_media_size=$(awk -F': ' '/^std_media_size:/ {print $2}' "$BASE_CONFIG")
avg_text_size=$(awk -F': ' '/^avg_text_size:/ {print $2}' "$BASE_CONFIG")
std_text_size=$(awk -F': ' '/^std_text_size:/ {print $2}' "$BASE_CONFIG")
base_output_root=$(awk -F': ' '/^output_root:/ {print $2}' "$BASE_CONFIG")
output_file=$(awk -F': ' '/^output_file:/ {print $2}' "$BASE_CONFIG")
manifest_file=$(awk -F': ' '/^manifest_file:/ {print $2}' "$BASE_CONFIG")
seed=$(awk -F': ' '/^seed:/ {print $2}' "$BASE_CONFIG")

base_output_root="${base_output_root%/}"

# Best-fit gamma params from analyze/output/datasets/gqa_ratio_fit_summary.txt
base_gamma_shape="1.6085751091028782"
base_gamma_scale="8.1264380836193"

# 11 curves: left 5 + baseline + right 5
factors=(0.08 0.24 0.4 0.56 0.72 1.0 1.2 1.4 1.6 1.8 2.0)
total=${#factors[@]}

mapping_file="${base_output_root}_gamma11_mapping.csv"
echo "index,factor,gamma_shape,gamma_scale,mean,std,ffslim_output_root,tdp_output_root" > "$mapping_file"

index=0
for factor in "${factors[@]}"; do
  index=$((index + 1))

  read -r gamma_scale mean std <<<"$($PYTHON_BIN - <<PY
k = float("$base_gamma_shape")
theta = float("$base_gamma_scale") * float("$factor")
mean = k * theta
std = (k ** 0.5) * theta
print(f"{theta} {mean} {std}")
PY
)"

  tag="f$(echo "$factor" | tr '.' 'p')"

  ff_output_root="${base_output_root}_ffslim_gamma_${tag}"
  tdp_output_root="${base_output_root}_tdp_gamma_${tag}"

  ff_cfg="$GEN_CONFIG_DIR/gqa_ffslim_gamma_${tag}.yaml"
  tdp_cfg="$GEN_CONFIG_DIR/gqa_tdp_gamma_${tag}.yaml"

  cat > "$ff_cfg" <<EOF
dataset_name: gqa_ffslim_gamma_${tag}
media_num: ${media_num}
avg_quantity_ratio: ${mean}
std_quantity_ratio: ${std}
avg_media_size: ${avg_media_size}
std_media_size: ${std_media_size}
avg_text_size: ${avg_text_size}
std_text_size: ${std_text_size}
ratio_model: gamma
ratio_gamma_shape: ${base_gamma_shape}
ratio_gamma_scale: ${gamma_scale}
output_root: ${ff_output_root}
output_file: ${output_file}
manifest_file: ${manifest_file}
seed: ${seed}
EOF

  cat > "$tdp_cfg" <<EOF
dataset_name: gqa_tdp_gamma_${tag}
media_num: ${media_num}
avg_quantity_ratio: ${mean}
std_quantity_ratio: ${std}
avg_media_size: ${avg_media_size}
std_media_size: ${std_media_size}
avg_text_size: ${avg_text_size}
std_text_size: ${std_text_size}
ratio_model: gamma
ratio_gamma_shape: ${base_gamma_shape}
ratio_gamma_scale: ${gamma_scale}
output_root: ${tdp_output_root}
csv_file: gqa_tdp_gamma_${tag}.csv
manifest_file: ${manifest_file}
seed: ${seed}
EOF

  echo "${index},${factor},${base_gamma_shape},${gamma_scale},${mean},${std},${ff_output_root},${tdp_output_root}" >> "$mapping_file"

  echo "[$index/$total] generating FFSlim with $ff_cfg"
  "$PYTHON_BIN" "$FFSLIM_GENERATOR" --config "$ff_cfg"

  echo "[$index/$total] generating TDP with $tdp_cfg"
  "$PYTHON_BIN" "$TDP_GENERATOR" --config "$tdp_cfg"
done

echo "Done. Generated 11 FFSlim + 11 TDP datasets for gqa gamma curves."
echo "Mapping saved to: $mapping_file"