#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_CONFIG="$ROOT_DIR/generaters/config/base_yaml/cc3m.yaml"
GEN_CONFIG_DIR="$ROOT_DIR/generaters/config/generate_yaml"
GENERATOR="$ROOT_DIR/generaters/generate_FFSlim_by_config.py"
PYTHON_BIN="$ROOT_DIR/.env/bin/python"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

if [[ ! -f "$GENERATOR" ]]; then
  echo "Generator not found: $GENERATOR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "$GEN_CONFIG_DIR"

media_num=$(awk -F': ' '/^media_num:/ {print $2}' "$BASE_CONFIG")
std_quantity_ratio=$(awk -F': ' '/^std_quantity_ratio:/ {print $2}' "$BASE_CONFIG")
avg_media_size=$(awk -F': ' '/^avg_media_size:/ {print $2}' "$BASE_CONFIG")
std_media_size=$(awk -F': ' '/^std_media_size:/ {print $2}' "$BASE_CONFIG")
avg_text_size=$(awk -F': ' '/^avg_text_size:/ {print $2}' "$BASE_CONFIG")
std_text_size=$(awk -F': ' '/^std_text_size:/ {print $2}' "$BASE_CONFIG")
base_output_root=$(awk -F': ' '/^output_root:/ {print $2}' "$BASE_CONFIG")
output_file=$(awk -F': ' '/^output_file:/ {print $2}' "$BASE_CONFIG")
manifest_file=$(awk -F': ' '/^manifest_file:/ {print $2}' "$BASE_CONFIG")
seed=$(awk -F': ' '/^seed:/ {print $2}' "$BASE_CONFIG")

if [[ -z "$base_output_root" ]]; then
  echo "output_root is missing in base config: $BASE_CONFIG" >&2
  exit 1
fi

base_output_root="${base_output_root%/}"

for ratio in $(seq 1 20); do
  cfg="$GEN_CONFIG_DIR/cc3m_ffslim_${ratio}.yaml"
  cat > "$cfg" <<EOF
dataset_name: cc3m_ffslim_${ratio}
media_num: ${media_num}
avg_quantity_ratio: ${ratio}.0
std_quantity_ratio: ${std_quantity_ratio}
avg_media_size: ${avg_media_size}
std_media_size: ${std_media_size}
avg_text_size: ${avg_text_size}
std_text_size: ${std_text_size}
output_root: ${base_output_root}_ffslim_${ratio}
output_file: ${output_file}
manifest_file: ${manifest_file}
seed: ${seed}
EOF

  echo "[${ratio}/20] generating FFSlim with $cfg"
  "$PYTHON_BIN" "$GENERATOR" --config "$cfg"
done

echo "Done. Generated configs in $GEN_CONFIG_DIR and FFSlim datasets in ${base_output_root}_ffslim_<ratio>."