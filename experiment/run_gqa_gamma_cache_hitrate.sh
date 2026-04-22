#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.env/bin/python"
MAPPING_FILE="${1:-/mnt/datasets/generate_datasets/gqa_gamma11_mapping.csv}"
LOG_FILE="${2:-$ROOT_DIR/output_sim.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ ! -f "$MAPPING_FILE" ]]; then
  echo "Mapping file not found: $MAPPING_FILE" >&2
  exit 1
fi

FFSLIM_ALGOS=(lru quiver coordl dgim_lru dgim_quiver dgim_coordl dgim_iacma dgim_iacma_admission dgim_iacma_with_memory dgim_iacma_with_memory_admission)
TDP_ALGOS=(lru quiver coordl iacma iacma_admission iacma_with_memory iacma_with_memory_admission)

python3 - "$MAPPING_FILE" <<'PY'
import csv
from pathlib import Path
import sys

mapping = Path(sys.argv[1])
rows = list(csv.DictReader(mapping.open('r', encoding='utf-8')))
if not rows:
  raise SystemExit('Mapping file is empty')

print('MAPPING_OK')
for row in sorted(rows, key=lambda r: float(r['mean'])):
    print(row['factor'], row['mean'], row['std'], row['ffslim_output_root'], row['tdp_output_root'])
PY

: > "$LOG_FILE"

cache_size=$($PYTHON_BIN - <<'PY'
print(int(round(72139 * 0.4)))
PY
)

request_seed=123

sorted_mapping_file=$(mktemp)
cleanup() {
  rm -f "$sorted_mapping_file"
}
trap cleanup EXIT

python3 - "$MAPPING_FILE" "$sorted_mapping_file" <<'PY'
import csv
import sys

src = sys.argv[1]
dst = sys.argv[2]
with open(src, 'r', encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('Mapping file is empty')

rows.sort(key=lambda r: float(r['mean']))

with open(dst, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
PY

while IFS=, read -r index factor gamma_shape gamma_scale mean std ff_root tdp_root; do
  [[ "$index" == "index" ]] && continue
  index="${index//$'\r'/}"
  factor="${factor//$'\r'/}"
  gamma_shape="${gamma_shape//$'\r'/}"
  gamma_scale="${gamma_scale//$'\r'/}"
  mean="${mean//$'\r'/}"
  std="${std//$'\r'/}"
  ff_root="${ff_root//$'\r'/}"
  tdp_root="${tdp_root//$'\r'/}"

  ff_manifest="$ff_root/generated_manifest.json"
  tdp_manifest="$tdp_root/generated_manifest.json"

  ff_requests=$($PYTHON_BIN - "$ff_manifest" <<'PY'
import json
import sys
from pathlib import Path

mf = Path(sys.argv[1])
data = json.loads(mf.read_text(encoding='utf-8'))
print(int(data.get('realized_stats', {}).get('sample_num', data.get('sample_num'))))
PY
)

  tdp_requests=$($PYTHON_BIN - "$tdp_manifest" <<'PY'
import json
import sys
from pathlib import Path

mf = Path(sys.argv[1])
data = json.loads(mf.read_text(encoding='utf-8'))
print(int(data.get('realized_stats', {}).get('sample_num', data.get('sample_num'))))
PY
)

  echo "[FFSlim] $ff_root cache_size=$cache_size requests=$ff_requests" | tee -a "$LOG_FILE"
  for algo in "${FFSLIM_ALGOS[@]}"; do
    echo "[FFSlim] dataset=$ff_root algo=$algo" | tee -a "$LOG_FILE"
    "$PYTHON_BIN" "$ROOT_DIR/loaders/random_FFSlim_loader_sim.py" \
      --manifest "$ff_manifest" \
      --num "$cache_size" \
      --cache-algorithm "$algo" \
      --index-structure hash_index \
      --request-count "$ff_requests" \
      --shuffle-seed "$request_seed" \
      >> "$LOG_FILE" 2>&1
  done

  echo "[TDP] $tdp_root cache_size=$cache_size requests=$tdp_requests" | tee -a "$LOG_FILE"
  for algo in "${TDP_ALGOS[@]}"; do
    echo "[TDP] dataset=$tdp_root algo=$algo" | tee -a "$LOG_FILE"
    "$PYTHON_BIN" "$ROOT_DIR/loaders/random_TDP_loader_sim.py" \
      --manifest "$tdp_manifest" \
      --num "$cache_size" \
      --cache-algorithm "$algo" \
      --request-count "$tdp_requests" \
      --shuffle-seed "$request_seed" \
      >> "$LOG_FILE" 2>&1
  done
done < "$sorted_mapping_file"

echo "DONE" | tee -a "$LOG_FILE"
