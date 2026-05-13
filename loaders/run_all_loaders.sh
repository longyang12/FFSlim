#!/usr/bin/env bash
set -euo pipefail

# Config defaults (override via flags)
DATASETS="mscoco"
NUM=0
MEMORY=0
ROOT="/mnt/datasets/mscoco_converted"
PYTHON="/home/llm/experiment/dataset_Analysis_Modeling/.env/bin/python"
ONLY=""
CONTINUE_ON_ERROR=0
DRY_RUN=0
DROP_CACHES=1
SWAPOFF=1
RESTORE_SWAP=1
MONITOR=1
MONITOR_INTERVAL=1
MONITOR_DIR="/home/llm/experiment/dataset_Analysis_Modeling/monitor"

usage() {
  cat <<'EOF'
Usage: run_all_loaders.sh [options]

Options:
  --datasets NAME1,NAME2   Comma-separated dataset names (default: mscoco)
  --num N                  LRU cache size (0 disables cache)
  --memory N               Memory size tag to log
  --root PATH              Converted dataset root
  --python PATH            Python executable to run loaders
  --only K1,K2             Loader keys to run (e.g. webdataset_B,tfrecord_F)
  --continue-on-error      Continue even if one loader fails
  --drop-caches 0|1        Drop page cache before each run (default: 1)
  --swapoff 0|1            Disable swap before each run (default: 1)
  --restore-swap 0|1       Re-enable swap after each run (default: 1)
  --monitor 0|1            Monitor RSS/CPU/IO during run (default: 1)
  --monitor-interval SEC   Sampling interval in seconds (default: 1)
  --monitor-dir PATH       Directory for monitor logs
  --dry-run                Print commands without running
  -h, --help               Show this help

Loader keys:
  tdp_B, tdp_F, webdataset_B, webdataset_F, tfrecord_B, tfrecord_F,
  ffrecord_B, ffrecord_F
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --num) NUM="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --drop-caches) DROP_CACHES="$2"; shift 2 ;;
    --swapoff) SWAPOFF="$2"; shift 2 ;;
    --restore-swap) RESTORE_SWAP="$2"; shift 2 ;;
    --monitor) MONITOR="$2"; shift 2 ;;
    --monitor-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
    --monitor-dir) MONITOR_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac

done

# Device readahead control (disable on start, restore on exit) for /dev/nvme0n1
# This runs after option parsing so `--dry-run` can be honored.
RA_DEVICE="/dev/nvme0n1"
RA_OLD=""
RA_CHANGED=0
if command -v blockdev >/dev/null 2>&1 && [[ -b "$RA_DEVICE" ]]; then
  RA_OLD=$(blockdev --getra "$RA_DEVICE" 2>/dev/null || echo "")
  if [[ -n "$RA_OLD" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "DRY RUN: would set readahead 0 on $RA_DEVICE (current: $RA_OLD)"
    else
      echo "Setting readahead=0 on $RA_DEVICE (was $RA_OLD)"
      sudo blockdev --setra 0 "$RA_DEVICE"
      RA_CHANGED=1
      restore_readahead() {
        if [[ "$RA_CHANGED" -eq 1 ]]; then
          echo "Restoring readahead=$RA_OLD on $RA_DEVICE"
          sudo blockdev --setra "$RA_OLD" "$RA_DEVICE" || echo "Warning: failed to restore readahead"
        fi
      }
      trap restore_readahead EXIT
    fi
  fi
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOADER_KEYS=(
  "tdp_F"
  "webdataset_F"
  "tfrecord_F"
  "ffrecord_F"
  "tdp_B"
  "webdataset_B"
  "tfrecord_B"
  "ffrecord_B"
)

loader_path() {
  local key="$1"
  case "$key" in
    webdataset_B) echo "$BASE_DIR/random_loader_webdataset_B.py" ;;
    webdataset_F) echo "$BASE_DIR/random_loader_webdataset_F.py" ;;
    tfrecord_B) echo "$BASE_DIR/random_loader_tfrecord_B.py" ;;
    tfrecord_F) echo "$BASE_DIR/random_loader_tfrecord_F.py" ;;
    ffrecord_B) echo "$BASE_DIR/random_loader_ffrecord_B.py" ;;
    ffrecord_F) echo "$BASE_DIR/random_loader_ffrecord_F.py" ;;
    tdp_B) echo "$BASE_DIR/random_loader_tdp_B.py" ;;
    tdp_F) echo "$BASE_DIR/random_loader_tdp_F.py" ;;
    *) echo "" ;;
  esac
}

if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a ONLY_KEYS <<< "$ONLY"
else
  ONLY_KEYS=("${LOADER_KEYS[@]}")
fi

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

run_cmd() {
  local cmd=("$@")
  echo "${cmd[*]}"
  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi
  if [[ $DROP_CACHES -eq 1 ]]; then
    sudo bash -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  fi
  if [[ $SWAPOFF -eq 1 ]]; then
    sudo swapoff -a
  fi

  if [[ $MONITOR -eq 1 ]]; then
    mkdir -p "$MONITOR_DIR"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local log_path="$MONITOR_DIR/monitor_${ts}.csv"
    local interval
    if [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] && [[ "$MONITOR_INTERVAL" -gt 0 ]]; then
      interval="$MONITOR_INTERVAL"
    else
      interval=1
    fi
    local cpu_count
    cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
    if [[ ! "$cpu_count" =~ ^[0-9]+$ ]] || [[ "$cpu_count" -le 0 ]]; then
      cpu_count=1
    fi
    local pid
    "${cmd[@]}" &
    pid=$!
    echo "ts_ms,rss_kb,anon_kb,file_kb,cpu_pct,read_bps,write_bps,pgfault,pgmajfault" > "$log_path"

    local last_proc_jiffies=0
    local last_total_jiffies=0
    local last_read=0
    local last_write=0
    local last_minflt=0
    local last_majflt=0
    local first=1

    while kill -0 "$pid" 2>/dev/null; do
      local now_ms
      now_ms=$(date +%s%3N)

      local rss_kb anon_kb file_kb
      rss_kb=$(awk '/VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      anon_kb=$(awk '/RssAnon:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      file_kb=$(awk '/RssFile:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      if [[ ! "$anon_kb" =~ ^[0-9]+$ ]]; then
        anon_kb=0
      fi
      if [[ ! "$file_kb" =~ ^[0-9]+$ ]]; then
        file_kb=0
      fi

      local stat
      stat=$(cat "/proc/$pid/stat" 2>/dev/null || echo "")
      local minflt majflt utime stime proc_jiffies
      if [[ -n "$stat" ]]; then
        minflt=$(echo "$stat" | awk '{print $10}')
        majflt=$(echo "$stat" | awk '{print $12}')
        utime=$(echo "$stat" | awk '{print $14}')
        stime=$(echo "$stat" | awk '{print $15}')
        proc_jiffies=$((utime + stime))
      else
        minflt=0
        majflt=0
        proc_jiffies=0
      fi

      local total_jiffies
      total_jiffies=$(awk '/^cpu / {print $2+$3+$4+$5+$6+$7+$8+$9+$10}' /proc/stat)

      local read_bytes write_bytes
      read_bytes=$(awk '/read_bytes/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)
      write_bytes=$(awk '/write_bytes/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)
      if [[ ! "$read_bytes" =~ ^[0-9]+$ ]]; then
        read_bytes=0
      fi
      if [[ ! "$write_bytes" =~ ^[0-9]+$ ]]; then
        write_bytes=0
      fi

      local cpu_pct=0
      local read_bps=0
      local write_bps=0
      local pgfault=0
      local pgmajfault=0

      if [[ $first -eq 0 ]]; then
        local dproc=$((proc_jiffies - last_proc_jiffies))
        local dtotal=$((total_jiffies - last_total_jiffies))
        if [[ $dtotal -gt 0 ]]; then
          cpu_pct=$(awk -v a="$dproc" -v b="$dtotal" -v c="$cpu_count" 'BEGIN {printf "%.2f", (a/b)*100*c}')
        fi
        local dread=$((read_bytes - last_read))
        local dwrite=$((write_bytes - last_write))
        read_bps=$((dread / interval))
        write_bps=$((dwrite / interval))
        pgfault=$((minflt - last_minflt))
        pgmajfault=$((majflt - last_majflt))
      fi

      echo "$now_ms,$rss_kb,$anon_kb,$file_kb,$cpu_pct,$read_bps,$write_bps,$pgfault,$pgmajfault" >> "$log_path"

      last_proc_jiffies=$proc_jiffies
      last_total_jiffies=$total_jiffies
      last_read=$read_bytes
      last_write=$write_bytes
      last_minflt=$minflt
      last_majflt=$majflt
      first=0

      sleep "$interval"
    done

    wait "$pid"
    exit_code=$?
  else
    "${cmd[@]}"
    exit_code=$?
  fi

  if [[ $RESTORE_SWAP -eq 1 ]]; then
    sudo swapon -a
  fi

  if [[ $exit_code -ne 0 ]]; then
    if [[ $CONTINUE_ON_ERROR -eq 1 ]]; then
      echo "Command failed: ${cmd[*]}" >&2
      return 0
    fi
    return $exit_code
  fi
}

for dataset in "${DATASET_LIST[@]}"; do
  for key in "${ONLY_KEYS[@]}"; do
    script_path="$(loader_path "$key")"
    if [[ -z "$script_path" ]]; then
      echo "Unknown loader key: $key" >&2
      exit 1
    fi
    run_cmd "$PYTHON" "$script_path" "$dataset" "$NUM" "$MEMORY" --root "$ROOT"
  done
 done
