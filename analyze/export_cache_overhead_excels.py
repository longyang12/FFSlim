import os
import re
from collections import defaultdict

import pandas as pd


LOG_PATH = "/home/llm/experiment/dataset_Analysis_Modeling/temp.log"
OUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/cache_overhead_excels"
OUT_WORKBOOK = "format_dataset_cache_overhead.xlsx"


DATASET_LINE_RE = re.compile(
    r"^dataset:(?P<dataset>[^,]+),num:(?P<num>\d+),cache_algorithm:(?P<algo>[^,]+)(?:,index_structure:(?P<index>[^,]+))?,memory_size:(?P<memory>\d+)"
)
HIT_RATE_RE = re.compile(r"cache hit rate:(?P<hit>[0-9.]+)")
KVS_RE = re.compile(r"([a-zA-Z_]+)=([^,\s]+)")


def parse_dataset_and_format(raw_dataset: str) -> tuple[str, str]:
    if raw_dataset.endswith("_FFSlim"):
        return raw_dataset[: -len("_FFSlim")], "FFSlim"
    return raw_dataset, "TDP"


def parse_log(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows: list[dict] = []
    current = None

    i = 0
    while i < len(lines):
        line = lines[i]

        m_dataset = DATASET_LINE_RE.match(line)
        if m_dataset:
            ds_raw = m_dataset.group("dataset")
            ds_name, ds_format = parse_dataset_and_format(ds_raw)
            current = {
                "dataset": ds_name,
                "format": ds_format,
                "num": int(m_dataset.group("num")),
                "cache_algorithm": m_dataset.group("algo"),
                "memory_size": int(m_dataset.group("memory")),
            }
            i += 1
            continue

        if current is not None and line.startswith("cache hits:"):
            m_hit = HIT_RATE_RE.search(line)
            if m_hit:
                current["cache hit rate"] = float(m_hit.group("hit"))
            i += 1
            continue

        if current is not None and line.startswith("cache mgmt overhead:"):
            combined = line
            # Some runs break this record into two lines.
            if i + 1 < len(lines) and "avg_get=" in lines[i + 1]:
                combined = combined + lines[i + 1]
                i += 1

            kvs = dict(KVS_RE.findall(combined))
            current["get_calls"] = int(kvs.get("get_calls", "0"))
            current["put_calls"] = int(kvs.get("put_calls", "0"))
            current["get_total"] = float(kvs.get("get_total", "0").replace("s", ""))
            current["put_total"] = float(kvs.get("put_total", "0").replace("s", ""))
            current["avg_get"] = float(kvs.get("avg_get", "0").replace("us", ""))
            current["avg_put"] = float(kvs.get("avg_put", "0").replace("us", ""))
            current["cache_total"] = float(kvs.get("cache_total", "0").replace("s", ""))

            rows.append(current)
            current = None

            i += 1
            continue

        i += 1

    return rows


def export_excels(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row['format']}_{row['dataset']}"
        grouped[key].append(row)

    columns = [
        "format",
        "cache_algorithm",
        "cache hit rate",
        "get_calls",
        "put_calls",
        "get_total",
        "put_total",
        "avg_get",
        "avg_put",
        "cache_total",
    ]
    column_rename = {
        "format": "format",
        "cache_algorithm": "cache_algorithm",
        "cache hit rate": "cache hit rate (%)",
        "get_calls": "get_calls (count)",
        "put_calls": "put_calls (count)",
        "get_total": "get_total (s)",
        "put_total": "put_total (s)",
        "avg_get": "avg_get (us)",
        "avg_put": "avg_put (us)",
        "cache_total": "cache_total (s)",
    }

    workbook_path = os.path.join(out_dir, OUT_WORKBOOK)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for format_dataset, items in grouped.items():
            df = pd.DataFrame(items)
            df = df[columns]
            df = df.rename(columns=column_rename)
            # Keep rows in original log order; do not sort.
            safe_sheet = format_dataset[:31]
            df.to_excel(writer, index=False, sheet_name=safe_sheet)

    print(f"saved: {workbook_path}")


def main() -> None:
    rows = parse_log(LOG_PATH)
    if not rows:
        raise RuntimeError("No valid cache overhead records were parsed from temp.log")
    export_excels(rows, OUT_DIR)
    print(f"datasets: {len(set(r['dataset'] for r in rows))}, records: {len(rows)}")


if __name__ == "__main__":
    main()
