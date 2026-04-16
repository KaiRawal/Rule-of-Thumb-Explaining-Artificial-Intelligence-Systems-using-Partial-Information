#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def parse_float(value: str) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.9f}".rstrip("0").rstrip(".")


def build_judicial_row(timing_path: Path):
    with timing_path.open(newline="", encoding="utf-8") as file_obj:
        timing_rows = list(csv.DictReader(file_obj))

    by_id = {}
    for entry in timing_rows:
        key = (entry.get("ID") or "").strip()
        numeric = parse_float(entry.get("time", ""))
        if math.isnan(numeric):
            continue
        by_id.setdefault(key, []).append(numeric)

    shap_times = by_id.get("shap", [])
    ig_times = by_id.get("ig", [])
    rot_combined = by_id.get("rot_combined", [])
    rot_train = by_id.get("rot_train___128_1e-05_0.5", [])

    if not shap_times or not ig_times or not rot_combined:
        return None

    datapoints = len(shap_times)
    total_shap = sum(shap_times)
    total_ig = sum(ig_times)
    total_rot = rot_combined[0]
    init_rot = rot_train[0] if rot_train else total_rot
    init_shap = total_shap / datapoints
    init_ig = total_ig / len(ig_times)

    return {
        "INFO": "JudicialCaseOutcomePrediction",
        "datapoints": str(datapoints),
        "init_rot_s": fmt(init_rot),
        "init_rot_l": fmt(init_rot),
        "init_shap": fmt(init_shap),
        "init_lime": fmt(init_ig),
        "total_rot_s": fmt(total_rot),
        "total_rot_l": fmt(total_rot),
        "total_shap": fmt(total_shap),
        "total_lime": fmt(total_ig),
        "ss": fmt(total_shap / total_rot) if total_rot else "",
        "sl": fmt(total_ig / total_rot) if total_rot else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Augment RunTimes.csv with optional judicial summary row.")
    parser.add_argument("--include-judicial-row", default="0", choices=["0", "1"])
    parser.add_argument("--run-times", default="RunTimes.csv")
    parser.add_argument("--timing", default="TIMING.csv")
    args = parser.parse_args()

    run_times_path = Path(args.run_times)
    timing_path = Path(args.timing)
    include_judicial = args.include_judicial_row == "1"

    if not run_times_path.exists():
        raise SystemExit("RunTimes.csv is required")

    with run_times_path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_cols = [
        "INFO",
        "datapoints",
        "init_rot_s",
        "init_rot_l",
        "init_shap",
        "init_lime",
        "total_rot_s",
        "total_rot_l",
        "total_shap",
        "total_lime",
        "ss",
        "sl",
    ]
    for column_name in required_cols:
        if column_name not in fieldnames:
            raise SystemExit(f"RunTimes.csv missing required column: {column_name}")

    changed = False
    if include_judicial and timing_path.exists():
        judicial_row = build_judicial_row(timing_path)
        if judicial_row is not None:
            replaced = False
            for index, row in enumerate(rows):
                if row.get("INFO", "").strip() == "JudicialCaseOutcomePrediction":
                    rows[index] = judicial_row
                    replaced = True
                    changed = True
                    break
            if not replaced:
                rows.append(judicial_row)
                changed = True

    if changed:
        with run_times_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
