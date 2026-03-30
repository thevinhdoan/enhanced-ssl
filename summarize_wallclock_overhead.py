#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean


TIMESTAMP_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO\] (?P<msg>.*)$")
EPOCH_RE = re.compile(r"^Epoch (?P<epoch>\d+)/(?P<total>\d+)$")
EPOCH_TIME_RE = re.compile(r"^Time taken for epoch (?P<epoch>\d+): (?P<seconds>[0-9.]+)$")
TOTAL_TIME_RE = re.compile(r"^Total time taken: (?P<seconds>[0-9.]+)$")
REFRESH_RE = re.compile(r"^Refreshed robust grouping: ")

METHOD_LABELS = {
    "pet-ensembled": "pet",
    "pet-ensembled-across-nets": "v-pet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize wallclock overhead metrics from saved_models/(v5) and saved_models/(v6) log.txt files."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["saved_models/(v5)", "saved_models/(v6)"],
        help="Experiment root directories to scan. Defaults to saved_models/(v5) and saved_models/(v6).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the summarized metrics as CSV.",
    )
    return parser.parse_args()


def parse_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")


def summarize_log(log_path):
    epoch_times = []
    total_time = None
    refresh_times = []
    current_epoch_ts = None
    current_epoch_has_refresh = False
    expected_epochs = None

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = TIMESTAMP_RE.match(line)
            if not match:
                continue

            timestamp = parse_timestamp(match.group("ts"))
            message = match.group("msg")

            epoch_match = EPOCH_RE.match(message)
            if epoch_match:
                current_epoch_ts = timestamp
                current_epoch_has_refresh = False
                expected_epochs = int(epoch_match.group("total"))
                continue

            epoch_time_match = EPOCH_TIME_RE.match(message)
            if epoch_time_match:
                epoch_times.append(float(epoch_time_match.group("seconds")))
                continue

            total_time_match = TOTAL_TIME_RE.match(message)
            if total_time_match:
                total_time = float(total_time_match.group("seconds"))
                continue

            if REFRESH_RE.match(message) and current_epoch_ts is not None and not current_epoch_has_refresh:
                refresh_times.append((timestamp - current_epoch_ts).total_seconds())
                current_epoch_has_refresh = True

    if not epoch_times:
        raise ValueError(f"No epoch timing lines found in {log_path}")
    if total_time is None:
        raise ValueError(f"No total timing line found in {log_path}")

    if expected_epochs is not None and len(epoch_times) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} epoch timing lines in {log_path}, found {len(epoch_times)}"
        )

    version = log_path.parents[6].name
    peft_method = log_path.parents[5].name
    pet_variant = log_path.parents[4].name

    return {
        "version": version,
        "peft": peft_method,
        "setting": METHOD_LABELS.get(pet_variant, pet_variant),
        "avg_epoch_time_s": mean(epoch_times),
        "total_time_s": total_time,
        "avg_refresh_time_s": (mean(refresh_times) if refresh_times else None),
    }


def format_cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows):
    headers = ["version", "peft", "setting", "avg_epoch_time_s", "total_time_s", "avg_refresh_time_s"]
    widths = {}
    for header in headers:
        widths[header] = max(len(header), *(len(format_cell(row[header])) for row in rows))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(format_cell(row[header]).ljust(widths[header]) for header in headers))


def write_csv(rows, output_csv):
    fieldnames = ["version", "peft", "setting", "avg_epoch_time_s", "total_time_s", "avg_refresh_time_s"]
    output_path = Path(output_csv)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    log_paths = []
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")
        log_paths.extend(sorted(root_path.rglob("log.txt")))

    if not log_paths:
        raise FileNotFoundError("No log.txt files found under the requested roots.")

    rows = [summarize_log(log_path) for log_path in log_paths]
    rows.sort(key=lambda row: (row["version"], row["peft"], row["setting"]))

    print_table(rows)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nSaved CSV to {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
