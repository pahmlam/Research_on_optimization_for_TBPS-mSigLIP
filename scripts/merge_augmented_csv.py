"""
Merge augmented captions from CSV into data_captions_vn3k.json.

Usage:
    python scripts/merge_augmented_csv.py --csv path/to/augmented.csv [--json path/to/data_captions_vn3k.json] [--train-only]

By default:
  - JSON path: VN3K/data_captions_vn3k.json
  - Only train split entries are merged (--train-only flag, on by default)
  - A backup is saved as <json_path>.backup.json
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


AUG_COLUMNS = [
    "Vietnamese_captions_synonyms",
    "Vietnamese_captions_deleted",
    "Vietnamese_captions_inserted",
    "Vietnamese_captions_swapped",
    "back translation",
]


def main():
    parser = argparse.ArgumentParser(description="Merge augmented CSV captions into JSON")
    parser.add_argument("--csv", required=True, help="Path to augmented CSV file")
    parser.add_argument(
        "--json",
        default="VN3K/data_captions_vn3k.json",
        help="Path to target JSON file (default: VN3K/data_captions_vn3k.json)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        default=True,
        help="Only merge train split entries (default: True)",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Merge all splits (overrides --train-only)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup of the JSON file",
    )
    args = parser.parse_args()

    train_only = args.train_only and not args.all_splits

    # Load CSV
    with open(args.csv, encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    print(f"CSV rows loaded: {len(csv_rows)}")

    # Load JSON
    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON entries loaded: {len(data)}")

    # Build lookup: file_path -> list of augmented captions
    aug_by_fp = {}
    skipped = 0
    for row in csv_rows:
        if train_only and row["split"] != "train":
            skipped += 1
            continue
        fp = row["file_path"]
        captions = []
        for col in AUG_COLUMNS:
            val = row.get(col, "").strip()
            if val:
                captions.append(val.lower())  # match JSON lowercase convention
        aug_by_fp[fp] = captions

    if train_only:
        print(f"Skipped {skipped} non-train rows")
    print(f"Entries to merge: {len(aug_by_fp)}")

    # Merge into JSON
    added_count = 0
    modified_count = 0
    for entry in data:
        fp = entry["file_path"]
        if fp not in aug_by_fp:
            continue
        new_captions = aug_by_fp[fp]
        existing = set(c.lower().strip() for c in entry["captions"])
        entry_added = 0
        for cap in new_captions:
            if cap.strip() not in existing:
                entry["captions"].append(cap)
                entry["processed_tokens"].append([])  # placeholder (unused in VN3K pipeline)
                added_count += 1
                entry_added += 1
                existing.add(cap.strip())
        if entry_added > 0:
            modified_count += 1

    # Backup
    if not args.no_backup:
        backup_path = args.json + ".backup.json"
        shutil.copy(args.json, backup_path)
        print(f"Backup saved: {backup_path}")

    # Save
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"New captions added: {added_count}")
    print(f"Entries modified: {modified_count}")
    print(f"Done. Saved to {args.json}")


if __name__ == "__main__":
    main()
