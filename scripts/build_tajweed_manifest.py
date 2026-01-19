import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src.qrec.data.manifests import write_jsonl


def _collect_from_folder(root: Path) -> List[Dict]:
    items = []
    for rule_dir in root.iterdir():
        if not rule_dir.is_dir():
            continue
        rule = rule_dir.name
        for label_dir in ("correct", "incorrect"):
            seg_dir = rule_dir / label_dir
            if not seg_dir.exists():
                continue
            label = 1 if label_dir == "correct" else 0
            for wav in seg_dir.rglob("*.wav"):
                items.append({
                    "segment_path": str(wav),
                    "label": label,
                    "rule": rule,
                    "source": "folder",
                })
    return items


def _collect_from_csv(csv_path: Path, base_dir: Path | None) -> List[Dict]:
    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("file_path"):
                continue
            rel_path = Path(row["file_path"])
            if base_dir and not rel_path.is_absolute():
                seg_path = base_dir / rel_path
            else:
                seg_path = rel_path
            item = {
                "segment_path": str(seg_path),
                "label": int(row.get("label", 0)),
                "rule": row.get("rule"),
                "duration": float(row["duration"]) if row.get("duration") else None,
                "error_type": row.get("error_type"),
                "start_time": float(row["start_time"]) if row.get("start_time") else None,
                "end_time": float(row["end_time"]) if row.get("end_time") else None,
                "surah": row.get("surah"),
                "ayah": row.get("ayah"),
                "reciter": row.get("reciter"),
                "source": "csv",
            }
            items.append(item)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tajweed jsonl manifest.")
    ap.add_argument("--data_dir", type=str, help="Folder with rule/correct|incorrect structure")
    ap.add_argument("--annotations", type=str, help="CSV annotations with file_path, rule, label")
    ap.add_argument("--base_dir", type=str, help="Base dir for relative CSV paths")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    items: List[Dict] = []
    if args.data_dir:
        items.extend(_collect_from_folder(Path(args.data_dir)))
    if args.annotations:
        base_dir = Path(args.base_dir) if args.base_dir else None
        items.extend(_collect_from_csv(Path(args.annotations), base_dir))

    if not items:
        raise SystemExit("No items collected. Provide --data_dir or --annotations.")

    write_jsonl(args.output, items)
    print(f"Wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
