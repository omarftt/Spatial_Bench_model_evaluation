import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--original", required=True, help="Path to the original JSONL file")
parser.add_argument("--corrected", required=True, help="Path to the corrected JSONL file")
parser.add_argument("--output", required=True, help="Path to the output JSONL file")
parser.add_argument("--join_right", action="store_true", help="Also add rows from corrected that don't exist in original")
args = parser.parse_args()

with open(args.corrected) as f:
    corrections = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

with open(args.original) as f:
    original = [json.loads(line) for line in f if line.strip()]

original_ids = {row["id"] for row in original}

with open(args.output, "w") as f:
    for row in original:
        row_id = row["id"]
        f.write(json.dumps(corrections.get(row_id, row)) + "\n")
    if args.join_right:
        for row_id, row in corrections.items():
            if row_id not in original_ids:
                f.write(json.dumps(row) + "\n")

replaced = sum(1 for row in original if row["id"] in corrections)
added = sum(1 for row_id in corrections if row_id not in original_ids) if args.join_right else 0
total = len(original) + added
print(f"Replaced {replaced} sample(s), added {added} new sample(s). Total rows: {total}. Output: {args.output}")
