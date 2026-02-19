import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", help="Path to input JSONL file")
parser.add_argument("--start", type=int, help="Starting ID value")
parser.add_argument("--output", help="Path to output JSONL file")
args = parser.parse_args()

with open(args.jsonl) as f:
    lines = f.readlines()

with open(args.output, "w") as f:
    for i, line in enumerate(lines):
        row = json.loads(line)
        row = {"id": args.start + i, **row}
        f.write(json.dumps(row) + "\n")
