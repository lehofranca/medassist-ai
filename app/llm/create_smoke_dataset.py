import json
from pathlib import Path

TRAIN_SOURCE = Path("data/training/train_formatted.json")
TRAIN_OUTPUT = Path("data/training/train_smoke.json")

with open(TRAIN_SOURCE, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

subset = data[:20]

with open(TRAIN_OUTPUT, "w", encoding="utf-8") as file:
    for item in subset:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Smoke training dataset created")
print("Samples:", len(subset))
print("Output:", TRAIN_OUTPUT)
