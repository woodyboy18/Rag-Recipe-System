import json
from bert_score import score

print("Starting...")

LOG_PATH = "evaluation/logs/qwen_outputs.jsonl"

references = []
candidates = []

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        if data["reference"].strip() and data["generated"].strip():
            references.append(data["reference"])
            candidates.append(data["generated"])

print("References:", len(references))
print("Candidates:", len(candidates))
print("Computing BERTScore...")

P, R, F1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)

print("Done!")

print(f"Average Precision : {P.mean().item():.4f}")
print(f"Average Recall    : {R.mean().item():.4f}")
print(f"Average F1 Score  : {F1.mean().item():.4f}")
