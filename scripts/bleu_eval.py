import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# ================= CONFIG =================
INPUT_FILE = "evaluation/logs/phi_outputs.jsonl"

smoothie = SmoothingFunction().method4

bleu1_scores = []
bleu2_scores = []
bleu4_scores = []

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        reference_text = data["reference"].lower()
        generated_text = data["generated"].lower()

        reference_tokens = word_tokenize(reference_text)
        generated_tokens = word_tokenize(generated_text)

        if len(generated_tokens) == 0:
            continue

        bleu1 = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie
        )

        bleu2 = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothie
        )

        bleu4 = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)

        count += 1

print("\n BLEU Evaluation Results")
print("--------------------------")
print(f"Total evaluated samples: {count}")

if count == 0:
    print(" No valid samples found. Check the log file format.")
else:
    print(f"Average BLEU-1: {sum(bleu1_scores)/len(bleu1_scores):.4f}")
    print(f"Average BLEU-2: {sum(bleu2_scores)/len(bleu2_scores):.4f}")
    print(f"Average BLEU-4: {sum(bleu4_scores)/len(bleu4_scores):.4f}")

