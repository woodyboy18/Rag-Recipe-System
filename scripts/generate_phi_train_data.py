import pandas as pd
import json
import os

# ================= CONFIG =================
CSV_PATH = "data/recipes.csv"   # change if your CSV is elsewhere
OUT_PATH = "data/phi_rag_train.jsonl"
MAX_SAMPLES = 1000   # start small, we can increase later

# ================= LOAD CSV =================
df = pd.read_csv(CSV_PATH)

# Try to find ingredient column automatically
ING_COL = None
for col in df.columns:
    if "ingredient" in col.lower():
        ING_COL = col
        break

if ING_COL is None:
    raise ValueError("❌ No ingredient column found in CSV")

print(f"✅ Using ingredient column: {ING_COL}")

# ================= GENERATE =================
count = 0
with open(OUT_PATH, "a", encoding="utf-8") as f:
    for _, row in df.iterrows():
        if count >= MAX_SAMPLES:
            break

        title = str(row.get("Name", "Recipe")).strip()
        ingredients = str(row[ING_COL]).strip()

        if not ingredients or ingredients.lower() == "nan":
            continue

        record = {
            "instruction": f"Extract ingredients for {title} using only the given context.",
            "input": f"Title: {title}\nIngredients: {ingredients}",
            "output": "\n".join(
                f"- {i.strip()}"
                for i in ingredients.replace("c(", "")
                                      .replace(")", "")
                                      .replace('"', "")
                                      .split(",")
                if i.strip()
            )
        }

        f.write(json.dumps(record) + "\n")
        count += 1

print(f"🎉 Generated {count} training samples")
