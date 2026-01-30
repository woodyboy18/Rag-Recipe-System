from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# ================= CONFIG =================
MODEL_NAME = "microsoft/phi-2"
DATA_PATH = "data/phi_rag_train.jsonl"
OUTPUT_DIR = "phi-rag-lora"

# ================= LOAD MODEL =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)


# ================= LoRA CONFIG =================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ================= LOAD DATA =================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def format_prompt(example):
    text = f"""### Instruction:
{example['instruction']}

### Context:
{example['input']}

### Answer:
{example['output']}"""
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# ================= TRAINING =================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Phi LoRA fine-tuning completed")
