import os
import ast
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Your evaluation module
from evaluation_metric import evaluate_text_generation


# ============================================================
# CONFIG — CHANGE PATHS IF NEEDED
# ============================================================
BASE_MODEL = "google/flan-t5-large"

ADAPTER_PATH = "/home/ubuntu/Final--Project-Group5/Code/story-flan-t5-final"  # your final saved LoRA folder
DATA_PATH = "/home/ubuntu/Final--Project-Group5/Data/val.csv"                 # val/test data
SAVE_GENERATED = "/home/ubuntu/Final--Project-Group5/Code/generated_stories.csv"
SAVE_METRICS = "/home/ubuntu/Final--Project-Group5/Code/generated_stories_with_metrics.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n>>> Using device: {DEVICE}\n")


# ============================================================
# LOAD BASE MODEL + TOKENIZER + LORA ADAPTER
# ============================================================
print(">>> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(">>> Loading base Flan-T5 model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

print(">>> Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\n>>> Model + LoRA adapter loaded successfully.\n")


# ============================================================
# LOAD DATA
# ============================================================
print(">>> Reading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["story", "story_beginning_prompt"]).reset_index(drop=True)
print(f">>> Loaded {len(df)} rows.\n")


# ============================================================
# PROMPT BUILDER (matches your training logic)
# ============================================================
def build_prompt(row):
    """Builds the exact prompt format used during training."""
    # Parse keyword list
    words = []
    if isinstance(row["words"], str) and row["words"].startswith("["):
        try:
            words = ast.literal_eval(row["words"])
        except:
            pass

    keywords = ", ".join(words) if words else "happy"

    prompt = (
        "You are a kind children's story writer.\n"
        f"Use these words: {keywords}\n"
        f"Start with: {row['story_beginning_prompt']}\n"
        "Write a complete short story:"
    )
    return prompt


# ============================================================
# INFERENCE FUNCTION
# ============================================================
def generate_text(prompt):
    """Generate story text using LoRA-finetuned Flan-T5."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ============================================================
# RUN GENERATION
# ============================================================
print(">>> Generating stories...")
df["generated_story"] = df["story_beginning_prompt"].apply(
    lambda _: ""
)  # initialize column

generated = []

for idx, row in df.iterrows():
    prompt = build_prompt(row)
    gen = generate_text(prompt)
    generated.append(gen)

df["generated_story"] = generated

df.to_csv(SAVE_GENERATED, index=False)
print(f"\n>>> Generated stories saved to:\n{SAVE_GENERATED}\n")


# ============================================================
# RUN FULL EVALUATION
# ============================================================
print(">>> Running evaluation...")

eval_df, summary = evaluate_text_generation(
    df.copy(),
    ref_col="story",
    hyp_col="generated_story",
    bleu_max_n=4,
    ppl_model_name="gpt2",
    ppl_max_length=512,
    ppl_batch_size=4,
    bert_model_type="bert-base-uncased",
    bert_lang="en",
)

eval_df.to_csv(SAVE_METRICS, index=False)

print(f"\n>>> Metrics saved to:\n{SAVE_METRICS}\n")


# ============================================================
# PRETTY PRINT METRICS
# ============================================================
print("=" * 70)
print("📊 FINAL EVALUATION SUMMARY")
print("=" * 70)

def pretty(k):
    return k.replace("_", " ").title()

for metric, value in summary.items():
    if isinstance(value, float):
        print(f"{pretty(metric):30}: {value:.4f}")
    else:
        print(f"{pretty(metric):30}: {value}")

print("=" * 70)
print("✓ Inference + Evaluation Complete")
print("=" * 70 + "\n")
