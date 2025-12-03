import os
import ast
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from evaluation_metric import evaluate_text_generation  # assuming this works

# ============================================================
# DIRECTORIES AND PATHS
# ============================================================
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
code_directory = os.path.join(parent_directory, "Code")
data_directory = os.path.join(parent_directory, "Data")

print(f"Code Directory: {code_directory}")
print(f"Data Directory: {data_directory}")

BASE_MODEL = "google/flan-t5-large"
ADAPTER_PATH = os.path.join(code_directory, "story-flan-t5-final")  # ← your trained adapter
VAL_CSV = os.path.join(data_directory, "val.csv")
SAVE_GENERATED = os.path.join(code_directory, "generated_stories.csv")
SAVE_METRICS = os.path.join(code_directory, "generated_stories_with_metrics.csv")

assert os.path.exists(ADAPTER_PATH), f"Adapter not found: {ADAPTER_PATH}"
assert os.path.exists(VAL_CSV), f"Validation file not found: {VAL_CSV}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n>>> Using device: {DEVICE}\n")

# ============================================================
# LOAD TOKENIZER & MODEL (Correct way!)
# ============================================================
print(">>> Loading tokenizer from adapter (includes any added tokens)...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)

print(">>> Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(">>> Merging LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(">>> Model + adapter fully loaded and ready!\n")

# ============================================================
# LOAD VALIDATION DATA
# ============================================================
print(">>> Loading validation data...")
df = pd.read_csv(VAL_CSV)
df = df.dropna(subset=["story", "story_beginning_prompt"]).reset_index(drop=True)
print(f">>> Loaded {len(df)} validation samples.\n")


# ============================================================
# PROMPT BUILDER
# ============================================================
def build_prompt(row):
    words = []
    if pd.notna(row.get("words")):
        try:
            if isinstance(row["words"], str) and row["words"].startswith("["):
                words = ast.literal_eval(row["words"])
        except:
            pass
    keywords = ", ".join(words) if words else "happy, friend, sun"

    prompt = (
        "You are a kind children's story writer.\n"
        f"Use these words: {keywords}\n"
        f"Start with: {row['story_beginning_prompt']}\n"
        "Write a complete short story:"
    )
    return prompt


# ============================================================
# GENERATION FUNCTION
# ============================================================
@torch.inference_mode()
def generate_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.8,
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ============================================================
# RUN INFERENCE
# ============================================================
print(">>> Starting story generation...")
generated_stories = []

for idx, row in df.iterrows():
    prompt = build_prompt(row)
    story = generate_story(prompt)
    generated_stories.append(story)
    print(f"[{idx + 1}/{len(df)}] Generated")

df["generated_story"] = generated_stories
df.to_csv(SAVE_GENERATED, index=False)
print(f"\n>>> All stories saved to: {SAVE_GENERATED}\n")

# ============================================================
# EVALUATION
# ============================================================
print(">>> Running evaluation metrics...")
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

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 70)
print("         FINAL EVALUATION RESULTS")
print("=" * 70)
for k, v in summary.items():
    print(f"{k:30}: {v:.4f}" if isinstance(v, float) else f"{k:30}: {v}")
print("=" * 70)
print("Inference + Evaluation Completed Successfully!")
print("=" * 70)