# ============================================================
# app_with_metrics.py - Children's Story Generator + Live Metrics
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

nltk.download('punkt', quiet=True)

# -------------------------- CONFIG --------------------------
st.set_page_config(page_title="Story Generator + Live Evaluation", page_icon="✨", layout="wide")

BASE_MODEL = "google/flan-t5-large"
ADAPTER_PATH = "story-flan-t5-final"

# -------------------------- CACHED MODEL LOADING --------------------------
@st.cache_resource(show_spinner="Loading your fine-tuned story model...")
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

@st.cache_resource(show_spinner="Loading evaluation models (GPT-2 for perplexity)...")
def load_ppl_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), tokenizer

story_model, story_tokenizer = load_story_model()
ppl_model, ppl_tokenizer = load_ppl_model()

# -------------------------- HELPERS --------------------------
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip())

def compute_perplexity(text, model=ppl_model, tokenizer=ppl_tokenizer):
    if not text.strip():
        return float("nan")
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return float(torch.exp(loss).item())

def generate_story(prompt, temperature=0.9):
    inputs = story_tokenizer(prompt, return_tensors="pt").to(story_model.device)
    with torch.no_grad():
        outputs = story_model.generate(
            **inputs,
            max_new_tokens=450,
            temperature=temperature,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
    full = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the story part
    story = full.split("Write a complete short story:")[-1].strip()
    return story

# -------------------------- UI --------------------------
st.title("✨ Magical Story Generator + Live Evaluation")
st.markdown("### Your fine-tuned Flan-T5-LoRA model with real-time quality metrics")

col1, col2 = st.columns([2, 1])
with col1:
    starting_sentence = st.text_area(
        "Story beginning:",
        value="Once upon a time, in a sunny forest,",
        height=100
    )
with col2:
    keywords_input = st.text_input(
        "Keywords (comma-separated):",
        value="dragon, rainbow, friendship, magic, courage"
    )

temperature = st.slider("Creativity", 0.3, 1.5, 0.9, 0.05)
max_length = st.slider("Max story length (tokens)", 200, 600, 450, 50)

if st.button("🪄 Generate & Evaluate Story", type="primary", use_container_width=True):
    if not starting_sentence.strip():
        st.error("Please enter a starting sentence!")
    else:
        with st.spinner("Generating magical story..."):
            keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
            keywords_str = ", ".join(keywords) if keywords else "happy, kind, magical"

            prompt = (
                "You are a kind children's story writer.\n"
                f"Use these words: {keywords_str}\n"
                f"Start with: {starting_sentence.strip()}\n"
                "Write a complete short story:"
            )

            generated_story = generate_story(prompt, temperature)

        st.success("Story Generated!")
        st.markdown(f"### {generated_story}")

        # -------------------------- EVALUATION --------------------------
        with st.spinner("Computing quality metrics..."):
            ref = starting_sentence.strip()
            hyp = generated_story

            # 1. Keyword Coverage
            hyp_lower = hyp.lower()
            used = [kw for kw in keywords if kw in hyp_lower]
            strict_match = len(used) == len(keywords)
            coverage_pct = len(used) / len(keywords) * 100 if keywords else 100

            # 2. BLEU (sentence level with smoothing)
            smoothie = SmoothingFunction().method1
            bleu = sentence_bleu(
                [ref.split()], hyp.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )

            # 3. ROUGE
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ref, hyp)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rougel = rouge_scores['rougeL'].fmeasure

            # 4. BERTScore
            P, R, F1 = bert_score([hyp], [ref], lang="en", verbose=False)
            bert_f1 = F1.mean().item()

            # 5. Perplexity
            ppl = compute_perplexity(hyp)

        # -------------------------- DISPLAY METRICS --------------------------
        st.markdown("### 📊 Live Evaluation Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Keyword Coverage", f"{coverage_pct:.1f}%",
                     f"{len(used)}/{len(keywords)} used" if keywords else "No keywords")
            st.write("**Strict Match**" if strict_match else "Partial")

        with col2:
            st.metric("BLEU Score", f"{bleu:.3f}")

        with col3:
            st.metric("ROUGE-L", f"{rougel:.3f}")

        with col4:
            st.metric("BERTScore F1", f"{bert_f1:.3f}")

        with col5:
            st.metric("Perplexity", f"{ppl:.1f}", help="Lower = more fluent")

        # Color-coded summary
        st.markdown("#### Quality Summary")
        quality = "Excellent" if (bert_f1 > 0.85 and ppl < 30 and coverage_pct == 100) else \
                  "Good" if (bert_f1 > 0.80 and coverage_pct >= 80) else "Fair"
        color = "green" if quality == "Excellent" else "orange" if quality == "Good" else "red"

        st.markdown(f"<h2 style='color:{color};'>Overall Quality: {quality} ✨</h2>", unsafe_allow_html=True)

        if keywords:
            st.caption(f"Keywords used: {', '.join(used)}")

st.markdown("---")
st.caption("Fine-tuned on children's stories • LoRA on Flan-T5-Large • Live metrics powered by BERTScore, ROUGE, GPT-2")