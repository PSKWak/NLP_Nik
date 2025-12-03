# App.py → FINAL VERSION – works with .safetensors + typo-proof
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
import torch

MODEL_FOLDER = Path(__file__).parent / "story-flan-t5-final"

# Debug (remove later if you want)
st.sidebar.write("Looking for model in:", str(MODEL_FOLDER.resolve()))
st.sidebar.write("Folder exists:", MODEL_FOLDER.exists())
if MODEL_FOLDER.exists():
    files = [p.name for p in MODEL_FOLDER.iterdir()]
    st.sidebar.write("Files inside:", files)

# Critical checks
if not MODEL_FOLDER.exists():
    st.error("Folder 'story-flan-t5-final' not found in repo root!")
    st.stop()

if not (MODEL_FOLDER / "adapter_config.json").exists():
    st.error("adapter_config.json missing!")
    st.stop()

# Check for correct weight file name
if not (MODEL_FOLDER / "adapter_model.safetensors").exists() and not (MODEL_FOLDER / "adapter_model.bin").exists():
    st.error("""
    Adapter weights not found!
    Expected one of:
    - adapter_model.safetensors
    - adapter_model.bin

    You probably have a typo: adapter_model.safetensor (missing 's')
    """)
    st.stop()

st.success("All model files found — loading your AI storyteller...")


@st.cache_resource(show_spinner="Loading google/flan-t5-large + your LoRA adapter...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(MODEL_FOLDER))
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


def generate_story(start: str, keywords: str = "", temp: float = 0.9):
    kws = ", ".join([k.strip() for k in keywords.split(",") if k.strip()]) or "magic, friendship"
    prompt = f"""You are a kind children's story writer.
Use these words: {kws}
Start with: {start.strip()}
Write a complete short story:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=temp,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Write a complete short story:")[-1].strip()


# UI
st.title("Magical Children's Story Generator")
st.markdown("Your fine-tuned model is now loading perfectly!")

c1, c2 = st.columns([3, 1])
with c1:
    beginning = st.text_area("Start the story:", "Once upon a time, in a magical forest,", height=120)
with c2:
    keywords = st.text_input("Keywords:", "dragon, rainbow, friendship, courage")

temp = st.slider("Creativity", 0.5, 1.5, 0.9, 0.05)

if st.button("Generate Magical Story", type="primary", use_container_width=True):
    with st.spinner("Writing your story..."):
        story = generate_story(beginning, keywords, temp)
    st.success("Story ready!")
    st.markdown("### Your Story")
    st.write(story)

st.caption("google/flan-t5-large + story-flan-t5-final (LoRA) • Running perfectly!")
