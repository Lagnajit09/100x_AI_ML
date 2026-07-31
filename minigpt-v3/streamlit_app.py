"""
MiniGPT v3 — Fine-Tuning Pipeline Demo (Streamlit Community Cloud)

Shows ONE prompt run through four checkpoints of the same SmolLM-135M base:
    Base  →  CPT  →  SFT  →  DPO
so a viewer can watch how each fine-tuning stage changes the model.

Deployment target: Streamlit Community Cloud (free, ~1 GB RAM, CPU-only).

Memory strategy (the important part):
    Four 135M models do not co-reside in 1 GB. So only ONE model is held in
    RAM at a time — @st.cache_resource(max_entries=1) keeps exactly one, and
    loading a different stage evicts the previous one. The four stages are
    therefore generated SEQUENTIALLY, not in parallel. Peak memory = 1 model.
"""
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════
# Config  —  set USER to your Hugging Face Hub username
# ═══════════════════════════════════════════════════════════════════════
USER = "m-lagnajit"

STAGES = [
    {"key": "base", "name": "1 · Base (SmolLM-135M)",
     "repo": "HuggingFaceTB/SmolLM-135M",
     "blurb": "Raw pretrained model. No fine-tuning at all."},
    {"key": "cpt", "name": "2 · CPT (WikiText)",
     "repo": f"{USER}/minigpt-v3-cpt",
     "blurb": "Continued pretraining on WikiText — adds domain knowledge."},
    {"key": "sft", "name": "3 · SFT (Alpaca)",
     "repo": f"{USER}/minigpt-v3-sft",
     "blurb": "Supervised fine-tuning on instruction/response pairs — learns to follow instructions."},
    {"key": "dpo", "name": "4 · DPO (Orca preferences)",
     "repo": f"{USER}/minigpt-v3-dpo",
     "blurb": "Preference alignment on chosen/rejected pairs — refines answer quality."},
]

# CPU precision note: float16 on CPU is a trap — several PyTorch CPU ops are
# not implemented for Half and will crash mid-generation. bfloat16 (~270 MB
# per model) fits the 1 GB box with headroom and has solid CPU inference
# support in modern torch. If you ever see garbled output, switch to
# torch.float32 (~540 MB — bulletproof, but tight on memory). Never float16 on CPU.
DTYPE = torch.bfloat16

# Stages 3 & 4 were trained on this exact template. Feeding the SAME string to
# all four checkpoints is the fair comparison — it isolates what each stage added.
TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


# ═══════════════════════════════════════════════════════════════════════
# Loading  —  tokenizer cached once; model cached ONE-AT-A-TIME
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(f"{USER}/minigpt-v3-dpo")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@st.cache_resource(max_entries=1, show_spinner=False)
def get_model(repo):
    # max_entries=1 → only the most-recently-used model stays resident.
    # Loading a different stage evicts the previous one. This is what keeps
    # peak memory at a single model on the 1 GB free tier.
    m = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    m.eval()
    return m


def generate(repo, instruction, temperature, top_p, max_new_tokens):
    tok = get_tokenizer()
    model = get_model(repo)
    enc = tok(TEMPLATE.format(instruction=instruction), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MiniGPT v3 — Pipeline Demo",
                   page_icon="◆", layout="centered")

st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {max-width: 760px; padding-top: 3rem; padding-bottom: 4rem;}
  h1 {font-weight: 700; letter-spacing: -0.02em;}
  .subtitle {color: #6b7280; font-size: 0.95rem; margin-top: -0.4rem; margin-bottom: 1.4rem;}
  .stage-name {font-weight: 600; font-size: 1.02rem; margin-bottom: 0.15rem;}
  .stage-blurb {color: #6b7280; font-size: 0.82rem; margin-bottom: 0.7rem;}
  .stage-resp {font-size: 0.95rem; line-height: 1.55;}
</style>
""", unsafe_allow_html=True)

st.markdown("# MiniGPT v3")
st.markdown(
    '<div class="subtitle">One SmolLM-135M base, four training checkpoints. '
    'Watch the same prompt evolve as each fine-tuning stage is added.</div>',
    unsafe_allow_html=True,
)

# ── Strong, upfront instructions ────────────────────────────────────────
with st.container(border=True):
    st.markdown("**How to read this demo**")
    st.markdown(
        "Your prompt is sent — *unchanged, in the same instruction template* — "
        "through four snapshots of one model, so every difference you see comes "
        "purely from **training**, not from the input.\n\n"
        "1. **Base** — raw pretrained model. Often rambles or ignores the instruction format.\n"
        "2. **CPT** — after continued pretraining. More fluent and knowledgeable, still not instruction-aware.\n"
        "3. **SFT** — after supervised fine-tuning. Now actually *answers the instruction*.\n"
        "4. **DPO** — after preference alignment. Cleaner, more helpful phrasing.\n\n"
        "Enter an instruction and press **Run all four stages**."
    )

# ── Input ────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Instruction",
    value="Explain why the sky is blue.",
    height=90,
    placeholder="e.g. Explain photosynthesis in simple terms.",
)

with st.expander("Generation settings"):
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p       = c2.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_tokens  = c3.slider("Max tokens", 32, 256, 120, 8)

st.caption(
    "⏳ Each run loads and runs four models one at a time on a free CPU — "
    "expect roughly **1–2 minutes** per run. Later runs are faster (models cache to disk)."
)

run = st.button("Run all four stages", type="primary", use_container_width=True)

# ── Run (sequential, one model in memory at a time) ──────────────────────
if run:
    if not prompt.strip():
        st.warning("Please enter an instruction first.")
    else:
        results = {}
        with st.status("Working…", expanded=True) as status:
            for i, stage in enumerate(STAGES, 1):
                status.update(label=f"Stage {i} of 4 — {stage['name']}: loading & generating…")
                results[stage["key"]] = generate(
                    stage["repo"], prompt, temperature, top_p, max_tokens
                )
            status.update(label="Done — all four stages complete.", state="complete")
        st.session_state["results"] = results
        st.session_state["ran_prompt"] = prompt

# ── Results (persisted in session state so reruns don't clear them) ──────
if "results" in st.session_state:
    st.divider()
    st.caption(f'Prompt: "{st.session_state["ran_prompt"]}"')
    for stage in STAGES:
        with st.container(border=True):
            st.markdown(f'<div class="stage-name">{stage["name"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="stage-blurb">{stage["blurb"]}</div>',
                        unsafe_allow_html=True)
            text = st.session_state["results"].get(stage["key"], "")
            st.markdown(f'<div class="stage-resp">{text or "—"}</div>',
                        unsafe_allow_html=True)
