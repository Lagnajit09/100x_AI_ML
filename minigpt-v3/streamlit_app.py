"""
MiniGPT v3 — Fine-Tuning Pipeline Demo (Streamlit Community Cloud)

Free-form comparison across four checkpoints of one SmolLM-135M base:
    Base  →  CPT  →  SFT  →  DPO
plus a dedicated RLVR (GRPO) arithmetic panel, since that stage was trained
on a few-shot verifiable-reward task and only behaves in-distribution there.

Deployment target: Streamlit Community Cloud (free, ~1 GB RAM, CPU-only).

Memory strategy: only ONE model resides in RAM at a time
(@st.cache_resource(max_entries=1)); loading a new stage evicts the previous.
"""
import re
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════
# Config
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

# The RLVR/GRPO stage lives in its own panel — different repo, different prompt.
GRPO_REPO = f"{USER}/minigpt-v3-grpo"

DTYPE = torch.bfloat16   # float16 on CPU is a trap; bfloat16 ~270 MB fits 1 GB. Never float16 on CPU.

# Free-form template for stages 1–4 (SFT/DPO were trained on this).
TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# GRPO was trained with THESE few-shot exemplars in front of every prompt.
# Feeding them is what makes the arithmetic stage behave in-distribution.
FEWSHOT = (
    "### Instruction:\nWhat is 1 + 1?\n\n### Response:\n<answer>2</answer>\n\n"
    "### Instruction:\nWhat is 3 + 4?\n\n### Response:\n<answer>7</answer>\n\n"
    "### Instruction:\nWhat is 5 + 2?\n\n### Response:\n<answer>7</answer>\n\n"
)
GRPO_TEMPLATE = FEWSHOT + "### Instruction:\n{question}\n\n### Response:\n"

# Same content-requiring, lenient extractor used by the reward + final grader.
ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</\s*answer\s*>", re.DOTALL | re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(f"{USER}/minigpt-v3-dpo")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@st.cache_resource(max_entries=1, show_spinner=False)
def get_model(repo):
    m = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    m.eval()
    return m


def _generate(repo, prompt_text, temperature, top_p, max_new_tokens, rep_pen=1.15, ngram=4):
    tok = get_tokenizer()
    model = get_model(repo)
    enc = tok(prompt_text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_stage(repo, instruction, temperature, top_p, max_new_tokens):
    return _generate(repo, TEMPLATE.format(instruction=instruction),
                     temperature, top_p, max_new_tokens)


def generate_grpo(a, b, temperature, top_p):
    """Few-shot arithmetic prompt → raw completion + extracted answer + correctness."""
    prompt = GRPO_TEMPLATE.format(question=f"What is {a} + {b}?")
    # rep_pen=1.0 / no ngram ban: the trained output is very short (<answer>N</answer>);
    # penalties that help long free-form text only corrupt this tiny structured output.
    raw = _generate(GRPO_REPO, prompt, temperature, top_p, max_new_tokens=32,
                    rep_pen=1.0, ngram=0)
    m = ANSWER_RE.search(raw)
    pred = re.sub(r"[^0-9-]", "", m.group(1)) if m else ""
    gold = str(a + b)
    return raw, pred, (pred == gold), gold


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
  .col-label {color: #6b7280; font-size: 0.72rem; text-transform: uppercase;
              letter-spacing: 0.04em; margin-bottom: 0.2rem;}
  .raw-box {font-family: ui-monospace, monospace; font-size: 0.82rem;
            background: #f7f7f8; border-radius: 6px; padding: 0.5rem 0.6rem;
            white-space: pre-wrap; word-break: break-word; color: #374151;}
  .pill {display: inline-block; background: #eef2ff; color: #4338ca; font-size: 0.72rem;
         font-weight: 600; padding: 0.12rem 0.5rem; border-radius: 999px; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("# MiniGPT v3")
st.markdown(
    '<div class="subtitle">One SmolLM-135M base, taken through the full '
    'post-training stack: CPT → SFT → DPO → RLVR. Each stage adds one kind of learning.</div>',
    unsafe_allow_html=True,
)

# ── Always-visible orientation: the whole story in a few lines ──────────
with st.container(border=True):
    st.markdown("**What you're looking at**")
    st.markdown(
        "This demo runs a single tiny language model (135M parameters) at **five points "
        "along its training journey**, so you can watch what each stage of fine-tuning adds:\n\n"
        "**Base → CPT → SFT → DPO** are *generalists* — ask them anything, they answer in free text. "
        "They live in the **left tab**.\n\n"
        "**RLVR (the 5th stage)** is a *specialist* — it was trained on one narrow, machine-checkable "
        "task (single-digit addition). It lives in the **right tab**, with its own input.\n\n"
        "*Why the split?* You can't fairly compare a specialist and a generalist in the same box — "
        "the details are in each tab. Open the ℹ️ sections to go as deep as you like."
    )

tab_pipeline, tab_rlvr = st.tabs(["① Fine-tuning pipeline (Base → DPO)", "② RLVR (arithmetic)"])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — the original four-stage free-form comparison
# ─────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    with st.container(border=True):
        st.markdown("**How to read this tab**")
        st.markdown(
            "Your prompt is sent — *unchanged, in the same instruction template* — through "
            "four snapshots of one model, so every difference you see comes purely from "
            "**training**, not from the input.\n\n"
            "1. **Base** — raw pretrained model. Often rambles or ignores the instruction format.\n"
            "2. **CPT** — continued pretraining. More fluent and knowledgeable, still not instruction-aware.\n"
            "3. **SFT** — supervised fine-tuning. Now actually *answers the instruction*.\n"
            "4. **DPO** — preference alignment. Cleaner, more helpful phrasing."
        )

    with st.expander("ℹ️ What do CPT, SFT, and DPO actually mean?"):
        st.markdown(
            "- **CPT (Continued Pre-Training)** — keep training the base model on more raw text "
            "(here, Wikipedia). It teaches *knowledge and fluency*, but not how to follow instructions.\n"
            "- **SFT (Supervised Fine-Tuning)** — train on `(instruction → ideal response)` pairs so the "
            "model learns the *behaviour* of answering a request rather than just continuing text.\n"
            "- **DPO (Direct Preference Optimization)** — show the model pairs of `(better, worse)` answers "
            "and push it toward the *preferred* style. This is 'alignment' — same facts, nicer delivery.\n\n"
            "All three here were trained as small **LoRA adapters** — tiny add-on weights (a few MB) that "
            "modify the frozen 135M base, so each stage is a cheap, stackable delta."
        )

    prompt = st.text_area(
        "Instruction",
        value="Explain why the sky is blue.",
        height=90,
        placeholder="e.g. Explain photosynthesis in simple terms.",
    )

    with st.expander("Generation settings"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("Temperature", 0.1, 1.5, 0.4, 0.1)
        top_p       = c2.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        max_tokens  = c3.slider("Max tokens", 32, 256, 96, 8)

    st.caption(
        "⏳ Each run loads and runs four models one at a time on a free CPU — roughly "
        "**1–2 minutes** per run. Only one model sits in memory at a time (the 1 GB free tier "
        "can't hold four), so they run sequentially. Later runs are faster (models cache to disk)."
    )

    run = st.button("Run all four stages", type="primary", use_container_width=True)

    if run:
        if not prompt.strip():
            st.warning("Please enter an instruction first.")
        else:
            results = {}
            with st.status("Working…", expanded=True) as status:
                for i, stage in enumerate(STAGES, 1):
                    status.update(label=f"Stage {i} of 4 — {stage['name']}: loading & generating…")
                    results[stage["key"]] = generate_stage(
                        stage["repo"], prompt, temperature, top_p, max_tokens
                    )
                status.update(label="Done — all four stages complete.", state="complete")
            st.session_state["results"] = results
            st.session_state["ran_prompt"] = prompt

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

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — RLVR / GRPO arithmetic stage (raw vs extracted)
# ─────────────────────────────────────────────────────────────────────────
with tab_rlvr:
    st.markdown('<span class="pill">STAGE 5 · RLVR (GRPO)</span>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Reinforcement Learning from Verifiable Rewards**")
        st.markdown(
            "The final stage taught the model to solve single-digit sums and wrap the answer in "
            "`<answer>…</answer>` tags — using **no labelled answers and no human feedback**, only "
            "two small Python functions that *check* each output. Type two numbers and watch it work "
            "on sums it was **never explicitly trained on**.\n\n"
            "👉 Every question in this tab is prompted differently from the left tab. **The five "
            "ℹ️ sections below explain exactly why** — two modes, single digits, arithmetic only, "
            "hidden examples, and the raw-vs-extracted view."
        )

    with st.expander("ℹ️ 1 · What is RLVR, and what is GRPO?"):
        st.markdown(
            "**RLVR = Reinforcement Learning from Verifiable Rewards.** In normal RLHF, a human (or a "
            "trained 'reward model') judges the output. RLVR replaces that judge with a plain **program** "
            "that checks correctness. Arithmetic is ideal: *'is the answer 8?'* is a one-line check — no "
            "human, no reward model, no labels needed.\n\n"
            "**GRPO = Group Relative Policy Optimization** — the training algorithm. For each question it "
            "samples several answers, scores each with the verifier, and then nudges the model toward the "
            "answers that scored **above the group's own average**. There's no separate 'value network' — "
            "the group average *is* the baseline. That's the 'Group Relative' part.\n\n"
            "Two reward functions ran together here, mirroring DeepSeek-R1:\n"
            "- **format reward** — is the answer wrapped in well-formed `<answer>…</answer>` tags?\n"
            "- **accuracy reward** — is the wrapped number actually correct?"
        )

    with st.expander("ℹ️ 2 · Why is this stage in a separate tab (the 'two modes')?"):
        st.markdown(
            "Stages 1–4 are **generalists**: they take any instruction and reply in free-form text. "
            "This stage is a **specialist** — it was trained on exactly one narrow, verifiable task.\n\n"
            "If we dropped it into the same box and asked it *'Explain the sky'*, it would emit "
            "tag-wrapped nonsense and look **worse** than DPO — which would badly misrepresent what it "
            "actually learned. So it gets its own input (two numbers) and its own prompt format, and it's "
            "tested in exactly the distribution it was trained on.\n\n"
            "The rule at work: **evaluate a model in the same distribution you trained it in.** Comparing a "
            "specialist against generalists in one shared box would be an unfair, misleading comparison."
        )

    with st.expander("ℹ️ 3 · Why only single-digit numbers (0–9)?"):
        st.markdown(
            "Because the base model is only **135M parameters** — very small — and RL has a hard limit:\n\n"
            "> **RL can only amplify abilities the model already has a faint grasp of. It cannot install "
            "brand-new skills.**\n\n"
            "A 135M model can *occasionally* land a single-digit sum by luck. That flicker of success is "
            "the signal GRPO sharpens into consistency. But it essentially **never** gets something like "
            "`47 × 83` right, so there's no flicker to amplify — training on that would produce nothing.\n\n"
            "We deliberately capped inputs at 0–9 because that's the **edge of what's reachable** for this "
            "model. Push past it and accuracy collapses — which is itself the honest lesson of RLVR: it "
            "*sharpens latent skill, it doesn't create new capability*."
        )

    with st.expander("ℹ️ 4 · Why only arithmetic — why can't I type a free-form question here?"):
        st.markdown(
            "Because RLVR needs a **verifiable** reward — a program that can automatically decide whether "
            "an answer is right. Addition qualifies perfectly: the checker is literally `a + b == answer`.\n\n"
            "Open-ended prompts like *'write a poem'* or *'explain gravity'* have **no automatic correctness "
            "check** — there's nothing for the verifier to score, so GRPO has no signal to learn from. "
            "That's why the classic RLVR domains are **math, code, and structured formats**: they're all "
            "machine-checkable. This demo picks the simplest checkable task there is."
        )

    with st.expander("ℹ️ 5 · Why are there hidden examples in the prompt (few-shot)?"):
        st.markdown(
            "A 135M model can't reliably *follow an instruction about format* like 'put your answer in tags' "
            "— that kind of zero-shot instruction-following only emerges in much larger models. But a small "
            "model **can imitate examples**.\n\n"
            "So behind the scenes, every question you ask is prefixed with three solved examples:\n\n"
            "```\n"
            "### Instruction:\nWhat is 1 + 1?\n\n### Response:\n<answer>2</answer>\n\n"
            "### Instruction:\nWhat is 3 + 4?\n\n### Response:\n<answer>7</answer>\n\n"
            "### Instruction:\nWhat is 5 + 2?\n\n### Response:\n<answer>7</answer>\n"
            "```\n\n"
            "You only type two numbers; these exemplars are added automatically. This is **few-shot "
            "prompting**, and it's the *exact* conditioning the model saw during training. Feeding it "
            "anything else would break it — which, during development, is precisely the bug that once made "
            "it score 0/5. Same-distribution prompting is mandatory for this stage."
        )

    with st.expander("ℹ️ 6 · 'Raw' vs 'Extracted' — what am I comparing?"):
        st.markdown(
            "- **Raw** is the model's *exact* output, cosmetic noise and all. You'll often see a stray "
            "`</ Answer >` with odd spacing/capitalisation, or a trailing `<br />`. The model picked up "
            "these harmless quirks because the reward only checked for a *well-formed tag somewhere* and "
            "stopped caring what came after.\n"
            "- **Extracted** is the number a verifier pulls out from between the tags — the only thing "
            "RLVR ever actually grades.\n\n"
            "Showing both lets you see the messy reality *and* the clean signal side by side. The **✓ / ✗** "
            "is the verifier's verdict: does the extracted number equal `a + b`? Held-out numbers (ones "
            "not in the examples) still come out right — proof the model learned the *pattern*, not just "
            "memorised the three examples."
        )

    with st.expander("ℹ️ 7 · Honest note — did it really 'learn maths'?"):
        st.markdown(
            "Not from scratch. The base model already had a faint, unreliable ability to add single digits; "
            "GRPO **amplified that flicker into consistency** and taught it to present the answer in a "
            "checkable format. That's the whole point of RLVR — it makes existing latent ability *reliable*, "
            "it doesn't conjure new ability out of nothing. On a task the base truly couldn't do at all "
            "(large multiplication, reasoning puzzles), the same recipe would produce nothing — and knowing "
            "*which* tasks are reachable for a given model size is the real skill this stage demonstrates."
        )

    cc1, cc2 = st.columns(2)
    a = cc1.number_input("First number",  min_value=0, max_value=9, value=4, step=1)
    b = cc2.number_input("Second number", min_value=0, max_value=9, value=5, step=1)
    st.caption("Capped at 0–9 on purpose — that's the range this 135M model was trained on and can "
               "reliably solve (see ℹ️ 3).")

    with st.expander("Generation settings"):
        g1, g2 = st.columns(2)
        g_temp = g1.slider("Temperature", 0.1, 1.5, 0.7, 0.1, key="grpo_temp")
        g_top  = g2.slider("Top-p", 0.1, 1.0, 0.9, 0.05, key="grpo_top")

    st.caption("⏳ Loads the RLVR model (evicts any pipeline model in memory) — first call ~1 minute.")

    run_g = st.button("Run RLVR stage", type="primary", use_container_width=True)

    if run_g:
        with st.status(f"Computing {a} + {b}…", expanded=False) as status:
            raw, pred, correct, gold = generate_grpo(a, b, g_temp, g_top)
            status.update(label="Done.", state="complete")
        st.session_state["grpo"] = dict(raw=raw, pred=pred, correct=correct, gold=gold, a=a, b=b)

    if "grpo" in st.session_state:
        g = st.session_state["grpo"]
        st.divider()
        st.caption(f'Prompt: "What is {g["a"]} + {g["b"]}?"  ·  expected: {g["gold"]}')
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('<div class="col-label">Raw output (what the model emitted)</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="raw-box">{g["raw"] or "—"}</div>', unsafe_allow_html=True)
        with rc2:
            st.markdown('<div class="col-label">Extracted answer (what the verifier reads)</div>',
                        unsafe_allow_html=True)
            if g["pred"] == "":
                st.markdown('<div class="raw-box">— (no well-formed &lt;answer&gt; tag found)</div>',
                            unsafe_allow_html=True)
            elif g["correct"]:
                st.success(f'{g["pred"]}  ✓ correct')
            else:
                st.error(f'{g["pred"]}  ✗ (expected {g["gold"]})')
