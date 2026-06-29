import torch
import gradio as gr
from model import MiniGPT

# ── Device ───────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load checkpoint ───────────────────────────────────────────────────────────
checkpoint = torch.load('minigpt_shakespeare.pt', map_location=device)

stoi       = checkpoint['stoi']
itos       = checkpoint['itos']
vocab_size = checkpoint['vocab_size']
d_model    = checkpoint['d_model']
n_heads    = checkpoint['n_heads']
d_ff       = checkpoint['d_ff']
n_layers   = checkpoint['n_layers']
dropout    = checkpoint['dropout']

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Load model ────────────────────────────────────────────────────────────────
model = MiniGPT(vocab_size, d_model, n_heads, d_ff, n_layers, dropout).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded on {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Sampling helpers ──────────────────────────────────────────────────────────
def _sample(logits_1d, temperature, top_p, generated, rep_penalty):
    if temperature == 0:
        return logits_1d.argmax().item()

    if generated and rep_penalty > 1.0:
        for token_id in set(generated[-15:]):
            logits_1d[token_id] /= rep_penalty

    logits_1d = logits_1d / temperature
    probs = torch.softmax(logits_1d, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[sampled_idx].item()


def generate(prompt, max_new_tokens, temperature, top_p, rep_penalty):
    # Validate prompt
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a prompt to get started."

    # Filter out chars not in vocabulary
    filtered = ''.join(c for c in prompt if c in stoi)
    if not filtered:
        return f"None of the characters in your prompt exist in the Shakespeare vocabulary. Try something like 'ROMEO:' or 'To be'"

    token_ids = torch.tensor([encode(filtered)], dtype=torch.long).to(device)
    generated = token_ids[0].tolist()

    with torch.no_grad():
        # Prefill
        logits, kv_caches = model(token_ids, kv_caches=None, start_pos=0)
        next_id = _sample(
            logits[0, -1].clone(), temperature, top_p, generated, rep_penalty
        )
        generated.append(next_id)

        # Decode
        for _ in range(max_new_tokens - 1):
            current_pos  = len(generated) - 1
            input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(device)
            logits, kv_caches = model(
                input_tensor, kv_caches=kv_caches, start_pos=current_pos
            )
            next_id = _sample(
                logits[0, -1].clone(), temperature, top_p, generated, rep_penalty
            )
            generated.append(next_id)

    return decode(generated)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 📜 MiniGPT — Tiny Shakespeare

A **character-level GPT built from scratch** in PyTorch — no pretrained weights, no libraries, just the raw Transformer architecture.

**Architecture:** 4 transformer blocks · 4 attention heads · d_model=128 · ~400K parameters  
**Training:** 3,000 steps on [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (~1.1M characters)  
**Built by:** Learning the full stack — tokenization → embeddings → multi-head attention → KV cache → generation

> Try prompts like `ROMEO:`, `To be or`, `HAMLET:`, `All:`, `The king`
"""

EXAMPLES = [
    ["ROMEO:", 80, 0.6, 0.9, 1.2],
    ["To be or", 80, 0.6, 0.9, 1.2],
    ["HAMLET:", 80, 0.7, 0.9, 1.2],
    ["All:", 60, 0.6, 0.85, 1.2],
    ["The king is", 80, 0.6, 0.9, 1.2],
]

with gr.Blocks(title="MiniGPT — Tiny Shakespeare") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="ROMEO:",
                lines=2,
            )
            max_tokens = gr.Slider(
                minimum=20, maximum=200, value=80, step=10,
                label="Max new tokens",
                info="How many characters to generate"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.6, step=0.05,
                label="Temperature",
                info="Lower = more conservative, Higher = more creative"
            )
            top_p = gr.Slider(
                minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                label="Top-p (nucleus sampling)",
                info="Probability mass to sample from — lower = tighter"
            )
            rep_penalty = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                label="Repetition penalty",
                info="Suppresses recently used characters"
            )
            generate_btn = gr.Button("Generate ✨", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Generated text",
                lines=12,
                show_copy_button=True,
            )

    generate_btn.click(
        fn=generate,
        inputs=[prompt_box, max_tokens, temperature, top_p, rep_penalty],
        outputs=output_box,
    )

    prompt_box.submit(
        fn=generate,
        inputs=[prompt_box, max_tokens, temperature, top_p, rep_penalty],
        outputs=output_box,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_box, max_tokens, temperature, top_p, rep_penalty],
        outputs=output_box,
        fn=generate,
        cache_examples=False,
        label="Try these prompts"
    )

    gr.Markdown("""
    ---
    **How it works:** Each character is encoded as an integer → embedded into a 128-dim vector →
    passed through 4 transformer blocks (multi-head attention + FFN) → decoded back to a character distribution.
    KV cache means only the new token is computed at each generation step — not the full sequence.

    **Source code:** Available on [GitHub](https://github.com/Lagnajit09/100x_AI_ML)
    """)

demo.launch()
