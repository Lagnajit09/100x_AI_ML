import torch
import gradio as gr

from model import MiniGPT
from bpe_tokenizer import bpe_encode, bpe_decode

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('minigpt_v2.pt', map_location=DEVICE, weights_only=False)

merges = checkpoint['merges']
vocab_size = checkpoint['vocab_size']

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=checkpoint['d_model'],
    num_heads=checkpoint['n_heads'],
    n_kv_heads=checkpoint['n_kv_heads'],
    d_ff=checkpoint['d_ff'],
    num_layers=checkpoint['n_layers'],
    dropout=checkpoint['dropout'],
    max_seq_len=checkpoint['block_size'],
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

BLOCK_SIZE = checkpoint['block_size']


def encode(s):
    return bpe_encode(s, merges)


def decode(ids):
    return bpe_decode(ids, merges)


def _sample(logits_1d, temperature, top_p=0.9, generated=None, rep_penalty=1.2):
    if generated and rep_penalty > 1.0:
        for token_id in set(generated[-15:]):
            logits_1d[token_id] /= rep_penalty

    if temperature == 0:
        return logits_1d.argmax().item()

    logits_1d = logits_1d / temperature
    probs = torch.softmax(logits_1d, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[sampled_idx].item()


@torch.no_grad()
def generate(prompt, max_new_tokens, temperature, top_p, rep_penalty):
    if not prompt.strip():
        return "Please enter a prompt."

    token_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)

    # Guard against prompts longer than the model's context window
    if token_ids.shape[1] > BLOCK_SIZE:
        token_ids = token_ids[:, -BLOCK_SIZE:]

    generated = token_ids[0].tolist()

    logits, kv_caches = model(token_ids, kv_caches=None, start_pos=0)
    next_id = _sample(logits[0, -1], temperature, top_p=top_p,
                       generated=generated, rep_penalty=rep_penalty)
    generated.append(next_id)

    for _ in range(int(max_new_tokens) - 1):
        current_pos = len(generated) - 1
        input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)
        logits, kv_caches = model(input_tensor, kv_caches=kv_caches, start_pos=current_pos)
        next_id = _sample(logits[0, -1], temperature, top_p=top_p,
                           generated=generated, rep_penalty=rep_penalty)
        generated.append(next_id)

    return decode(generated)


with gr.Blocks(title="MiniGPT v2 — TinyStories") as demo:
    gr.Markdown(
        """
        # MiniGPT v2 — TinyStories
        A GPT-style transformer built from scratch in PyTorch: RMSNorm, RoPE,
        SwiGLU, and grouped query attention, with a from-scratch byte-pair
        encoding tokenizer trained on the TinyStories dataset.

        **Known limitation:** the tokenizer occasionally fuses word-fragment
        tokens into non-words in longer generations (e.g. "harmeagrow") —
        a documented capacity/vocabulary limitation of this small a model,
        not a bug.
        """
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Once upon a time",
                value="Once upon a time",
            )
            max_tokens = gr.Slider(20, 300, value=150, step=10, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            rep_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Generated text", lines=12)

    btn.click(
        fn=generate,
        inputs=[prompt, max_tokens, temperature, top_p, rep_penalty],
        outputs=output,
    )

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
