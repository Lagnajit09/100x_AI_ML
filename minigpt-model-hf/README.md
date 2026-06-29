---
title: MiniGPT Tiny Shakespeare
emoji: 📜
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# MiniGPT — Tiny Shakespeare

A character-level GPT language model built **entirely from scratch** in PyTorch.
No pretrained weights. No Hugging Face `transformers`. Just raw architecture.

## What's inside

| Component | Detail |
|---|---|
| Tokenizer | Character-level (65 unique chars) |
| Embedding | `nn.Embedding(65, 128)` |
| Architecture | 4 transformer blocks |
| Attention | Multi-head (4 heads, d_k=32) with KV cache |
| Masking | Causal mask (upper triangular) |
| FFN | Linear(128→512)→ReLU→Linear(512→128) |
| Normalization | Pre-LN LayerNorm + residual connections |
| Parameters | ~400K |
| Training data | Tiny Shakespeare (~1.1M characters) |
| Training steps | 3,000 |
| Final val loss | ~1.81 |

## Architecture built from scratch

Every component implemented without shortcuts:

- **Tokenizer** — character-level encode/decode dictionaries
- **Positional encoding** — sinusoidal (sin/cos waves)
- **Multi-head attention** — Q, K, V projections + scaled dot-product
- **Causal masking** — upper triangular `-inf` mask preventing future peeking
- **KV cache** — stores past K and V per block for efficient generation
- **Nucleus sampling** — top-p + temperature + repetition penalty

## Generation

Uses a two-phase generation loop:

1. **Prefill** — full prompt processed in one forward pass, KV cache built
2. **Decode** — one new character per step, reusing cached K and V

## Training curve

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.32 | 4.11 |
| 1000 | 2.04 | 2.04 |
| 2000 | 1.82 | 1.90 |
| 2999 | 1.77 | 1.81 |

## Usage

Type any Shakespeare-style prompt and hit Generate:

- `ROMEO:` → speech continuation
- `To be or` → phrase completion  
- `HAMLET:` → monologue start
- `The king` → narrative continuation
