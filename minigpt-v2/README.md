# MiniGPT v2 — TinyStories

A GPT-style transformer built entirely from scratch in PyTorch, trained on
the TinyStories dataset.

## Architecture

- **RMSNorm** in place of LayerNorm
- **RoPE** (rotary positional embeddings) in place of sinusoidal positional
  encoding
- **SwiGLU** feed-forward network in place of ReLU
- **Grouped Query Attention (GQA)** — 4 query heads sharing 2 KV groups
- From-scratch **byte-pair encoding** tokenizer (vocab size 2000), trained
  on the same TinyStories corpus

## Model config

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 8 |
| n_heads | 4 |
| n_kv_heads | 2 |
| d_ff | 683 |
| block_size | 64 |
| vocab_size | 2000 |
| Parameters | ~6.8M |

## Training curve

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 7.77 | 7.78 |
| 2000 | 3.54 | 3.50 |
| 6000 | 2.86 | 2.92 |
| 9999 | 2.74 | 2.84 |

Trained with linear warmup (10% of steps) + cosine decay learning rate
scheduling, peak LR 3e-4.

## Known limitation: token fusion

Generated text occasionally fuses word-fragment tokens into non-words
(e.g. "harmeagrow", "sucturkey"), most visibly at higher sampling
temperatures. At low temperature (~0.4–0.7) the model produces clean,
coherent text by leaning on the simple, repetitive patterns it saw in
TinyStories — to the point where output can closely echo memorized
training boilerplate. The fusion failure mode was diagnosed across multiple
training runs — varying model capacity (1.3M → 6.8M parameters) and
training steps (5,000 → 10,000) — as a **token-boundary learning
problem**, not primarily a model-capacity limitation: increasing both
capacity and steps improved overall coherence and loss but did not
resolve the fusion failure mode, which stayed roughly constant in
frequency across all three checkpoints.

Inspecting the actual BPE merge table (including the tail, rank
1600-1999) showed the vocabulary itself contains sensible words,
phrases, and sub-word fragments — not corpus noise — ruling out
"vocab size 2000 produces junk merges" as the cause.

### Investigated and ruled out: whitespace-crossing merges

One hypothesis: the original from-scratch BPE trainer allowed merges
to cross whitespace boundaries (e.g. `"was so happy"` became a single
token), so a genuinely different mechanism — word-fragment tokens
without positional disambiguation — might be the real driver. A
whitespace-respecting variant was implemented and tested (merges
constrained to never cross a space, verified with zero boundary
violations on the resulting vocabulary), holding vocab size fixed at
2000 for a clean comparison.

Result: **worse**, not better. Loss dropped further (1.57 vs. 2.84),
but generated text degraded into heavy word-level repetition
(`"three three very and very"`, `"loved loved loved"`) rather than
sub-word fusion. Root cause: whitespace-respecting merges stop tokens
from *crossing* boundaries, but don't disambiguate *which side* of a
boundary a token sits on — e.g. the standalone word "he" and the
word-initial fragment "he" (as in "he-llo") can still collapse to the
same token ID, forcing one embedding to represent multiple distinct
roles. GPT-2's actual tokenizer avoids this with an explicit
space-prefix marker (its `Ġ` convention) that was not implemented
here. This checkpoint intentionally reverts to the original
(whitespace-crossing) tokenizer, which produces a milder and better
overall trade-off given the remaining time budget.

### Not yet tried

Space-prefix boundary marking (GPT-2-style), a larger and more
diverse training corpus, or substantially greater model scale.

## Context window

The model was trained with a 64-token context (`block_size=64`), and
RoPE position tables are precomputed only up to that length. Generation
therefore stops cleanly once the context window is full; requesting more
tokens than that produces no additional output. This is a deliberate cap,
not a failure — the model has no learned behavior beyond position 64.

## Usage

Type a prompt and hit Generate. TinyStories-style openers work best:

- `Once upon a time`
- `There was a little`
- `One day, a`

## Previous version

[MiniGPT v1](https://huggingface.co/spaces/m-lagnajit/minigpt-shakespeare)
— character-level tokenizer, trained on Tiny Shakespeare, vanilla
transformer architecture (LayerNorm, sinusoidal positional encoding,
ReLU FFN, standard multi-head attention).
