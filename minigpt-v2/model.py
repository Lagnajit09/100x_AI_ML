"""
MiniGPT v2 — architecture definition.

Same four upgrades over v1's vanilla transformer:
  - RMSNorm   (replaces LayerNorm)
  - RoPE      (replaces sinusoidal positional encoding)
  - SwiGLU    (replaces ReLU feed-forward)
  - GQA       (grouped query attention — shared K,V across query head groups)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(d_k, max_seq_len, base=10000, device='cpu'):
    pair_indices = torch.arange(0, d_k, 2, device=device).float()
    thetas = 1.0 / (base ** (pair_indices / d_k))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, thetas)
    return angles.cos(), angles.sin()


def apply_rope(x, cos_table, sin_table, start_pos=0):
    B, num_heads, T, d_k = x.shape
    cos = cos_table[start_pos: start_pos + T]
    sin = sin_table[start_pos: start_pos + T]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    return x_rotated.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, n_kv_heads, dropout, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % n_kv_heads == 0

        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = num_heads // n_kv_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        cos_table, sin_table = precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer('cos_table', cos_table)
        self.register_buffer('sin_table', sin_table)

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        Q = apply_rope(Q, self.cos_table, self.sin_table, start_pos)
        K = apply_rope(K, self.cos_table, self.sin_table, start_pos)

        if kv_cache is not None:
            K = torch.cat([kv_cache['K'], K], dim=2)
            V = torch.cat([kv_cache['V'], V], dim=2)

        new_kv_cache = {'K': K, 'V': V}
        full_len = K.shape[2]

        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if T > 1:
            mask = torch.triu(
                torch.ones(T, full_len, device=x.device),
                diagonal=full_len - T + 1
            ).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out), new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))
        value = self.W2(x)
        x = gate * value
        x = self.W3(x)
        return self.dropout(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.gamma * x_norm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, n_kv_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, n_kv_heads, dropout, max_seq_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, kv_cache=None, start_pos=0):
        attn_out, new_kv_cache = self.attention(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x, new_kv_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, n_kv_heads, d_ff,
                 num_layers, dropout, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, n_kv_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, kv_caches=None, start_pos=0):
        B, T = token_ids.shape

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        x = self.embedding(token_ids)
        x = self.emb_dropout(x)

        new_kv_caches = []
        for block, block_cache in zip(self.blocks, kv_caches):
            x, new_block_cache = block(x, kv_cache=block_cache, start_pos=start_pos)
            new_kv_caches.append(new_block_cache)

        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits, new_kv_caches
