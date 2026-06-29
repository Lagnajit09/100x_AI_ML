import torch
import torch.nn as nn
import math


def positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (i / d_model)
            pe[pos, i] = math.sin(pos / denom)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(pos / denom)
    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)
        self.W_K     = nn.Linear(d_model, d_model, bias=False)
        self.W_V     = nn.Linear(d_model, d_model, bias=False)
        self.W_O     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T,  self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T,  self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T,  self.num_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache['K'], K], dim=2)
            V = torch.cat([kv_cache['V'], V], dim=2)

        new_kv_cache = {'K': K, 'V': V}
        full_len = K.shape[2]

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
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention    = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attention(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x, new_kv_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm  = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, kv_caches=None, start_pos=0):
        B, T = token_ids.shape
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        x = self.embedding(token_ids)
        x = self.emb_dropout(x)

        pe      = positional_encoding(start_pos + T, self.embedding.embedding_dim, token_ids.device)
        pe_slice = pe[start_pos : start_pos + T]
        x       = x + pe_slice

        new_kv_caches = []
        for block, block_cache in zip(self.blocks, kv_caches):
            x, new_block_cache = block(x, kv_cache=block_cache)
            new_kv_caches.append(new_block_cache)

        x      = self.final_norm(x)
        logits = self.output_head(x)
        return logits, new_kv_caches
