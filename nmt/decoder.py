#decoder.py
import torch
import torch.nn as nn
from .layer import MultiHeadAttention, FeedForward
from .encoder import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None, past_kv=None):
        self_attn_out, new_kv = self.self_attn(
            x, x, x, tgt_mask, past_kv
        )
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(
            x, enc_out, enc_out, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, new_kv


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None, past_kvs=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        new_kvs = []

        for i, layer in enumerate(self.layers):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, kv = layer(x, enc_out, tgt_mask, src_mask, past_kv)
            new_kvs.append(kv)

        logits = self.fc_out(x)
        return logits, new_kvs
