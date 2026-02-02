# transformer.py
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)
