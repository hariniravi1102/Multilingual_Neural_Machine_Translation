#masks.py
import torch


def create_padding_mask(seq, pad_id=0):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def create_causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
