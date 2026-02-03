# inference/run_inference.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sentencepiece as spm

from nmt.transformer import TransformerNMT
from inference.decode import beam_search_decode
from inference.masks import create_padding_mask


ROOT = "C:/Users/Harini/PycharmProjects/NMT"

TOKENIZER_MODEL = os.path.join(
    ROOT, "tokenization", "spm_en_hi_ta.model"
)

CKPT_PATH = os.path.join(
    ROOT, "checkpoints_multilingual", "best_model.pt"
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


VOCAB_SIZE = 16000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
PAD_ID = 0


sp = spm.SentencePieceProcessor()
assert sp.load(TOKENIZER_MODEL), "Failed to load SentencePiece model"

BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

HI_TAG_ID = sp.piece_to_id("<2hi>")
TA_TAG_ID = sp.piece_to_id("<2ta>")

print("<2hi> id:", HI_TAG_ID)
print("<2ta> id:", TA_TAG_ID)

model = TransformerNMT(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
).to(DEVICE)

model.load_state_dict(
    torch.load(CKPT_PATH, map_location=DEVICE)
)

model.eval()
print("Loaded multilingual NMT model.")


@torch.no_grad()
def translate(
    sentence: str,
    target_lang: str,
    beam_size: int = 4,
    max_len: int = 60
):

    if target_lang == "hi":
        sentence = "<2hi> " + sentence
    elif target_lang == "ta":
        sentence = "<2ta> " + sentence
    else:
        raise ValueError("target_lang must be 'hi' or 'ta'")


    src_ids = sp.encode(sentence, out_type=int)

    src = torch.tensor(
        src_ids, dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    src_mask = create_padding_mask(src, PAD_ID)


    out_ids = beam_search_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=max_len,
        sos_idx=BOS_ID,
        eos_idx=EOS_ID,
        beam_size=beam_size,
        device=DEVICE
    )

    out_ids = out_ids[0].tolist()

    if out_ids and out_ids[0] == BOS_ID:
        out_ids = out_ids[1:]

    if EOS_ID in out_ids:
        out_ids = out_ids[:out_ids.index(EOS_ID)]

    return sp.decode(out_ids)


if __name__ == "__main__":
    tests = [
        "I like machine learning.",
        "How are you today?",
        "The weather is nice today.",
        "I want to play cricket."
    ]

    for s in tests:
        print("\nEN :", s)
        print("TA :", translate(s, "ta"))
        print("HI :", translate(s, "hi"))
