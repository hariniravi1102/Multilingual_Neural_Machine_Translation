import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sentencepiece as spm

from nmt.transformer import TransformerNMT
from inference.decode import beam_search_decode
from inference.masks import create_padding_mask
from inference.llama_post_edit import llm_post_edit



ROOT = "NMT"

TOKENIZER_MODEL = os.path.join(
    ROOT, "tokenization", "spm_en_hi_ta.model"
)

CKPT_PATH = os.path.join(
    ROOT, "checkpoints_multilingual", "best_model.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

VOCAB_SIZE = 16000
PAD_ID = 0

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048


sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()


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
    src_ids = [BOS_ID] + src_ids + [EOS_ID]

    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
    src_mask = create_padding_mask(src, PAD_ID)

    # ---- beam decode ----
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

    raw_text = sp.decode(out_ids)

    final_text = llm_post_edit(raw_text, target_lang)

    return final_text


if __name__ == "__main__":
    tests = [
        "I never thought it would end like this.",
        "We don’t have much time—let’s go, now"
        "Are you sure this is the right place",
        "I did everything I could to save her.No matter what happens, we stay together."
    ]

    for s in tests:
        print("\nEN :", s)
        print("TA :", translate(s, "ta"))
        print("HI :", translate(s, "hi"))
