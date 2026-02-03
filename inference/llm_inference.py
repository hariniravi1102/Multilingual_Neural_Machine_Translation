import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import torch
import sentencepiece as spm
import requests

from nmt.transformer import TransformerNMT
from inference.decode import beam_search_decode
from inference.masks import create_padding_mask


ROOT = "C:/Users/Harini/PycharmProjects/NMT"

TOKENIZER_MODEL = os.path.join(ROOT, "tokenization", "spm_en_hi_ta.model")
CKPT_PATH = os.path.join(ROOT, "checkpoints_multilingual", "best_model.pt")

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

model = TransformerNMT(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
).to(DEVICE)

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

print("Loaded multilingual NMT model.")


OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.1:8b"

def is_reasonable_tamil(text: str) -> bool:
    tamil_chars = sum(0x0B80 <= ord(c) <= 0x0BFF for c in text)
    return len(text) >= 6 and tamil_chars / max(len(text), 1) > 0.6

def is_reasonable_hindi(text: str) -> bool:
    hindi_chars = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    return len(text) >= 6 and hindi_chars / max(len(text), 1) > 0.6

def llm_post_edit(text: str, lang: str) -> str:
    if lang == "ta" and not is_reasonable_tamil(text):
        return text
    if lang == "hi" and not is_reasonable_hindi(text):
        return text

    if lang == "ta":
        instruction = (
            "You are Tamil grammar corrector.\n"
            "RULES:\n"
            "- DO NOT change meaning\n"
            "- Output ONLY Tamil\n"
        )
    elif lang == "hi":
        instruction = (
            "You are a Hindi grammar corrector.\n"
            "RULES:\n"
            "- DO NOT change meaning\n"
            "- Output ONLY Hindi\n"
        )
    else:
        return text

    prompt = f"{instruction}\nINPUT:\n{text}\nOUTPUT:\n"

    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "top_p": 0.9,
            "num_predict": 60,
            "stop": ["\n"]
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()["response"].strip()

        out = re.sub(r"[A-Za-z].*", "", out).strip()

        return out if out else text

    except Exception:
        return text

@torch.no_grad()
def translate(
    sentence: str,
    target_lang: str,
    beam_size: int = 4,
    max_len: int = 60,
    post_edit: bool = True
):
    if target_lang == "hi":
        sentence = "<2hi> " + sentence
    elif target_lang == "ta":
        sentence = "<2ta> " + sentence
    else:
        raise ValueError("target_lang must be 'hi' or 'ta'")

    src_ids = sp.encode(sentence, out_type=int)
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
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
    )[0].tolist()

    if out_ids and out_ids[0] == BOS_ID:
        out_ids = out_ids[1:]
    if EOS_ID in out_ids:
        out_ids = out_ids[:out_ids.index(EOS_ID)]

    mt = sp.decode(out_ids)

    if post_edit:
        mt = llm_post_edit(mt, target_lang)

    return mt


if __name__ == "__main__":
    tests = [
        "I like machine learning.",
        "How are you today?",
        "The weather is nice today.",
        "I want to play cricket."
    ]

    for s in tests:
        print("\nEN:", s)
        print("TA:", translate(s, "ta"))
        print("HI:", translate(s, "hi"))
