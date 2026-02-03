import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sentencepiece as spm
from tqdm import tqdm
import requests
import json
import re

from nmt.transformer import TransformerNMT
from inference.decode import beam_search_decode
from inference.masks import create_padding_mask

ROOT = "C:/Users/Harini/PycharmProjects/NMT"

TOKENIZER_MODEL = os.path.join(ROOT, "tokenization", "spm_en_hi_ta.model")
CKPT_PATH = os.path.join(ROOT, "checkpoints_multilingual", "best_model.pt")

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


VALID_EN_HI = os.path.join(ROOT, "data", "raw", "en-hi", "valid_small.en")
VALID_EN_TA = os.path.join(ROOT, "data", "raw", "en-ta", "valid_small.en")

OUT_HI = os.path.join(OUTPUT_DIR, "valid_hi_pred_llm.txt")
OUT_TA = os.path.join(OUTPUT_DIR, "valid_ta_pred_llm.txt")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 16000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
PAD_ID = 0


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

def llm_post_edit(text: str, lang: str) -> str:
    if lang == "ta":
        instruction = (
            "You are a Tamil grammar corrector.\n"
            "- Fix spelling and grammar ONLY\n"
            "- Do NOT translate\n"
            "- Do NOT explain\n"
            "- Do NOT add English\n"
            "- Output ONLY corrected Tamil\n"
            "- Output ONE sentence only\n"
        )
    elif lang == "hi":
        instruction = (
            "You are a Hindi grammar corrector.\n"
            "- Fix spelling and grammar ONLY\n"
            "- Do NOT translate\n"
            "- Do NOT explain\n"
            "- Do NOT add English\n"
            "- Output ONLY corrected Hindi\n"
            "- Output ONE sentence only\n"
        )
    else:
        return text

    prompt = f"""
{instruction}

INPUT:
{text}

OUTPUT:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 80,
            "stop": ["\n", "Explanation", "I'm", "I am"]
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        output = response.json()["response"].strip()
        output = re.sub(r"[A-Za-z].*", "", output).strip()
        return output if output else text
    except Exception:
        return text


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

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

print("Loaded multilingual NMT model.")

@torch.no_grad()
def translate(sentence: str, tag: str):
    sentence = f"{tag} {sentence}"

    src_ids = sp.encode(sentence, out_type=int)
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    src_mask = create_padding_mask(src, PAD_ID)

    out_ids = beam_search_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=45,
        sos_idx=BOS_ID,
        eos_idx=EOS_ID,
        beam_size=2,
        device=DEVICE
    )[0].tolist()

    if out_ids and out_ids[0] == BOS_ID:
        out_ids = out_ids[1:]
    if EOS_ID in out_ids:
        out_ids = out_ids[:out_ids.index(EOS_ID)]

    return sp.decode(out_ids)


def translate_file(src_path, out_path, tag, lang):
    with open(src_path, encoding="utf-8") as f:
        lines = f.readlines()

    with open(out_path, "w", encoding="utf-8") as out:
        for s in tqdm(lines, desc=f"Translating {lang}"):
            nmt_out = translate(s.strip(), tag)
            final_out = llm_post_edit(nmt_out, lang)
            out.write(final_out + "\n")

translate_file(VALID_EN_HI, OUT_HI, "<2hi>", "hi")
translate_file(VALID_EN_TA, OUT_TA, "<2ta>", "ta")

print("\nInference + LLaMA post-edit completed.")
print("Files written:")
print(" outputs/valid_hi_pred_llm.txt")
print(" outputs/valid_ta_pred_llm.txt")
