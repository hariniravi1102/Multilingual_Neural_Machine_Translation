import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sentencepiece as spm
from tqdm import tqdm

from nmt.transformer import TransformerNMT
from inference.decode import beam_search_decode
from inference.masks import create_padding_mask


ROOT = "NMT"

TOKENIZER_MODEL = os.path.join(ROOT, "tokenization", "spm_en_hi_ta.model")
CKPT_PATH = os.path.join(ROOT, "checkpoints_multilingual", "best_model.pt")

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validation data
VALID_EN_HI = os.path.join(ROOT, "data", "raw", "en-hi", "valid.en")
VALID_HI = os.path.join(OUTPUT_DIR, "valid_hi_pred.txt")

VALID_EN_TA = os.path.join(ROOT, "data", "raw", "en-ta", "valid.en")
VALID_TA = os.path.join(OUTPUT_DIR, "valid_ta_pred.txt")

# -------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 16000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
PAD_ID = 0


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
def translate_sentence(sentence, lang_tag, beam_size=4, max_len=60):
    sentence = f"{lang_tag} {sentence}"

    src_ids = sp.encode(sentence, out_type=int)
    src_ids = [BOS_ID] + src_ids + [EOS_ID]

    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
    src_mask = create_padding_mask(src)

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

def translate_file(src_path, out_path, lang_tag):
    with open(src_path, encoding="utf-8") as f:
        sentences = f.readlines()

    with open(out_path, "w", encoding="utf-8") as out:
        for s in tqdm(sentences, desc=f"Translating {lang_tag}"):
            translation = translate_sentence(s.strip(), lang_tag)
            out.write(translation + "\n")


translate_file(VALID_EN_HI, VALID_HI, "<2hi>")

translate_file(VALID_EN_TA, VALID_TA, "<2ta>")

print("Validation translations completed.")
print("Files created:")
print(" - outputs/valid_hi_pred.txt")
print(" - outputs/valid_ta_pred.txt")
