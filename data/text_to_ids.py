#text_to_ids.py


import os
import sentencepiece as spm


TOKENIZER_MODEL = "C:/Users/Harini/PycharmProjects/NMT/tokenization/spm_en_hi_ta.model"

RAW_DATA_DIRS = {
    "en-hi": [("en", "hi")],
    "en-ta": [("en", "ta")]
}

OUTPUT_ROOT = "C:/Users/Harini/PycharmProjects/NMT/data/ids"

ADD_BOS = True
ADD_EOS = True


sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()


def encode_line(text):
    ids = sp.encode(text.strip(), out_type=int)
    if ADD_BOS:
        ids = [BOS_ID] + ids
    if ADD_EOS:
        ids = ids + [EOS_ID]
    return ids

def process_file(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(tgt_path, "w", encoding="utf-8") as out:
        for line in lines:
            ids = encode_line(line)
            out.write(" ".join(map(str, ids)) + "\n")


for pair, langs in RAW_DATA_DIRS.items():
    for src_lang, tgt_lang in langs:
        for split in ["train", "valid"]:
            src_file = f"C:/Users/Harini/PycharmProjects/NMT/data/raw/{pair}/{split}.{src_lang}"
            tgt_file = f"C:/Users/Harini/PycharmProjects/NMT/data/raw/{pair}/{split}.{tgt_lang}"

            out_dir = os.path.join(OUTPUT_ROOT, pair)
            os.makedirs(out_dir, exist_ok=True)

            out_src = os.path.join(out_dir, f"{split}.{src_lang}.ids")
            out_tgt = os.path.join(out_dir, f"{split}.{tgt_lang}.ids")

            process_file(src_file, out_src)
            process_file(tgt_file, out_tgt)

            print(f"Processed {pair} {split}: {src_lang}, {tgt_lang}")

print("Text to token ID conversion completed.")
