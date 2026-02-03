# text_to_ids.py


import os
import sentencepiece as spm


TOKENIZER_MODEL = "C:/Users/Harini/PycharmProjects/NMT/tokenization/spm_en_hi_ta.model"

RAW_DATA_DIRS = {
    "en-hi": [("en", "hi")],
    "en-ta": [("en", "ta")]
}

OUTPUT_ROOT = "C:/Users/Harini/PycharmProjects/NMT/data/ids"



sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()



def encode_src(text):
    """Encoder input: TAGGED source, NO BOS / EOS"""
    return sp.encode(text.strip(), out_type=int)


def encode_tgt(text):
    """Decoder target: ADD BOS + EOS"""
    ids = sp.encode(text.strip(), out_type=int)
    return [BOS_ID] + ids + [EOS_ID]


def process_file(src_path, out_path, is_target=False):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing file: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as out:

        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_target:
                ids = encode_tgt(line)
            else:
                ids = encode_src(line)

            out.write(" ".join(map(str, ids)) + "\n")

for pair, langs in RAW_DATA_DIRS.items():
    for src_lang, tgt_lang in langs:
        for split in ["train", "valid"]:


            src_file = (
                f"C:/Users/Harini/PycharmProjects/NMT/data/raw/"
                f"{pair}/{split}.{src_lang}.tagged"
            )


            tgt_file = (
                f"C:/Users/Harini/PycharmProjects/NMT/data/raw/"
                f"{pair}/{split}.{tgt_lang}"
            )

            out_dir = os.path.join(OUTPUT_ROOT, pair)
            os.makedirs(out_dir, exist_ok=True)

            out_src = os.path.join(out_dir, f"{split}.{src_lang}.ids")
            out_tgt = os.path.join(out_dir, f"{split}.{tgt_lang}.ids")

            # Encoder input
            process_file(src_file, out_src, is_target=False)

            # Decoder target
            process_file(tgt_file, out_tgt, is_target=True)

            print(f"Processed {pair} {split}: {src_lang}.tagged → {tgt_lang}")

print("\nText to token ID conversion completed successfully.")
