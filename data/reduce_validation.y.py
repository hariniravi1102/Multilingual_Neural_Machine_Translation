import os
import random

ROOT = "C:/Users/Harini/PycharmProjects/NMT"

pairs = {
    "en-hi": ("en", "hi"),
    "en-ta": ("en", "ta"),
}

MAX_SAMPLES = 500
SEED = 42

random.seed(SEED)

for pair, (src_lang, tgt_lang) in pairs.items():
    src_path = os.path.join(ROOT, "data", "raw", pair, f"valid.{src_lang}")
    tgt_path = os.path.join(ROOT, "data", "raw", pair, f"valid.{tgt_lang}")

    out_src = os.path.join(ROOT, "data", "raw", pair, f"valid_small.{src_lang}")
    out_tgt = os.path.join(ROOT, "data", "raw", pair, f"valid_small.{tgt_lang}")

    with open(src_path, encoding="utf-8") as fs, \
         open(tgt_path, encoding="utf-8") as ft:
        src_lines = fs.readlines()
        tgt_lines = ft.readlines()

    assert len(src_lines) == len(tgt_lines)

    indices = list(range(len(src_lines)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]

    with open(out_src, "w", encoding="utf-8") as fs_out, \
         open(out_tgt, "w", encoding="utf-8") as ft_out:
        for i in indices:
            fs_out.write(src_lines[i])
            ft_out.write(tgt_lines[i])

    print(f"{pair}: reduced validation set → {MAX_SAMPLES} samples")
