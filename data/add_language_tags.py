# add_language_tags_separate.py


import os

ROOT = "C:/Users/Harini/PycharmProjects/NMT"
DATA_DIR = os.path.join(ROOT, "data", "raw")

def add_tag_separate(src_path, out_path, tag):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing file: {src_path}")

    with open(src_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(f"{tag} {line}\n")

    print(f"Created tagged file: {out_path}")


add_tag_separate(
    os.path.join(DATA_DIR, "en-hi", "train.en"),
    os.path.join(DATA_DIR, "en-hi", "train.en.tagged"),
    "<2hi>"
)

add_tag_separate(
    os.path.join(DATA_DIR, "en-hi", "valid.en"),
    os.path.join(DATA_DIR, "en-hi", "valid.en.tagged"),
    "<2hi>"
)



add_tag_separate(
    os.path.join(DATA_DIR, "en-ta", "train.en"),
    os.path.join(DATA_DIR, "en-ta", "train.en.tagged"),
    "<2ta>"
)

add_tag_separate(
    os.path.join(DATA_DIR, "en-ta", "valid.en"),
    os.path.join(DATA_DIR, "en-ta", "valid.en.tagged"),
    "<2ta>"
)

print("\nLanguage tags added as separate files successfully.")
