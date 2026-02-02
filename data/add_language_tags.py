# add_language_tags.py

import os

ROOT = "C:/Users/Harini/PycharmProjects/NMT"
DATA_DIR = os.path.join(ROOT, "data", "ids")

def add_tag(src_path, tag):
    new_lines = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            new_lines.append(f"{tag} {line}")

    with open(src_path, "w", encoding="utf-8") as f:
        for l in new_lines:
            f.write(l + "\n")


# EN to HI
add_tag(os.path.join(DATA_DIR, "en-hi", "train.en.ids"), "<2hi>")
add_tag(os.path.join(DATA_DIR, "en-hi", "valid.en.ids"), "<2hi>")

# EN to TA
add_tag(os.path.join(DATA_DIR, "en-ta", "train.en.ids"), "<2ta>")
add_tag(os.path.join(DATA_DIR, "en-ta", "valid.en.ids"), "<2ta>")

print("Language tags added successfully.")