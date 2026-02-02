import os
from sklearn.model_selection import train_test_split

SRC_FILE = "data/raw/en-ta/opensubtitles.en"
TGT_FILE = "data/raw/en-ta/opensubtitles.ta"

OUT_DIR = "data/raw/en-ta"
MAX_LEN = 128
VAL_SPLIT = 0.05
SEED = 42

def clean(text):
    return text.strip().replace("\n", " ")

def valid_pair(en, ta):
    if not en or not ta:
        return False
    if len(en.split()) > MAX_LEN:
        return False
    if len(ta.split()) > MAX_LEN:
        return False
    return True

with open(SRC_FILE, "r", encoding="utf-8") as f:
    en_lines = f.readlines()

with open(TGT_FILE, "r", encoding="utf-8") as f:
    ta_lines = f.readlines()

assert len(en_lines) == len(ta_lines), "EN and TA files are not aligned"


pairs = []

for en, ta in zip(en_lines, ta_lines):
    en = clean(en)
    ta = clean(ta)

    if valid_pair(en, ta):
        pairs.append((en, ta))

print(f"Clean parallel pairs: {len(pairs)}")


train_pairs, valid_pairs = train_test_split(
    pairs,
    test_size=VAL_SPLIT,
    random_state=SEED
)


os.makedirs(OUT_DIR, exist_ok=True)

def save(pairs, prefix):
    with open(f"{prefix}.en", "w", encoding="utf-8") as fen, \
         open(f"{prefix}.ta", "w", encoding="utf-8") as fta:
        for en, ta in pairs:
            fen.write(en + "\n")
            fta.write(ta + "\n")

save(train_pairs, f"{OUT_DIR}/train")
save(valid_pairs, f"{OUT_DIR}/valid")

print("English–Tamil OpenSubtitles dataset prepared.")
