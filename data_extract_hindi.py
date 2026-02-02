import os
from datasets import load_dataset


DATA_DIR = "data/raw/en-hi"
MAX_LEN = 128
VAL_SPLIT = 0.05
SEED = 42



print(" Loading IITB English–Hindi dataset...")
ds = load_dataset("cfilt/iitb-english-hindi")

print(" Creating train/validation split...")
ds = ds["train"].train_test_split(
    test_size=VAL_SPLIT,
    seed=SEED
)

train_ds = ds["train"]
valid_ds = ds["test"]


def filter_fn(batch):
    translations = batch["translation"]
    keep = []

    for t in translations:
        en = t["en"]
        hi = t["hi"]
        keep.append(
            len(en.split()) < MAX_LEN and
            len(hi.split()) < MAX_LEN and
            len(en.strip()) > 0 and
            len(hi.strip()) > 0
        )

    return keep


print(" Filtering long / empty sentences (batched)...")
train_ds = train_ds.filter(filter_fn, batched=True)
valid_ds = valid_ds.filter(filter_fn, batched=True)

os.makedirs(DATA_DIR, exist_ok=True)


def save_parallel(ds, src_lang, tgt_lang, prefix):
    src_path = f"{prefix}.{src_lang}"
    tgt_path = f"{prefix}.{tgt_lang}"

    with open(src_path, "w", encoding="utf-8") as fs, \
         open(tgt_path, "w", encoding="utf-8") as ft:
        for row in ds:
            fs.write(row["translation"][src_lang].strip() + "\n")
            ft.write(row["translation"][tgt_lang].strip() + "\n")

    print(f" Saved: {src_path}, {tgt_path}")


print(" Writing training files...")
save_parallel(train_ds, "en", "hi", f"{DATA_DIR}/train")

print(" Writing validation files...")
save_parallel(valid_ds, "en", "hi", f"{DATA_DIR}/valid")


print("\n Dataset summary:")
print(f"Train samples: {len(train_ds)}")
print(f"Valid samples: {len(valid_ds)}")

print("\n English–Hindi dataset prepared successfully!")
