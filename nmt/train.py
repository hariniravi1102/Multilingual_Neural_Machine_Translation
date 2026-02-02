import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.optim import Adam
from itertools import cycle
from tqdm import tqdm

from data.dataset import build_dataloader
from nmt.transformer import TransformerNMT

ROOT = "C:/Users/Harini/PycharmProjects/NMT"
DATA_DIR = os.path.join(ROOT, "data", "ids")
CKPT_DIR = os.path.join(ROOT, "checkpoints_multilingual")
os.makedirs(CKPT_DIR, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


VOCAB_SIZE = 16000
PAD_ID = 0

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048


BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-4


train_loader_en_hi = build_dataloader(
    src_file=os.path.join(DATA_DIR, "en-hi", "train.en.ids"),
    tgt_file=os.path.join(DATA_DIR, "en-hi", "train.hi.ids"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

train_loader_en_ta = build_dataloader(
    src_file=os.path.join(DATA_DIR, "en-ta", "train.en.ids"),
    tgt_file=os.path.join(DATA_DIR, "en-ta", "train.ta.ids"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader_en_hi = build_dataloader(
    src_file=os.path.join(DATA_DIR, "en-hi", "valid.en.ids"),
    tgt_file=os.path.join(DATA_DIR, "en-hi", "valid.hi.ids"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

valid_loader_en_ta = build_dataloader(
    src_file=os.path.join(DATA_DIR, "en-ta", "valid.en.ids"),
    tgt_file=os.path.join(DATA_DIR, "en-ta", "valid.ta.ids"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = TransformerNMT(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = Adam(model.parameters(), lr=LR)


def make_src_mask(src):

    return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt):

    B, T = tgt.size()

    pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(
        torch.ones((T, T), device=tgt.device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    return pad_mask & causal_mask


def train_one_epoch():
    model.train()
    total_loss = 0.0
    steps = 0

    hi_iter = cycle(train_loader_en_hi)
    ta_iter = cycle(train_loader_en_ta)

    num_steps = min(len(train_loader_en_hi), len(train_loader_en_ta))

    for _ in tqdm(range(num_steps), desc="Training"):
        for loader in (hi_iter, ta_iter):
            batch = next(loader)

            src = batch["src"].to(DEVICE)
            tgt_in = batch["tgt_input"].to(DEVICE)
            tgt_out = batch["tgt_output"].to(DEVICE)

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)

            optimizer.zero_grad()

            logits, _ = model(src, tgt_in, src_mask, tgt_mask)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_out.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


@torch.no_grad()
def validate(loader):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt_in = batch["tgt_input"].to(DEVICE)
        tgt_out = batch["tgt_output"].to(DEVICE)

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        logits, _ = model(src, tgt_in, src_mask, tgt_mask)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_out.reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss = train_one_epoch()
    val_hi = validate(valid_loader_en_hi)
    val_ta = validate(valid_loader_en_ta)
    val_loss = (val_hi + val_ta) / 2

    print(f"Train loss : {train_loss:.4f}")
    print(f"Val EN-HI  : {val_hi:.4f}")
    print(f"Val EN-TA  : {val_ta:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(CKPT_DIR, "best_model.pt")
        )
        print("Saved best multilingual model")
