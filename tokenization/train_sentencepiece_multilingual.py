# train_sentencepiece_multilingual.py

import os
import sentencepiece as spm


# PATH CONFIG

ROOT = "C:/Users/Harini/PycharmProjects/NMT"

TRAIN_FILES = [
    os.path.join(ROOT, "data", "raw", "en-hi", "train.en"),
    os.path.join(ROOT, "data", "raw", "en-hi", "train.hi"),
    os.path.join(ROOT, "data", "raw", "en-ta", "train.en"),
    os.path.join(ROOT, "data", "raw", "en-ta", "train.ta"),
]

TOKENIZER_DIR = os.path.join(ROOT, "tokenization")
os.makedirs(TOKENIZER_DIR, exist_ok=True)

MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "spm_en_hi_ta")



VOCAB_SIZE = 16000
MODEL_TYPE = "unigram"
CHAR_COVERAGE = 1.0

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

LANGUAGE_TAGS = ["<2hi>", "<2ta>"]

spm.SentencePieceTrainer.train(
    input=",".join(TRAIN_FILES),
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=CHAR_COVERAGE,
    pad_id=PAD_ID,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    user_defined_symbols=LANGUAGE_TAGS
)

print("SentencePiece tokenizer trained successfully.")
print(f"Model file : {MODEL_PREFIX}.model")
print(f"Vocab file : {MODEL_PREFIX}.vocab")
print(f"Language tags registered: {LANGUAGE_TAGS}")
