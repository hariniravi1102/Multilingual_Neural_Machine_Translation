import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from comet import load_from_checkpoint
from huggingface_hub import hf_hub_download
from sacrebleu import corpus_bleu, corpus_chrf


model_path = hf_hub_download(
    repo_id="Unbabel/wmt22-comet-da",
    filename="checkpoints/model.ckpt"
)
model = load_from_checkpoint(model_path)



def run_comet(src_file, hyp_file, ref_file, batch_size=8):
    data = []
    with open(src_file, encoding="utf-8") as fs, \
         open(hyp_file, encoding="utf-8") as fh, \
         open(ref_file, encoding="utf-8") as fr:
        for s, h, r in zip(fs, fh, fr):
            data.append({
                "src": s.strip(),
                "mt": h.strip(),
                "ref": r.strip()
            })

    scores = model.predict(
        data,
        batch_size=batch_size,
        gpus=0  # CPU-safe
    )
    return scores.system_score



def run_bleu_chrf(hyp_file, ref_file):
    with open(hyp_file, encoding="utf-8") as fh, \
         open(ref_file, encoding="utf-8") as fr:
        hyps = [h.strip() for h in fh]
        refs = [[r.strip() for r in fr]]

    bleu = corpus_bleu(hyps, refs).score
    chrf = corpus_chrf(hyps, refs).score
    return bleu, chrf



src_hi = "C:/Users/Harini/PycharmProjects/NMT/data/raw/en-hi/valid_small.en"
hyp_hi = "C:/Users/Harini/PycharmProjects/NMT/outputs/valid_hi_pred_llm.txt"
ref_hi = "C:/Users/Harini/PycharmProjects/NMT/data/raw/en-hi/valid_small.hi"

comet_hi = run_comet(src_hi, hyp_hi, ref_hi)
bleu_hi, chrf_hi = run_bleu_chrf(hyp_hi, ref_hi)



src_ta = "C:/Users/Harini/PycharmProjects/NMT/data/raw/en-ta/valid_small.en"
hyp_ta = "C:/Users/Harini/PycharmProjects/NMT/outputs/valid_ta_pred_llm.txt"
ref_ta = "C:/Users/Harini/PycharmProjects/NMT/data/raw/en-ta/valid_small.ta"

comet_ta = run_comet(src_ta, hyp_ta, ref_ta)
bleu_ta, chrf_ta = run_bleu_chrf(hyp_ta, ref_ta)



print("EN to HI")
print(f"BLEU : {bleu_hi:.2f}")
print(f"chrF : {chrf_hi:.2f}")
print(f"COMET: {comet_hi:.3f}")

print("EN to TA ")
print(f"BLEU : {bleu_ta:.2f}")
print(f"chrF : {chrf_ta:.2f}")
print(f"COMET: {comet_ta:.3f}")

print(" AVERAGE")
print(f"BLEU : {(bleu_hi + bleu_ta) / 2:.2f}")
print(f"chrF : {(chrf_hi + chrf_ta) / 2:.2f}")
print(f"COMET: {(comet_hi + comet_ta) / 2:.3f}")
