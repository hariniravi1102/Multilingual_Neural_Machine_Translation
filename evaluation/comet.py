from comet import download_model, load_from_checkpoint


model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

def run_comet(src_file, hyp_file, ref_file):
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

    scores = model.predict(data, batch_size=8)
    return scores.system_score

# EN → HI
score_hi = run_comet(
    "data/raw/en-hi/valid.en",
    "outputs/valid_hi_pred.txt",
    "data/raw/en-hi/valid.hi"
)

score_ta = run_comet(
    "data/raw/en-ta/valid.en",
    "outputs/valid_ta_pred.txt",
    "data/raw/en-ta/valid.ta"
)

print("COMET EN to HI:", score_hi)
print("COMET EN to TA:", score_ta)
print("AVG COMET:", (score_hi + score_ta) / 2)
