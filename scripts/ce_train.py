"""Cross-encoder (intfloat/multilingual-e5-small, MIT) fine-tuned on hard (S1, S2/S3) candidate pairs.

Reads train/val_band/test_band parquet files (columns a, b [, y, p]) from --ce_data, trains for at most
--train_min minutes, saves the model to <work>/ce_model and writes <work>/ce_val.parquet and ce_test.parquet
with a probability column `ce`. The text of a record is "business_name | business_address".
"""
import argparse, glob, os, time
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--work", required=True)
ap.add_argument("--ce_data", default="/kaggle/input/**/ber-ce-data*")
ap.add_argument("--model_dir", default=None, help="score only, with an already fine-tuned model")
ap.add_argument("--train_min", type=float, default=55)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--maxlen", type=int, default=128)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
root = [d for d in glob.glob(a.ce_data, recursive=True) if os.path.isdir(d)]
f = lambda name: glob.glob(os.path.join(root[0], "**", name), recursive=True)[0]
print("ce data:", root[0], flush=True)
if a.model_dir and not os.path.isdir(a.model_dir):     # allow a glob (mounted kernel output)
    a.model_dir = [d for d in glob.glob(a.model_dir, recursive=True) if os.path.isdir(d)][0]
MODEL = a.model_dir or "intfloat/multilingual-e5-small"
print("model:", MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
if not a.model_dir:
    tr = pl.read_parquet(f("train.parquet"))
    A, B, Y = tr["a"].to_list(), tr["b"].to_list(), tr["y"].to_numpy().astype(np.float32)
    n_steps = len(A) // a.bs
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, 500, n_steps)
    scaler = torch.cuda.amp.GradScaler(); lossf = torch.nn.BCEWithLogitsLoss()
    model.train(); t = time.time(); run = 0.0
    for step in range(n_steps):
        i = step * a.bs
        enc = tok(A[i:i + a.bs], B[i:i + a.bs], truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(Y[i:i + a.bs]).cuda())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 1000 == 0:
            print(f"step {step}/{n_steps} loss {run:.4f} {a.bs * (step + 1) / (time.time() - t):.0f} pairs/s", flush=True)
        if time.time() - t > a.train_min * 60:
            print(f"time cap at step {step} ({a.bs * step} pairs)", flush=True); break
    model.save_pretrained(f"{a.work}/ce_model"); tok.save_pretrained(f"{a.work}/ce_model")
model.eval()


@torch.no_grad()
def score(df, bs=512):
    x, y, out = df["a"].to_list(), df["b"].to_list(), []
    for i in range(0, len(x), bs):
        enc = tok(x[i:i + bs], y[i:i + bs], truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
    return np.concatenate(out)


for name in ["val_band", "test_band"]:
    t = time.time()
    d = pl.read_parquet(f(f"{name}.parquet"))
    d = d.with_columns(pl.Series("ce", score(d))).drop("a", "b")
    d.write_parquet(f"{a.work}/ce_{name.split('_')[0]}.parquet")
    print(name, d.height, f"scored in {time.time() - t:.0f}s", flush=True)
    if "y" in d.columns:
        from sklearn.metrics import roc_auc_score
        print("VAL band AUC  stage-2 p:", round(roc_auc_score(d["y"], d["p"]), 4), " cross-encoder:",
              round(roc_auc_score(d["y"], d["ce"]), 4), " mean:", round(roc_auc_score(d["y"], (d["p"] + d["ce"]) / 2), 4), flush=True)
print("DONE", flush=True)
