"""LLM cross-encoder: an Apache-2.0 decoder LLM (Qwen2.5-0.5B/1.5B/7B-Instruct) fine-tuned with LoRA as a pair
classifier on the same hard pairs and bands as ce_train.py. Multi-GPU with torchrun (DDP).

  torchrun --nproc_per_node=8 scripts/llm_ce.py --model Qwen/Qwen2.5-7B-Instruct --ce_data ce_data --work work_llm \\
      --train_pairs 600000 --bs 16 --lr 1e-4
  (single GPU: python scripts/llm_ce.py ...)

Writes <work>/ce_val.parquet and <work>/ce_test.parquet (column `ce`), same format as ce_train.py, so blend.py uses it.
"""
import argparse, glob, os, time
import numpy as np, polars as pl, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--ce_data", required=True, help="dir with train.parquet, val_band.parquet, test_band.parquet")
ap.add_argument("--extra_train", default=None, help="more training parquet(s) (a, b, y), comma-separated")
ap.add_argument("--work", required=True)
ap.add_argument("--train_pairs", type=int, default=600_000)
ap.add_argument("--bs", type=int, default=16, help="per-GPU batch")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--maxlen", type=int, default=160)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--train_min", type=float, default=240)
ap.add_argument("--score_only", default=None, help="dir with a saved adapter: skip training")
ap.add_argument("--init_adapter", default=None, help="continue training from a saved adapter (after a lost VM)")
ap.add_argument("--skip_pairs", type=int, default=0, help="skip the first N shuffled training pairs (already seen)")
ap.add_argument("--save_every", type=int, default=1500, help="save the adapter every N steps (sessions can die)")
ap.add_argument("--score_bs", type=int, default=128)
ap.add_argument("--parts", default="val_band,test_band")
ap.add_argument("--chunk", type=int, default=100_000, help="score in chunks of this many pairs; finished chunks are kept "
                                                         "and skipped on a rerun (Colab VMs can disappear)")
a = ap.parse_args()
if not a.score_only and os.path.exists(f"{a.work}/train_done"):     # resumed after training finished: score only
    a.score_only = f"{a.work}/adapter"

ddp = "LOCAL_RANK" in os.environ
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))      # before NCCL init: no stray contexts on GPU 0
if ddp:
    import datetime
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))    # ranks wait for each other's chunks
rank = dist.get_rank() if ddp else 0
world = dist.get_world_size() if ddp else 1
dev = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(dev)
os.makedirs(a.work, exist_ok=True)
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True) if rank == 0 else None

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id


def enc(A, B):
    text = [f"Record A: {x}\nRecord B: {y}\nSame business?" for x, y in zip(A, B)]
    return tok(text, truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt")


if a.score_only:
    model = PeftModel.from_pretrained(model, a.score_only)
else:
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = (PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True) if a.init_adapter
             else get_peft_model(model, cfg))
    for q in model.parameters():                 # trainable adapter + head in fp32, frozen base in bf16
        if q.requires_grad:
            q.data = q.data.float()
model = model.cuda()

if not a.score_only:
    files = [glob.glob(os.path.join(a.ce_data, "**", "train.parquet"), recursive=True)[0]]
    files += a.extra_train.split(",") if a.extra_train else []
    tr = pl.concat([pl.read_parquet(f, columns=["a", "b", "y"]) for f in files]).sample(fraction=1.0, shuffle=True, seed=0)
    tr = tr.slice(a.skip_pairs)
    tr = tr.head(min(a.train_pairs, tr.height) // (world * a.bs) * (world * a.bs))   # equal steps on every rank
    tr = tr[rank::world]                                    # each rank its own shard
    A, B, Y = tr["a"].to_list(), tr["b"].to_list(), tr["y"].to_numpy().astype(np.float32)
    n_steps = len(A) // a.bs
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev]) if ddp else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, 100, n_steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train(); t = time.time(); run = 0.0
    for step in range(n_steps):
        i = step * a.bs
        e = enc(A[i:i + a.bs], B[i:i + a.bs]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(**e).logits.squeeze(-1).float()
        loss = lossf(logit, torch.from_numpy(Y[i:i + a.bs]).cuda())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 200 == 0:
            log(f"step {step}/{n_steps} loss {run:.4f} {a.bs * world * (step + 1) / (time.time() - t):.0f} pairs/s")
        stop = torch.tensor([time.time() - t > a.train_min * 60], device="cuda")
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            log(f"time cap at step {step}"); break
        if rank == 0 and step and step % a.save_every == 0:
            model.save_pretrained(f"{a.work}/adapter")
            with open(f"{a.work}/progress.json", "w") as fp:        # pairs seen, for a resume after a lost machine
                fp.write('{"pairs_seen": %d}' % (a.skip_pairs + (step + 1) * a.bs * world))
    if rank == 0:
        model.save_pretrained(f"{a.work}/adapter"); tok.save_pretrained(f"{a.work}/adapter")
        open(f"{a.work}/train_done", "w").write("1")
model.eval()


@torch.no_grad()
def score(df, bs=None):
    """Scores in order of text length (little padding), returned in the input order."""
    bs = bs or a.score_bs
    x, y = df["a"].to_list(), df["b"].to_list()
    order = np.argsort([len(u) + len(v) for u, v in zip(x, y)], kind="stable")
    out = np.zeros(len(x), np.float32); t = time.time()
    for k, i in enumerate(range(0, len(x), bs)):
        j = order[i:i + bs]
        e = enc([x[q] for q in j], [y[q] for q in j]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[j] = torch.sigmoid(model(**e).logits.squeeze(-1).float()).cpu().numpy()
        if k % 500 == 0:
            log(f"scored {i}/{len(x)} {i / (time.time() - t + 1e-9):.0f} pairs/s")
    return out


for name in a.parts.split(","):
    fs = glob.glob(os.path.join(a.ce_data, "**", f"{name}.parquet"), recursive=True)
    if not fs:
        continue
    d = pl.read_parquet(fs[0]).with_row_index("_i")
    os.makedirs(f"{a.work}/chunks", exist_ok=True)
    for k, c0 in enumerate(range(0, d.height, a.chunk)):
        fn = f"{a.work}/chunks/{name}_{k:04d}.parquet"
        if k % world != rank or os.path.exists(fn):       # chunks are shared out over the GPUs
            continue
        part = d.slice(c0, a.chunk); t = time.time()
        part.with_columns(pl.Series("ce", score(part))).drop("a", "b").write_parquet(fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        print(time.strftime("%H:%M:%S"), f"rank {rank}", name, "chunk", k, part.height, f"scored in {time.time() - t:.0f}s", flush=True)
    if ddp:
        dist.barrier()
    if rank == 0:
        d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{a.work}/chunks/{name}_*.parquet"))]).sort("_i").drop("_i")
        d.write_parquet(f"{a.work}/ce_{name.split('_')[0]}.parquet")
        if "y" in d.columns:
            from sklearn.metrics import roc_auc_score
            log("VAL band AUC  stage-2 p:", round(roc_auc_score(d["y"], d["p"]), 4), " LLM:", round(roc_auc_score(d["y"], d["ce"]), 4))
if ddp:
    dist.destroy_process_group()
