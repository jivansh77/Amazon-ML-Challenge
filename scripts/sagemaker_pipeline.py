"""SageMaker entry point: the whole matching pipeline on one large machine, then cross-encoder scoring.

Channels: ds (train/ and test/ folders of the dataset), ce_base (fine-tuned e5-base cross-encoder),
ce_fr (French-adapted e5-small cross-encoder). Large intermediates stay on the local volume; the small
results (scores, configs, models, logs) are copied to /opt/ml/checkpoints/out, which SageMaker syncs to S3
while the job runs. Extra hyperparameters are appended to the train stage (e.g. --cap_rname 5).
"""
import os, shutil, subprocess, sys, time

JOBS = str(max(4, (os.cpu_count() or 8) - 4))
ROOT = os.environ.get("BER_ROOT", "/opt/ml")            # (override for a local dry run)
D = os.environ.get("BER_INPUT", f"{ROOT}/input/data")
W, WFR = f"{ROOT}/work", f"{ROOT}/work_fr"
OUT = f"{ROOT}/checkpoints/out"
os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
EXTRA = sys.argv[1:]


def sh(name, cmd):
    t = time.time()
    print(f"==== {name}: {' '.join(cmd)}", flush=True)
    with open(f"{OUT}/{name}.log", "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); f.write(line); f.flush()
    print(f"==== {name} exit {p.wait()} in {time.time() - t:.0f}s", flush=True)
    if p.returncode:
        sys.exit(p.returncode)


def keep(src_dir, names, prefix=""):
    for n in names:
        if os.path.exists(f"{src_dir}/{n}"):
            dst = f"{OUT}/{prefix}{n}"
            (shutil.copytree if os.path.isdir(f"{src_dir}/{n}") else shutil.copy)(f"{src_dir}/{n}", dst)


if not os.environ.get("BER_ROOT"):
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.35.2", "xgboost==3.2.0", "sparse_dot_topn",
                "indic_transliteration", "rapidfuzz", "lightgbm", "scikit-learn", "transformers==4.57.1", "sentence-transformers"], check=True)
run = [sys.executable, "-u", "scripts/run.py", "--data", f"{D}/ds", "--jobs", JOBS]
sh("block", run + ["--work", W, "--stages", "norm,block", "--splits", "train,test", "--routes", "tok",
                   "--tok_max_df", "0.01", "--indic", "artifacts/indic_dict.json"])
if not os.environ.get("BER_SKIP_DENSE"):       # (dry run without a GPU)
    sh("dense", run + ["--work", W, "--stages", "dense", "--splits", "train,test", "--k_dense", "30", "--k_rev", "5"])
sh("train", run + ["--work", W, "--stages", "namerev,train", "--splits", "train,test", "--stage2", "--global_sims",
                   "--decoy_feats", "--model", "xgb", "--neg_rate", "0.2", "--frac_a", "0.45", "--frac_b", "0.45",
                   "--cap_tok", "20", "--cap_dense", "20", "--cap_rdense", "2", "--rounds", "4000", "--s2_topk", "15",
                   "--s2_minp", "0.005"] + EXTRA)
keep(W, ["val_scores.parquet", "cfg.json", "model.json", "model1.json", "tokodds_extra.parquet", "tokodds_missing.parquet"])
sh("test", run + ["--work", W, "--stages", "test", "--prior_thr"])
keep(W, ["test_scores.parquet"])
# France: the same model with the calibrated French decoy words (artifacts/odds_extra_france_iso.parquet)
os.makedirs(WFR, exist_ok=True)
sh("test_fr", run + ["--work", WFR, "--reuse", W, "--stages", "test", "--prior_thr",
                     "--odds_extra", "artifacts/odds_extra_france_iso.parquet"])
keep(WFR, ["test_scores.parquet"], prefix="fr_")
# cross-encoders on the new uncertain bands: e5-base for validation + every test pair, French CE for France
py = [sys.executable, "-u"]
sh("ce_data", py + ["scripts/ce_data.py", "--data", f"{D}/ds", "--work", W, "--out", f"{ROOT}/ce_data", "--parts", "val,test"])
sh("ce_base", py + ["scripts/ce_train.py", "--work", f"{ROOT}/ce_base", "--ce_data", f"{ROOT}/ce_data",
                    "--model_dir", f"{D}/ce_base", "--maxlen", "128"])
keep(f"{ROOT}/ce_base", ["ce_val.parquet", "ce_test.parquet"], prefix="cebase_")
sh("ce_data_fr", py + ["scripts/ce_data.py", "--data", f"{D}/ds", "--work", WFR, "--out", f"{ROOT}/ce_data_fr",
                       "--parts", "test", "--test_countries", "unlabelled"])
sh("ce_fr", py + ["scripts/ce_train.py", "--work", f"{ROOT}/ce_fr", "--ce_data", f"{ROOT}/ce_data_fr",
                  "--model_dir", f"{D}/ce_fr", "--maxlen", "128"])
keep(f"{ROOT}/ce_fr", ["ce_test.parquet"], prefix="cefr_")
print("ALL DONE", flush=True)
