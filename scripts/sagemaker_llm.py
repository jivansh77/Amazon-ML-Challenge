"""SageMaker entry: Qwen2.5-7B LoRA pair classifier on all GPUs (scripts/llm_ce.py), resumable.
Work dir /opt/ml/checkpoints/llm is synced to S3 (adapter, progress.json, scored chunks). On a restart (spot
interruption) it continues from the adapter there; otherwise from the `init` channel adapter (a Colab run)."""
import json, os, subprocess, sys, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "transformers==4.57.1", "peft", "scikit-learn"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
W = "/opt/ml/checkpoints/llm"; INIT = "/opt/ml/input/data/init"
args = sys.argv[1:]
if os.path.exists(f"{W}/progress.json") and os.path.exists(f"{W}/adapter/adapter_model.safetensors"):
    seen = json.load(open(f"{W}/progress.json"))["pairs_seen"]
    args = [x for x in args]
    args += ["--init_adapter", f"{W}/adapter", "--skip_pairs", str(seen)]
    print("resuming from checkpoint,", seen, "pairs seen", flush=True)
elif os.path.exists(f"{INIT}/adapter_model.safetensors"):
    seen = json.load(open(f"{INIT}/progress.json"))["pairs_seen"]
    args += ["--init_adapter", INIT, "--skip_pairs", str(seen)]
    print("starting from the init adapter,", seen, "pairs seen", flush=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
n = torch.cuda.device_count()
cmd = ["torchrun", f"--nproc_per_node={n}", "scripts/llm_ce.py" if os.path.exists("scripts/llm_ce.py") else "llm_ce.py", "--model", "Qwen/Qwen2.5-7B-Instruct", "--ce_data", "/opt/ml/input/data/llm",
       "--work", W, "--parts", "val_band,test_sub", "--chunk", "25000"] + args
print(" ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd).returncode)
