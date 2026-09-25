"""Bundle src/, scripts/ and artifacts/ into a single Kaggle script kernel and push it.

  python scripts/kaggle_push.py --slug ber-baseline --args "--stages norm,block,train,test" [--gpu]
       [--src-kernel jvmusic/ber-xxx]   # mount a previous kernel's /kaggle/working as input

The kernel extracts the code, installs the few missing pip packages and runs scripts/run.py
with --data pointing at the mounted competition dataset.
"""
import argparse, base64, io, json, os, subprocess, tarfile

ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--entry", default="scripts/run.py")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--src-kernel", action="append", default=[])
ap.add_argument("--user", default="jvmusic")
ap.add_argument("--dataset", action="append", default=[], help="extra Kaggle datasets to mount")
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for d in ["src", "scripts", "artifacts"]:
        t.add(os.path.join(root, d), arcname=d,
              filter=lambda x: None if "__pycache__" in x.name else x)
blob = base64.b64encode(buf.getvalue()).decode()
code = f'''import base64, io, os, subprocess, sys, tarfile, glob
os.makedirs("/kaggle/working/code", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{blob}")), mode="r:gz").extractall("/kaggle/working/code")
subprocess.run("pip install -q sparse_dot_topn indic_transliteration rapidfuzz polars lightgbm", shell=True)
data = glob.glob("/kaggle/input/**/dataset/train", recursive=True)[0].rsplit("/", 1)[0]
cmd = f"cd /kaggle/working/code && python -u {a.entry} --data {{data}} --work /kaggle/working/work {a.args}"
print(cmd, flush=True)
r = subprocess.run(cmd, shell=True)
subprocess.run("rm -rf /kaggle/working/code", shell=True)
# never fail the kernel itself: Kaggle discards /kaggle/working of failed runs, and a crash late in the
# pipeline (e.g. OOM kill) must not throw away stage outputs that were already written
print("PIPELINE_EXIT=%d" % r.returncode, flush=True)
'''
kd = f"/tmp/claude-0/kk/{a.slug}"
os.makedirs(kd, exist_ok=True)
open(f"{kd}/main.py", "w").write(code)
meta = {"id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": "main.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": a.gpu, "enable_internet": True,
        "dataset_sources": [f"{a.user}/ml-challenge"] + a.dataset, "competition_sources": [],
        "kernel_sources": a.src_kernel}
json.dump(meta, open(f"{kd}/kernel-metadata.json", "w"))
print(subprocess.run(["kaggle", "kernels", "push", "-p", kd], capture_output=True, text=True).stdout)
