"""Build the code zip for portal uploads: code/business_entity_resolution/{src,artifacts,README.md,requirements.txt}.

  python scripts/make_code_zip.py <out.zip>
"""
import os, sys, zipfile, importlib.metadata as md

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = sys.argv[1]
base = "code/business_entity_resolution"
PKGS = ["polars", "numpy", "scipy", "scikit-learn", "lightgbm", "rapidfuzz", "sparse_dot_topn",
        "indic_transliteration", "pyarrow"]
req = []
for p in PKGS:
    try:
        req.append(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        req.append(p)
req += ["torch  # GPU dense retrieval stage only", "sentence-transformers  # GPU dense retrieval stage only"]
readme = """# Business Entity Resolution (team Yoddhas)

Pipeline: normalise -> block (word TF-IDF + multilingual-e5 dense retrieval, unioned) -> pair features
-> LightGBM (optionally two-stage with candidate-competition context) -> F0.5-tuned decoding.

## Reproduce
```
pip install -r requirements.txt
python src/fit_translit.py <dataset_dir> artifacts/indic_dict.json     # learn Indic->Latin map from train GT
# 1) normalise + TF-IDF blocking (CPU)
python src/run.py --data <dataset_dir> --work work --stages norm,block --routes tok --tok_max_df 0.01 --indic artifacts/indic_dict.json
# 2) dense retrieval (GPU)
python src/run.py --data <dataset_dir> --work work --stages dense --k_dense 30
# 3) model + test predictions -> work/output/{matching_results,candidate_pairs}.tsv
python src/run.py --data <dataset_dir> --work work --stages train,test --train_frac 0.3 --cap_tok 20 --cap_dense 20
```
`<dataset_dir>` is the folder containing `train/` and `test/`. See the docstrings in `src/ber/` for details.
"""
def fix(txt):
    return txt.replace('os.path.join(os.path.dirname(__file__), "..", "src")', "os.path.dirname(os.path.abspath(__file__))")

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(f"{root}/src/ber")):
        if f.endswith(".py"):
            z.write(f"{root}/src/ber/{f}", f"{base}/src/ber/{f}")
    for f in ["run.py", "fit_translit.py"]:
        z.writestr(f"{base}/src/{f}", fix(open(f"{root}/scripts/{f}").read()))
    z.write(f"{root}/artifacts/indic_dict.json", f"{base}/artifacts/indic_dict.json")
    z.writestr(f"{base}/README.md", readme)
    z.writestr(f"{base}/requirements.txt", "\n".join(req) + "\n")
print(out)
