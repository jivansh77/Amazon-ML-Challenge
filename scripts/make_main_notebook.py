"""Generate and push the main deliverable notebook (jvmusic/amazon-ml-challenge).

The notebook is self-contained and readable: every source file is written out with
%%writefile, the pipeline runs end-to-end on the attached dataset, the output is checked
with the official validator, and the final submission package is assembled:

  /kaggle/working/output/{matching_results.tsv, candidate_pairs.tsv}
  /kaggle/working/submission/  (output/, code/business_entity_resolution/, Documentation_template.md)
  /kaggle/working/<team>_submission.zip

  python scripts/make_main_notebook.py --args "--stages norm,block,dense,train,test" [--no-push]
"""
import argparse, glob, json, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--args", default="--stages norm,block,train,test")
ap.add_argument("--team", default="team")
ap.add_argument("--slug", default="jvmusic/amazon-ml-challenge")
ap.add_argument("--title", default="Amazon ML Challenge")
ap.add_argument("--note", default="")
ap.add_argument("--no-push", action="store_true")
a = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = "/kaggle/working/submission/code/business_entity_resolution"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [], "execution_count": None, "source": s}


def src_file(path, dest):
    txt = open(os.path.join(root, path)).read()
    # scripts live next to the package inside src/ in the submission layout
    txt = txt.replace('os.path.join(os.path.dirname(__file__), "..", "src")', "os.path.dirname(os.path.abspath(__file__))")
    return code(f"%%writefile {PKG}/src/{dest}\n" + txt)


cells = [
    md("# Amazon ML Challenge 2026: Business Entity Resolution\n\n"
       "The main pipeline notebook. It writes the full source, runs normalise → block → features → "
       "LightGBM → F0.5-aware decoding on the attached dataset, validates the output, and builds the "
       "submission package.\n\n" + (f"**This version:** {a.note}\n\n" if a.note else "") +
       f"Run arguments: `{a.args}`"),
    code("!pip install -q sparse_dot_topn indic_transliteration rapidfuzz polars lightgbm 2>&1 | grep -v WARN | tail -2\n"
         "import os, glob\n"
         f"os.makedirs('{PKG}/src/ber', exist_ok=True)\n"
         "DATA = glob.glob('/kaggle/input/**/dataset/train', recursive=True)[0].rsplit('/', 1)[0]\n"
         "RES = DATA.rsplit('/', 1)[0]\nprint(DATA)"),
    md("## Source code"),
]
for f in sorted(glob.glob(os.path.join(root, "src/ber/*.py"))):
    cells.append(src_file(os.path.relpath(f, root), "ber/" + os.path.basename(f)))
cells += [src_file("scripts/fit_translit.py", "fit_translit.py"), src_file("scripts/run.py", "run.py")]
cells += [
    md("## 1. Learn the Indic→Latin word dictionary from the training ground truth"),
    code(f"!mkdir -p {PKG}/artifacts && python -u {PKG}/src/fit_translit.py $DATA {PKG}/artifacts/indic_dict.json"),
    md("## 2. Run the pipeline end-to-end"),
    code(f"!python -u {PKG}/src/run.py --data $DATA --work /kaggle/working/work --out /kaggle/working/output "
         f"--indic {PKG}/artifacts/indic_dict.json {a.args}"),
    md("## 3. Validate with the official checker"),
    code("!python $RES/utils/validate_submission.py --matching /kaggle/working/output/matching_results.tsv "
         "--candidate /kaggle/working/output/candidate_pairs.tsv --test-dir $DATA/test"),
    md("## 4. Assemble the submission package"),
    code(f"""import shutil, subprocess, json
readme = '''# Business Entity Resolution: reproduction

```
pip install -r requirements.txt
python src/fit_translit.py <dataset_dir> artifacts/indic_dict.json      # learn Indic->Latin map from train GT
python src/run.py --data <dataset_dir> --work work --out output --indic artifacts/indic_dict.json {a.args}
```
`<dataset_dir>` is the folder with `train/` and `test/`. Stages: `norm` (normalisation), `block` (TF-IDF
candidate generation), `dense` (GPU multilingual-e5 retrieval, optional), `train` (features + LightGBM +
decision tuning on a held-out 10% of S1), `test` (writes output/matching_results.tsv and
output/candidate_pairs.tsv).
'''
open('{PKG}/README.md', 'w').write(readme)
pk = ['polars', 'numpy', 'scipy', 'scikit-learn', 'lightgbm', 'rapidfuzz', 'sparse_dot_topn',
      'indic_transliteration', 'torch', 'sentence-transformers']
fr = subprocess.run('pip freeze', shell=True, capture_output=True, text=True).stdout.lower().splitlines()
req = [l for l in fr if l.split('==')[0].replace('_', '-') in {{p.replace('_', '-') for p in pk}}]
open('{PKG}/requirements.txt', 'w').write('\\n'.join(req) + '\\n')
shutil.copytree('/kaggle/working/output', '/kaggle/working/submission/output', dirs_exist_ok=True)
shutil.copy(f'{{RES}}/Documentation_template.md', '/kaggle/working/submission/Documentation_template.md')
shutil.make_archive('/kaggle/working/{a.team}_submission', 'zip', '/kaggle/working/submission')
for f in ['cfg.json']:
    if os.path.exists('/kaggle/working/work/' + f):
        shutil.copy('/kaggle/working/work/' + f, '/kaggle/working/output_cfg.json')
shutil.rmtree('/kaggle/working/work', ignore_errors=True)   # large intermediates
print(open('{PKG}/requirements.txt').read()); print(os.listdir('/kaggle/working'))"""),
]
nb = {"metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4, "cells": cells}
kd = "/tmp/claude-0/kk/main"
os.makedirs(kd, exist_ok=True)
json.dump(nb, open(f"{kd}/amazon-ml-challenge.ipynb", "w"), indent=1)
meta = {"id": a.slug, "title": a.title, "code_file": "amazon-ml-challenge.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": ["jvmusic/ml-challenge"], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open(f"{kd}/kernel-metadata.json", "w"), indent=1)
print("notebook written:", len(cells), "cells")
if not a.no_push:
    print(subprocess.run(["kaggle", "kernels", "push", "-p", kd], capture_output=True, text=True).stdout)
