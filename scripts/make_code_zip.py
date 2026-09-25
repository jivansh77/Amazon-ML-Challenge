"""Build the submission package.

  python scripts/make_code_zip.py <out.zip>                       # code only (portal "code file" upload)
  python scripts/make_code_zip.py <out.zip> --output <dir> --doc Documentation.md
      # final package: output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/,
      # Documentation_template.md

The code folder holds every source file under src/ (the ber package, the pipeline scripts and the learned
Indic dictionary), a README.md with the exact run order and a pinned requirements.txt.
"""
import argparse, os, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--output", default=None, help="dir with matching_results.tsv and candidate_pairs.tsv")
ap.add_argument("--doc", default=None, help="filled-in methodology document (stored as Documentation_template.md)")
a = ap.parse_args()

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base = "code/business_entity_resolution"
# the Kaggle image the pipeline ran on (python 3.12.13; printed by scripts/print_env.py)
req = ["polars==1.35.2", "pyarrow==24.0.0", "numpy==2.0.2", "scipy==1.16.3", "scikit-learn==1.6.1", "lightgbm==4.6.0",
       "xgboost==3.2.0", "rapidfuzz==3.14.6", "sparse_dot_topn==1.2.0", "indic_transliteration==2.3.82",
       "torch==2.10.0  # CUDA build for the dense / XGBoost / cross-encoder steps", "transformers==5.0.0",
       "sentence-transformers==5.4.1", "tokenizers==0.22.2"]
SCRIPTS = ["run.py", "fit_translit.py", "ce_data.py", "ce_train.py", "fit_pseudo_odds.py", "blend.py"]

readme = """# Business Entity Resolution (team Yoddhas)

Pipeline: normalise -> block (word TF-IDF + multilingual-e5 dense retrieval, forward + reverse, unioned)
-> stage-1 XGBoost (also the learned candidate filter: top 15, p >= 0.005 -> ~4.2 candidates per S1)
-> stage-2 XGBoost -> multilingual-e5 cross-encoder on the uncertain band -> logit blend
-> exclusivity + per-country prior-corrected thresholds.
Models: XGBoost (Apache-2.0), intfloat/multilingual-e5-small (MIT, 118M parameters). No external data.

## Environment
Python 3.12, `pip install -r requirements.txt`. Developed on Kaggle (4 CPU, 30 GB RAM, 1x T4 16 GB).
The dense, train, test and cross-encoder steps need a CUDA GPU; everything else runs on CPU.
`<D>` is the folder that contains `train/` and `test/`; `work/` holds all intermediate files.

## Reproduce (data -> blocking -> matching -> output)
```
cd code/business_entity_resolution
# 0) Indic -> Latin word dictionary learned from the training pairs (CPU, ~10 min; a copy ships in src/artifacts)
python src/fit_translit.py <D> src/artifacts/indic_dict.json
# 1) normalise + TF-IDF blocking, per country (CPU, ~1.5 h train, ~1 h test)
python src/run.py --data <D> --work work --stages norm,block --splits train,test --routes tok --tok_max_df 0.01
# 2) dense retrieval: multilingual-e5-small, forward top 30 + reverse top 5 (GPU, ~2 h)
python src/run.py --data <D> --work work --stages dense --k_dense 30 --k_rev 5
# 3) two-stage XGBoost with decoy-signature features; validation report + val_scores.parquet (GPU, ~1.5 h)
python src/run.py --data <D> --work work --stages train --stage2 --global_sims --decoy_feats --model xgb \\
    --neg_rate 0.2 --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 \\
    --s2_topk 15 --s2_minp 0.005
# 4) score test (writes work/test_scores.parquet and a first work/output/)
python src/run.py --data <D> --work work --stages test --prior_thr
# 5) cross-encoder: pair files, fine-tune for 55 min, score the val + test uncertain bands (GPU)
python src/ce_data.py --data <D> --work work --out work/ce_data
python src/ce_train.py --work work/ce --ce_data work/ce_data --train_min 55
# 6) countries without training labels (France): decoy words from confident test predictions, re-score test
python src/fit_pseudo_odds.py --data <D> --scores work/test_scores.parquet --ce work/ce/ce_test.parquet \\
    --s1n work/test_s1n.parquet --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet \\
    --out work/odds_extra.parquet
python src/run.py --data <D> --work work --stages test --prior_thr --odds_extra work/odds_extra.parquet
python src/ce_data.py --data <D> --work work --out work/ce_data2 --parts test --skip_scored work/ce/ce_test.parquet
python src/ce_train.py --work work/ce2 --ce_data work/ce_data2 --model_dir work/ce/ce_model
# 7) blend + per-country thresholds -> output/matching_results.tsv, output/candidate_pairs.tsv
python src/blend.py --data <D> --work work --ce work/ce,work/ce2 --out output --w 0.6
```
`candidate_pairs.tsv` is the exact set the stage-2 model (and the cross-encoder) score: the stage-1 filter's output.

## Source layout (src/)
- `ber/normalize.py`, `ber/translit.py`: normalisation (accents, OCR digits, legal forms, street types EN/FR,
  house numbers) and the learned Indic->Latin dictionary
- `ber/blocking.py`, `ber/dense.py`: TF-IDF (sparse_dot_topn) and dense (exact blocked top-k on GPU) retrieval
- `ber/features.py`: pair / competition / decoy-signature / global features, word log-odds fitting
- `ber/pipeline.py`: stages, candidate union + pruning, decoders, prior-shift per-country thresholds
- `run.py` (stages norm, block, dense, train, test), `ce_data.py` + `ce_train.py` (cross-encoder),
  `fit_pseudo_odds.py` (unlabelled-country decoy words), `blend.py` (final decode + both output files)
"""


def fix(txt):   # scripts live next to the ber package inside src/
    return txt.replace('os.path.join(os.path.dirname(__file__), "..", "src")', "os.path.dirname(os.path.abspath(__file__))") \
              .replace('os.path.join(os.path.dirname(__file__), "..", "artifacts", "indic_dict.json")',
                       'os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "indic_dict.json")')


with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(f"{root}/src/ber")):
        if f.endswith(".py"):
            z.write(f"{root}/src/ber/{f}", f"{base}/src/ber/{f}")
    for f in SCRIPTS:
        z.writestr(f"{base}/src/{f}", fix(open(f"{root}/scripts/{f}").read()))
    z.write(f"{root}/artifacts/indic_dict.json", f"{base}/src/artifacts/indic_dict.json")
    z.writestr(f"{base}/README.md", readme)
    z.writestr(f"{base}/requirements.txt", "\n".join(req) + "\n")
    if a.output:
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            z.write(os.path.join(a.output, f), f"output/{f}")
    if a.doc:
        z.write(a.doc, "Documentation_template.md")
print(a.out, [i.filename for i in zipfile.ZipFile(a.out).infolist()][:40])
