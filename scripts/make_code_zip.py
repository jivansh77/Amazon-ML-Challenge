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
SCRIPTS = ["run.py", "fit_translit.py", "ce_data.py", "ce_train.py", "fit_pseudo_odds.py", "blend.py", "final_build.py"]

readme = """# Business Entity Resolution (team Yoddhas)

Pipeline: normalise -> block (word TF-IDF + multilingual-e5 dense retrieval, forward + reverse, + a reverse
character-3-gram name route for records without an address) -> stage-1 XGBoost (also the learned candidate
filter: top 15, p >= 0.005) -> stage-2 XGBoost -> multilingual-e5 cross-encoders on the uncertain band -> logit
blend -> exclusivity + per-country thresholds -> label-free France generator rules.
The pipeline runs twice (reverse name route k = 5 and k = 10) and the two runs are averaged.
Models: XGBoost (Apache-2.0), intfloat/multilingual-e5-small (MIT, 118M) and -base (MIT, 278M). No external data.

## Environment
Python 3.12, `pip install -r requirements.txt`. Developed on Kaggle (4 CPU, 30 GB RAM, 1x T4 / P100 16 GB) and
AWS (the reverse-name-route runs need ~190 GB RAM: ml.g5.12xlarge). The dense, train, test and cross-encoder steps
need a CUDA GPU; everything else runs on CPU. `<D>` is the folder that contains `train/` and `test/`.

## Reproduce (data -> blocking -> matching -> output)
```
cd code/business_entity_resolution
# 0) Indic -> Latin word dictionary learned from the training pairs (CPU, ~10 min; a copy ships in src/artifacts)
python src/fit_translit.py <D> src/artifacts/indic_dict.json

# 1) normalise + TF-IDF blocking, per country (CPU)
python src/run.py --data <D> --work work --stages norm,block --splits train,test --routes tok --tok_max_df 0.01 \\
    --indic src/artifacts/indic_dict.json
# 2) dense retrieval: multilingual-e5-small, forward top 30 + reverse top 5 (GPU)
python src/run.py --data <D> --work work --stages dense --splits train,test --k_dense 30 --k_rev 5
cp -r work work10                                     # the second run shares steps 1-2
# 3) reverse name route (records without an address) + two-stage XGBoost with decoy-signature features (GPU)
TRAIN="--stages namerev,train --splits train,test --stage2 --global_sims --decoy_feats --model xgb --neg_rate 0.2 \\
    --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 --s2_topk 15 --s2_minp 0.005"
python src/run.py --data <D> --work work   $TRAIN --k_namerev 5  --cap_rname 5
python src/run.py --data <D> --work work10 $TRAIN --k_namerev 10 --cap_rname 10
# 4) score test; France (no training labels) is re-scored with its calibrated decoy words (see step 6)
for W in work work10; do
  python src/run.py --data <D> --work $W --stages test --prior_thr
  python src/run.py --data <D> --work ${W}_fr --reuse $W --stages test --prior_thr \\
      --odds_extra src/artifacts/odds_extra_france_iso.parquet
done

# 5) cross-encoders (GPU). Pair files of 160k training S1 each, never in validation (new S1 per round),
#    plus the uncertain bands (0.02 <= p < 0.998) of validation and test
python src/ce_data.py --data <D> --work work --out ce_data                                        # round 1 data
python src/ce_data.py --data <D> --work work --out ce_data2 --parts train --n_s1 160000 --seed 5  --exclude_s1 ce_data/train.parquet
python src/ce_data.py --data <D> --work work --out ce_data3 --parts train --n_s1 160000 --seed 9  --exclude_s1 ce_data/train.parquet,ce_data2/train.parquet
python src/ce_data.py --data <D> --work work --out ce_data4 --parts train --n_s1 160000 --seed 13 --exclude_s1 ce_data/train.parquet,ce_data2/train.parquet,ce_data3/train.parquet
python src/ce_data.py --data <D> --work work   --out bands   --parts val,test                    # bands of both runs
python src/ce_data.py --data <D> --work work10 --out bands10 --parts val,test
#    a) multilingual-e5-small (55 min): used for France (step 6)
python src/ce_train.py --work ce --ce_data ce_data --train_min 55
#    b) multilingual-e5-base rounds; each continues the previous checkpoint
python src/ce_train.py --work ce_b1 --ce_data ce_data --base_model intfloat/multilingual-e5-base --lr 3e-5 --train_min 150
python src/ce_train.py --work ce_b2 --ce_data ce_data --init_dir ce_b1/ce_model --train_mix ce_data2/train.parquet:1 --lr 2e-5 --train_min 170
python src/ce_train.py --work ce_b3  --ce_data bands --init_dir ce_b2/ce_model --train_mix ce_data3/train.parquet:1 --lr 1.5e-5 --train_min 150
python src/ce_train.py --work ce_b3b --ce_data bands --init_dir ce_b2/ce_model --train_mix ce_data4/train.parquet:1 --lr 1.5e-5 --train_min 150
python src/ce_train.py --work ce_b4  --ce_data bands --init_dir ce_b3/ce_model  --train_mix ce_data4/train.parquet:1 --lr 1.5e-5 --train_min 300
python src/ce_train.py --work ce_b4b --ce_data bands --init_dir ce_b3b/ce_model --train_mix ce_data3/train.parquet:1 --lr 1.5e-5 --train_min 300
#    round 3 also scores the k = 10 run's band
python src/ce_train.py --work ce_b3_10 --ce_data bands10 --model_dir ce_b3/ce_model

# 6) France: decoy words from confident test predictions, isotonic-calibrated onto the training scale on the
#    labelled countries (writes the shipped src/artifacts/odds_extra_france_iso.parquet), and the e5-small
#    cross-encoder adapted to France (self-training on confident French test pairs), scoring France's band
python src/fit_pseudo_odds.py --data <D> --scores work/test_scores.parquet --ce ce/ce_test.parquet \\
    --s1n work/test_s1n.parquet --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet \\
    --calib isotonic --out src/artifacts/odds_extra_france_iso.parquet
python src/ce_data.py --data <D> --work work --out ce_fr_data --parts pseudo --ce_scores ce/ce_test.parquet
for W in work work10; do
  python src/ce_data.py --data <D> --work ${W}_fr --out ce_fr_band_$W --parts test --test_countries unlabelled
done
python src/ce_train.py --work ce_fr --ce_data ce_fr_band_work --init_dir ce/ce_model \\
    --train_mix pseudo.parquet:1,ce_data/train.parquet:0.13 --lr 2e-5 --train_min 30
python src/ce_train.py --work ce_fr10 --ce_data ce_fr_band_work10 --model_dir ce_fr/ce_model

# 7) final decode: average of the two runs, cross-encoder ensemble (rounds 3, 4, 4b; French CE for France),
#    logit blend w = 0.6, exclusivity, thresholds US/India 0.8 and France 0.87, then the France rules
#    A2 (French noise suffixes), HN_A (changed house number), D (category swaps), E (records without an address)
python src/final_build.py --data <D> --runs work,work10 \\
    --ce ce_b3/ce_test.parquet:1,ce_b3_10/ce_test.parquet:1,ce_b4/ce_test.parquet:1,ce_b4b/ce_test.parquet:1 \\
    --ce_fr ce_fr/ce_test.parquet,ce_fr10/ce_test.parquet --rules A2,HN_A,D,E --out output
```
`final_build.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (every pair the two runs
score, 4.41 per S1). With `--runs work --ce ce_b3/ce_test.parquet:1 --ce_fr ce_fr/ce_test.parquet --rules A2,HN_A`
it reproduces our best leaderboard file (0.98667) exactly.

Validation (held-out training S1, US/India, macro F0.5): 0.9902. The France rules use no labels: they come from
statistics of the test records (house-number change rate on confident French pairs, added-word classes) and from
leaderboard probes of the France threshold; see Documentation_template.md, section 4.

## Source layout (src/)
- `ber/normalize.py`, `ber/translit.py`: normalisation (accents, OCR digits, legal forms, street types EN/FR,
  house numbers) and the learned Indic->Latin dictionary
- `ber/blocking.py`, `ber/dense.py`: TF-IDF (sparse_dot_topn) and dense (exact blocked top-k on GPU) retrieval
- `ber/features.py`: pair / competition / decoy-signature / global features, word log-odds fitting
- `ber/pipeline.py`: stages (incl. the reverse name route), candidate union + pruning, decoders, thresholds
- `run.py` (stages norm, block, dense, namerev, train, test), `ce_data.py` + `ce_train.py` (cross-encoders),
  `fit_pseudo_odds.py` (unlabelled-country decoy words), `blend.py` (single-run decode),
  `final_build.py` (final decode + France rules + both output files)
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
    for f in ["indic_dict.json", "odds_extra_france_iso.parquet"]:
        z.write(f"{root}/artifacts/{f}", f"{base}/src/artifacts/{f}")
    z.writestr(f"{base}/README.md", readme)
    z.writestr(f"{base}/requirements.txt", "\n".join(req) + "\n")
    if a.output:
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            z.write(os.path.join(a.output, f), f"output/{f}")
    if a.doc:
        z.write(a.doc, "Documentation_template.md")
print(a.out, [i.filename for i in zipfile.ZipFile(a.out).infolist()][:40])
