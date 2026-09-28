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
       "sentence-transformers==5.4.1", "tokenizers==0.22.2",
       "peft  # Qwen LoRA (llm_ce.py)", "metaphone==0.6  # Double Metaphone for the R3 route (BSD)"]
SCRIPTS = ["run.py", "fit_translit.py", "ce_data.py", "ce_train.py", "fit_pseudo_odds.py", "blend.py", "llm_ce.py",
           "sagemaker_pipeline.py", "sagemaker_llm.py", "native_route.py", "final_build.py", "france_name_replaced.py",
           "france_thr_safe.py", "france_shared_addr.py", "legal_tie.py", "append_routes.py"]
# route / rule pair files the submitted build reads (the --extra_pairs of final_build.py and the append_routes.py steps)
ROUTES = {"v11": ["native_adds_q90", "kv_r3r6_extras"],
          "v13": ["fr_inv_adds", "fr_acr_adds", "fr_fuzzy_street_adds", "legal_tiebreak_adds", "fr_inv_city_adds_f",
                  "fr_acr_city_adds", "fr_inv_legal_adds"],
          "v15": ["fr_acr_shared_adds", "fr_thr85_safe_adds"],
          "v16": ["fr_unit_adds", "fr_acr_city_shared_adds"],
          "v17": ["legal_tie_adds_v17c"]}
ODDS = ["odds_extra_france_iso.parquet"]   # calibrated French decoy words used by the final runs
R3R6 = ["rt-miss.py", "rt-score.py", "rt-build.py", "rt-miss2-tr.py", "rt-miss2-te.py", "rt-score2.py"]   # R3 / R6 route scripts

readme = """# Business Entity Resolution (team Yoddhas)

Pipeline: normalise -> block (word TF-IDF + multilingual-e5 dense retrieval, forward + reverse, unioned)
-> stage-1 XGBoost (also the learned candidate filter: top 15, p >= 0.005 -> ~4.2 candidates per S1)
-> stage-2 XGBoost -> multilingual-e5 cross-encoder on the uncertain band -> logit blend
-> exclusivity + per-country prior-corrected thresholds.
Models: XGBoost (Apache-2.0), intfloat/multilingual-e5-small and -base (MIT, 118M / 278M parameters) and, in the final build,
a Qwen2.5-7B-Instruct LoRA pair classifier (Apache-2.0, 7.6B parameters). No external data.

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
# 5) cross-encoders (GPU). Pair files of 160k training S1 outside validation (+ val/test uncertain bands), then
#    a) multilingual-e5-small, 55 min (used for the French words and the French cross-encoder below)
python src/ce_data.py --data <D> --work work --out work/ce_data
python src/ce_train.py --work work/ce --ce_data work/ce_data --train_min 55
#    b) multilingual-e5-base, 150 min, then a second round on 160k fresh S1 from its checkpoint
python src/ce_train.py --work work/ce_b1 --ce_data work/ce_data --base_model intfloat/multilingual-e5-base --lr 3e-5 --train_min 150
python src/ce_data.py --data <D> --work work --out work/ce_data2 --parts train --seed 5 --exclude_s1 work/ce_data/train.parquet
python src/ce_train.py --work work/ce_b2 --ce_data work/ce_data --init_dir work/ce_b1/ce_model \\
    --train_mix work/ce_data2/train.parquet:1 --lr 2e-5 --train_min 170
# 6) countries without training labels (France): learn their decoy words from confident test predictions
#    (isotonic-calibrated on the labelled countries), re-score test with them in a second work dir
python src/fit_pseudo_odds.py --data <D> --scores work/test_scores.parquet --ce work/ce/ce_test.parquet \\
    --s1n work/test_s1n.parquet --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet \\
    --out work/odds_extra.parquet
python src/run.py --data <D> --work work_fr --reuse work --stages test --prior_thr --odds_extra work/odds_extra.parquet
#    ... and adapt the e5-small cross-encoder to them (self-training on 300k confident test pairs + 13% of its pairs),
#    scoring their whole uncertain band
python src/ce_data.py --data <D> --work work --out work/ce_fr_data --parts pseudo --ce_scores work/ce/ce_test.parquet
python src/ce_data.py --data <D> --work work_fr --out work/ce_fr_data --parts test --test_countries unlabelled --lo 0.003 --hi 0.9995
python src/ce_train.py --work work/ce_fr --ce_data work/ce_fr_data --init_dir work/ce/ce_model \\
    --train_mix pseudo.parquet:1,work/ce_data/train.parquet:0.13 --lr 2e-5 --train_min 30
# 7) blend (logit, weight 0.6 for XGBoost) + exclusivity + per-country thresholds -> output/
python src/blend.py --data <D> --work work --ce work/ce_b2 --override_scores unlabelled:work_fr/test_scores.parquet \\
    --ce_test_override work/ce_fr/ce_test.parquet --out output --w 0.6 --thr US=0.8,India=0.8,France=0.93
```
Thresholds: `blend.py` prints a prior-shift estimate per country (validation matches per S1 in each score band
divided by the test pairs per S1 in that band = the band's precision on test; keep bands above ~0.78).
US/India use the validation optimum (0.8, which the estimate agrees with); France, which has no labels, uses 0.93,
chosen with leaderboard probes (0.97 -> 0.93 improved it; lower bands fall below break-even precision).
`candidate_pairs.tsv` is the exact set the matching stage runs on:
- the stage-1 filter's output, which the stage-2 model and the cross-encoders score;
- in the final build, the stage-1 output of both runs plus the pairs accepted by the targeted routes.

That is 4.43 pairs per S1, 1.31× the final matches.

## Final build of the submitted file (`avg_ce4_v17c`, public LB 0.989828, the file in `output/`)
Steps 1-6 above produce the model and cross-encoder scores. Step 7 (`blend.py`, one run, France 0.93) was our single-run decode
until 26 Sep. The submitted file replaces it: `src/final_build.py` decodes it from two pipeline runs, a cross-encoder ensemble
and route/rule pair files, and `src/append_routes.py` then appends two more route steps.

1. **Two pipeline runs with the reverse-name route** for address-less records (`namerev` stage: top-5 and top-10 S1 per record by
   name). `src/sagemaker_pipeline.py` holds the exact commands (AWS ml.g5.12xlarge):
```
python src/run.py --data <D> --work nr --stages norm,block --splits train,test --routes tok --tok_max_df 0.01 --indic src/artifacts/indic_dict.json
python src/run.py --data <D> --work nr --stages dense --splits train,test --k_dense 30 --k_rev 5
python src/run.py --data <D> --work nr --stages namerev,train --splits train,test --stage2 --global_sims --decoy_feats --model xgb \\
    --neg_rate 0.2 --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 --s2_topk 15 --s2_minp 0.005
python src/run.py --data <D> --work nr --stages test --prior_thr
python src/run.py --data <D> --work nr_fr --reuse nr --stages test --prior_thr --odds_extra src/artifacts/odds_extra_france_iso.parquet
cp nr_fr/test_scores.parquet nr/fr_test_scores.parquet          # France is scored with the calibrated French decoy words
# second run: the same with --k_namerev 10 into nr10/
```
2. **Cross-encoders** on the uncertain band (steps 5-6 above): multilingual-e5-base rounds 3, 4 and 4b (`ce_data.py` + `ce_train.py`),
   a Qwen2.5-7B-Instruct LoRA pair classifier (`src/llm_ce.py`, Apache-2.0, 7.6B parameters; `src/sagemaker_llm.py` runs it on AWS),
   and the French-adapted e5-small cross-encoder for France.
3. **Route / rule pair files** (in `src/artifacts/routes/`; each adds pairs only for records the decode leaves unclaimed):
   - `v11/native_adds_q90.parquet`: India native-script route, `src/native_route.py`.
   - `v11/kv_r3r6_extras.parquet`: R3 phonetic route (Double Metaphone of the name + house number) and R6 same-name compact-key
     route, written as Kaggle kernels (`src/routes_r3_r6/`): R3 = `rt-miss.py` -> `rt-score.py` -> `rt-build.py`,
     R6 = `rt-miss2-tr.py` + `rt-miss2-te.py` -> `rt-score2.py`. Each kernel finds its inputs (the pipeline's normalised records,
     candidates and scores, and the previous kernel's output) under `/kaggle/input` and writes to `/kaggle/working`; the pairs
     are the R3/R6 adds on records the decode left unclaimed.
   - France name-replaced copies (`src/france_name_replaced.py`, modes default / `--unique_by city` / `fuzzy_street` / `inv_legal` /
     `acr_shared`), the France 0.85 safe subset (`src/france_thr_safe.py`), shared-address sub-number / acronym rules
     (`src/france_shared_addr.py`) and the legal-form tie-break for address-less same-name ties (`src/legal_tie.py`).
   - The France generator rules (`--rules` below: suffix adds, house-number drops, ...) are implemented in `final_build.py`.
4. **Decode** (US/India thresholds 0.8, France 0.87; France generator rules; first match for empty US/India S1):
```
R=src/artifacts/routes
python src/final_build.py --data <D> --runs nr,nr10 \\
  --ce nr/ce_test.parquet:1,nr10/cebase_ce_test.parquet:1,ce_base4/ce_test.parquet:1,ce_base4b/ce_test.parquet:1,qwen_ce_test.parquet:1 \\
  --ce_fr nr/ce_test_france.parquet,nr10/ce_test_france_new.parquet \\
  --rules A2P,A2F,A2N,HN_A,DP,E2,OOC,HNK --collapse_legal --first_min US=0.5,India=0.5 --first_skip_ea_ties \\
  --extra_pairs $R/v11/native_adds_q90.parquet,$R/v11/kv_r3r6_extras.parquet,$R/v13/fr_inv_adds.parquet,$R/v13/fr_acr_adds.parquet,$R/v13/fr_fuzzy_street_adds.parquet,$R/v13/legal_tiebreak_adds.parquet,$R/v13/fr_inv_city_adds_f.parquet,$R/v13/fr_acr_city_adds.parquet,$R/v13/fr_inv_legal_adds.parquet,$R/v15/fr_acr_shared_adds.parquet,$R/v15/fr_thr85_safe_adds.parquet \\
  --out v15b                                                   # 5,838,236 matches
BER_TEST_S1=<D>/test/test_source1.tsv python src/append_routes.py v15b v16 $R/v16/fr_unit_adds.parquet,$R/v16/fr_acr_city_shared_adds.parquet
python src/legal_tie.py --data <D> --claimed v16/pred.parquet --gate_run nr --out legal_tie_adds_v17c.parquet   # = $R/v17/legal_tie_adds_v17c.parquet (735 pairs)
BER_TEST_S1=<D>/test/test_source1.tsv python src/append_routes.py v16 v17c $R/v17/legal_tie_adds_v17c.parquet   # 5,839,939 matches = output/
```
With the shipped route files these commands regenerate `output/matching_results.tsv` byte-for-byte and the same candidate pairs.
The route files themselves are regenerated by the scripts above from the decoded pairs of the previous step (commands in each
script's docstring). Validator: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
--test-dir dataset/test --check-ids` -> PASS.

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
    for f in ODDS:
        z.write(f"{root}/artifacts/{f}", f"{base}/src/artifacts/{f}")
    for v, names in ROUTES.items():
        for f in names:
            z.write(f"{root}/artifacts/{v}/{f}.parquet", f"{base}/src/artifacts/routes/{v}/{f}.parquet")
    for f in R3R6:
        z.write(f"{root}/third_party/kavya_r3r6/{f}", f"{base}/src/routes_r3_r6/{f}")
    z.writestr(f"{base}/README.md", readme)
    z.writestr(f"{base}/requirements.txt", "\n".join(req) + "\n")
    if a.output:
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            z.write(os.path.join(a.output, f), f"output/{f}")
    if a.doc:
        z.write(a.doc, "Documentation_template.md")
print(a.out, [i.filename for i in zipfile.ZipFile(a.out).infolist()][:40])
