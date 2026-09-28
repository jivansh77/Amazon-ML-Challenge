"""Build the submission package.

  python scripts/make_code_zip.py <out.zip>                       # code only (portal "code file" upload)
  python scripts/make_code_zip.py <out.zip> --output <dir> --doc Documentation.md
      # final package: output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/,
      # Documentation_template.md

The code folder holds every source file under src/ (the ber package, the pipeline and route scripts, the learned Indic
dictionary, the calibrated French words and the route pair files of the final build), a README.md with the exact run order
and a pinned requirements.txt.
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
       "peft==0.20.0  # Qwen LoRA (llm_ce.py)", "metaphone==0.6  # Double Metaphone for the phonetic route (BSD)",
       "# AWS steps (sagemaker_pipeline.py, sagemaker_llm.py): SageMaker PyTorch 2.7.1 GPU image + transformers==4.57.1"]
SCRIPTS = ["run.py", "fit_translit.py", "ce_data.py", "ce_train.py", "fit_pseudo_odds.py", "blend.py", "llm_ce.py",
           "sagemaker_pipeline.py", "sagemaker_llm.py", "native_route.py", "phonetic_route_candidates.py",
           "phonetic_route_score.py", "phonetic_route_build.py", "samename_route_candidates.py", "samename_route_score.py",
           "phonetic_samename_merge.py", "france_name_replaced.py", "france_thr_safe.py",
           "france_shared_addr.py", "legal_tie.py", "final_build.py", "append_routes.py"]
# route / rule pair files the submitted build reads (the --extra_pairs of final_build.py and the append_routes.py steps)
ROUTES = {"v11": ["native_adds_q90", "kv_r3r6_extras"],
          "v13": ["fr_inv_adds", "fr_acr_adds", "fr_fuzzy_street_adds", "legal_tiebreak_adds", "fr_inv_city_adds_f",
                  "fr_acr_city_adds", "fr_inv_legal_adds"],
          "v15": ["fr_acr_shared_adds", "fr_thr85_safe_adds"],
          "v16": ["fr_unit_adds", "fr_acr_city_shared_adds"],
          "v17": ["legal_tie_adds_v17c"]}
ODDS = ["odds_extra_france_iso.parquet"]   # calibrated French decoy words used by the final runs
ZIP_NAME = {"kv_r3r6_extras": "phonetic_samename_adds"}   # route files shipped under a descriptive name

readme = r"""# Business Entity Resolution (team Yoddhas, public leaderboard 0.989828)

## What this is
The task: for every business in Source 1 (S1), find all records in Sources 2 and 3 (S2/S3) that describe the same business.
Names and addresses are noisy, there are no shared ids, and many S2/S3 records are generated lookalikes that match nothing.
This folder is the full pipeline that produced our submission. Its outputs (in `output/` next to this folder):
- `matching_results.tsv`: for every test S1, the S2/S3 ids we match to it (comma-separated; empty = no match).
- `candidate_pairs.tsv`: for every test S1, the S2/S3 ids our blocking proposed, i.e. every pair the matcher scored.

Method: retrieve candidates (word TF-IDF, multilingual dense retrieval, a name-only route for records without an address),
score them with a two-stage XGBoost, re-score the uncertain ones with a cross-encoder ensemble, decode with exclusivity and
per-country thresholds, then add France rules and targeted routes for records the decode leaves unclaimed.
Models: XGBoost (Apache-2.0), intfloat/multilingual-e5-small and -base (MIT, 118M / 278M parameters), Qwen2.5-7B-Instruct
LoRA pair classifier (Apache-2.0, 7.6B parameters). No external data. `Documentation_template.md` explains the method.

## Terms used below
- **S1**: a Source 1 business. **Record**: an S2 or S3 entry. **Pair**: an (S1, record) combination.
- **Candidates**: the pairs we score at all, found by the retrieval ("blocking") routes.
- **p**: the stage-2 XGBoost probability that a pair matches. **Uncertain band**: 0.02 <= p < 0.998, the pairs the
  cross-encoders re-score.
- **Cross-encoder (CE)**: a transformer that reads both texts ("name | address") together and scores the pair. The e5 models
  are fine-tuned in rounds; Qwen is an LLM tuned the same way with LoRA. **Blend**: 0.6 x XGBoost + 0.4 x cross-encoders, on
  the logit scale.
- **Decode**: turning scores into matches. Each record goes to at most one S1, its best (**exclusivity**), and is kept if the
  blend clears that country's threshold. A record the decode has given to an S1 is **claimed**.
- **Route file**: a list of extra (S1, record) pairs for records the decode leaves unclaimed (e.g. the India native-script
  route, France acronym records at the S1's address).
- **Run**: one full pipeline pass with its scores (`work`, `nr`, `nr10`). **Build**: one decode of the runs' scores plus
  route files (v2h, v3, ... v17c); v17c is the submitted file.
- **France rules**: label-free rules for France (training has no French labels). A2P, DP, HN_A, ... are their names in
  `final_build.py --rules`; the documentation (section 6.3) explains each.

## How the steps fit together
| Step | Produces | Where it ran | Time |
|---|---|---|---|
| 1. Main run `work` | normalised records, candidates, stage-2 scores | Kaggle (T4) | 6-7 h |
| 2. Cross-encoders | training sets, e5-small and e5-base rounds 1-3b | Kaggle (T4) | about 13 GPU h |
| 3. France | French decoy words, French-adapted cross-encoder | Kaggle (T4) | about 1 h |
| 4. Reverse-name runs `nr`, `nr10` | the two runs the final build averages | AWS | 3 h and 5.4 h |
| 5. Rounds 4/4b and Qwen | more cross-encoder scores on the runs' bands | Kaggle (P100), AWS | 5 h each; Qwen 13 h |
| 6. Route files | already in `src/artifacts/routes/`; the commands show how each was made | | |
| 7. Final build | `output/matching_results.tsv` and `output/candidate_pairs.tsv` | CPU | under 1 h |

## Environment
- Python 3.12, `pip install -r requirements.txt`: pinned, the Kaggle image most steps ran on (4 CPU, 30 GB RAM,
  1x T4 or P100 16 GB).
- AWS steps: SageMaker PyTorch 2.7.1 GPU image (Python 3.12, CUDA 12.8) with transformers 4.57.1 and peft 0.20.0. The top-5
  run and Qwen used ml.g5.12xlarge (4x A10G 24 GB, 192 GB RAM), the top-10 run ml.g4dn.16xlarge (1x T4, 256 GB RAM).
- GPU steps: dense, train, test, the cross-encoders, phonetic / same-name route scoring and Qwen; everything else runs on CPU.
- `<D>` is the challenge's dataset folder (it contains `train/` and `test/`). Every other folder name below (`work`, `ce_b2`,
  `nr`, ...) is a work folder the commands create, relative to this folder.
- All sampling is seeded. GPU training can differ in the last digits from one GPU to another, so a rerun gives near-identical
  scores; from the scores, step 7 regenerates `output/` exactly.

## 1. Main run `work` (Kaggle)
```
python src/fit_translit.py <D> src/artifacts/indic_dict.json      # Indic -> Latin word dictionary (a copy ships)
python src/run.py --data <D> --work work --stages norm,block --splits train,test --routes tok --tok_max_df 0.01 \
    --indic src/artifacts/indic_dict.json
python src/run.py --data <D> --work work --stages dense --k_dense 30 --k_rev 5
python src/run.py --data <D> --work work --stages train --stage2 --global_sims --decoy_feats --model xgb --neg_rate 0.2 \
    --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 --s2_topk 15 --s2_minp 0.005
python src/run.py --data <D> --work work --stages test --prior_thr
```

## 2. Cross-encoders (Kaggle GPU)
Each training set holds 160k training S1 outside the validation split and the earlier sets: every true match and the hard
negatives of their candidate lists (about 2.2M pairs). The first set's S1 sample was drawn outside the validation S1 of an
earlier train run (`big`, with `--drop_s1 0.19`), which is rerun here only for that.
```
python src/run.py --data <D> --work big --reuse work --stages train --stage2 --global_sims --model xgb --neg_rate 0.25 \
    --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 --drop_s1 0.19 --s2_topk 15 --s2_minp 0.005
python src/ce_data.py --data <D> --work big --out ce_data --parts train
python src/ce_data.py --data <D> --work work --out ce_data2 --parts train --seed 5 --exclude_s1 ce_data/train.parquet
python src/ce_data.py --data <D> --work work --out ce_data3 --parts train --seed 9 \
    --exclude_s1 ce_data/train.parquet,ce_data2/train.parquet
python src/ce_data.py --data <D> --work work --out ce_data4 --parts train --seed 13 \
    --exclude_s1 ce_data/train.parquet,ce_data2/train.parquet,ce_data3/train.parquet
python src/ce_data.py --data <D> --work work --out ce_dec --parts val,test          # uncertain band (0.02 <= p < 0.998)
# e5-small (for the French words and the French cross-encoder), then e5-base rounds 1, 2, 3 and 3b
python src/ce_train.py --work ce --ce_data ce_dec --train_mix ce_data/train.parquet:1 --train_min 55
python src/ce_train.py --work ce_b1 --ce_data ce_dec --train_mix ce_data/train.parquet:1 \
    --base_model intfloat/multilingual-e5-base --lr 3e-5 --train_min 150
python src/ce_train.py --work ce_b2 --ce_data ce_dec --init_dir ce_b1/ce_model --train_mix ce_data2/train.parquet:1 --lr 2e-5 --train_min 170
python src/ce_train.py --work ce_b3 --ce_data ce_dec --init_dir ce_b2/ce_model --train_mix ce_data3/train.parquet:1 --lr 1.5e-5 --train_min 170
python src/ce_train.py --work ce_b3b --ce_data ce_dec --init_dir ce_b2/ce_model --train_mix ce_data4/train.parquet:1 --lr 1.5e-5 --train_min 150
```
Our e5-small model was trained with the bands of `big` in its folder and then re-scored on `ce_dec`; the first
`ce_train.py` line does both at once.

## 3. France (no training labels)
```
# French decoy words learned from confident test predictions, isotonic-calibrated on US/India
# (= src/artifacts/odds_extra_france_iso.parquet)
python src/fit_pseudo_odds.py --data <D> --scores work/test_scores.parquet --ce ce/ce_test.parquet --s1n work/test_s1n.parquet \
    --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet --out odds_extra_france_iso.parquet
# French-adapted e5-small: self-training on 300k confident French test pairs + 13% of the first training set
python src/ce_data.py --data <D> --work work --out ce_fr_data --parts pseudo --ce_scores ce/ce_test.parquet
python src/ce_data.py --data <D> --work work --out ce_fr_data --parts test --test_countries unlabelled --lo 0.003 --hi 0.9995
python src/ce_train.py --work ce_fr --ce_data ce_fr_data --init_dir ce/ce_model \
    --train_mix pseudo.parquet:1,ce_data/train.parquet:0.13 --lr 2e-5 --train_min 30
```

## 4. Reverse-name runs `nr` (top 5) and `nr10` (top 10) (AWS, `src/sagemaker_pipeline.py`)
Step 1 plus the `namerev` stage: every S2/S3 record without an address gets its top-k S1 by character 3-gram TF-IDF on the
name as extra candidates. The uncertain band is then scored by the e5-base round-3 cross-encoder, and France is re-scored
with the French decoy words.
```
python src/run.py --data <D> --work nr --stages norm,block --splits train,test --routes tok --tok_max_df 0.01 \
    --indic src/artifacts/indic_dict.json
python src/run.py --data <D> --work nr --stages dense --splits train,test --k_dense 30 --k_rev 5
python src/run.py --data <D> --work nr --stages namerev,train --splits train,test --stage2 --global_sims --decoy_feats \
    --model xgb --neg_rate 0.2 --frac_a 0.45 --frac_b 0.45 --cap_tok 20 --cap_dense 20 --cap_rdense 2 --rounds 4000 \
    --s2_topk 15 --s2_minp 0.005 --cap_rname 5 --k_namerev 5
python src/run.py --data <D> --work nr --stages test --prior_thr
python src/run.py --data <D> --work nr_fr --reuse nr --stages test --prior_thr --odds_extra src/artifacts/odds_extra_france_iso.parquet
cp nr_fr/test_scores.parquet nr/fr_test_scores.parquet             # France uses these scores
python src/ce_data.py --data <D> --work nr --out ce_nr --parts val,test
python src/ce_train.py --work nr_ce --ce_data ce_nr --model_dir ce_b3/ce_model
cp nr_ce/ce_test.parquet nr/ce_test.parquet
# French cross-encoder on the France band pairs that ce_fr has not scored yet
python src/ce_data.py --data <D> --work nr_fr --out ce_nr_fr --parts test --test_countries unlabelled --skip_scored ce_fr/ce_test.parquet
python src/ce_train.py --work nr_cefr --ce_data ce_nr_fr --model_dir ce_fr/ce_model
#   nr/ce_test_france.parquet = ce_fr/ce_test.parquet + nr_cefr/ce_test.parquet
```
The AWS jobs: `sagemaker_pipeline.py --cap_rname 5 --k_namerev 5` (`nr`) and `--cap_rname 10 --k_namerev 10` (`nr10`, the same
steps into `nr10/`). The round-3 scores of `nr10` are `nr10/cebase_ce_test.parquet`, and the French cross-encoder scores of
its France band pairs that `nr` does not have are `nr10/ce_test_france_new.parquet`.

## 5. Cross-encoder rounds 4 and 4b (Kaggle P100) and the Qwen LoRA (AWS)
```
python src/ce_train.py --work ce_base4 --ce_data ce_nr --init_dir ce_b3/ce_model --train_mix ce_data4/train.parquet:1 --lr 1.5e-5 --train_min 300
python src/ce_train.py --work ce_base4b --ce_data ce_nr --init_dir ce_b3b/ce_model --train_mix ce_data3/train.parquet:1 --lr 1.5e-5 --train_min 300
# Qwen2.5-7B-Instruct LoRA (rank 16, alpha 32, sequence-classification head) on the 4 GPUs (src/sagemaker_llm.py):
torchrun --nproc_per_node=4 src/llm_ce.py --model Qwen/Qwen2.5-7B-Instruct --ce_data llm --work qwen --parts val_band,test_sub \
    --chunk 25000 --bs 6 --lr 0.0001 --maxlen 128 --save_every 500 --score_bs 64 --train_min 720 --train_pairs 2300000 \
    --init_adapter qwen_init --skip_pairs 312768
cp qwen/ce_test.parquet qwen_ce_test.parquet
```
`llm/` holds `train.parquet` (= `ce_data2/train.parquet`), `val_band.parquet` (= `ce_dec/val_band.parquet`) and
`test_sub.parquet`: the `ce_dec` test band cut to the pairs whose blend of stage-2 p and e5-base cross-encoder is in
[0.05, 0.995) (France keeps the French cross-encoder). `qwen_init/` is the adapter after the first 312,768 pairs of the same
shuffled file, trained with the same script on a Colab A100; training ran 720 minutes, to 1,920,792 pairs.

## 6. Route and rule pair files (`src/artifacts/routes/`)
These files ship ready-made, and step 7 reads them directly; this section records how each one was made. Each file adds pairs
only for records the decode leaves unclaimed; `final_build.py --extra_pairs` applies them in the order of step 7 (an earlier
file wins a record). Each file was made against the build current at the time, i.e. the decode of step 7 with the files
before it in that order ("Against" = the build the script read as `--claimed`):
- v2h: the first decode of `nr` + `nr10` with the rounds 3/4/4b ensemble and the France rules A2, A2F, HN_A, DP, E2, OOC, HNK.
- v3: v2h + rule fixes (A2P, A2N, the DP guard, dotted legal forms). v4: + native-script route. v6: + phonetic and same-name routes.
- v9s: + Qwen in the ensemble and the first tie-break. v10s ... v16: + the files below, in order.

| File | Pairs | Made by | Against |
|---|---|---|---|
| `v11/native_adds_q90` | 8,480 | `native_route.py --work nat --claimed <v3> --cand <v3 candidates> --thr 0.9` | v3 |
| `v11/phonetic_samename_adds` | 7,156 | phonetic and same-name route scripts below, then `phonetic_samename_merge.py` | v2h (base), v4 |
| `v13/fr_inv_adds` | 5,135 | `france_name_replaced.py --claimed <v9s> --out_inv` | v9s |
| `v13/fr_acr_adds` | 1,727 | `france_name_replaced.py --claimed <v10s> --out_acr` | v10s |
| `v13/fr_fuzzy_street_adds` | 1,181 | `france_name_replaced.py --mode fuzzy_street --out` | v11 |
| `v13/legal_tiebreak_adds` | 1,258 | first version of the legal-form tie-break (rule below) | v9s (before it) |
| `v13/fr_inv_city_adds_f` | 1,819 | `france_name_replaced.py --unique_by city --drop_foreign_handles --out_inv` | v12 |
| `v13/fr_acr_city_adds` | 844 | the same run, `--out_acr` | v12 |
| `v13/fr_inv_legal_adds` | 84 | `france_name_replaced.py --mode inv_legal --out` | v13 before it |
| `v15/fr_acr_shared_adds` | 371 | `france_name_replaced.py --mode acr_shared --out` | v14b |
| `v15/fr_thr85_safe_adds` | 288 | `france_thr_safe.py --base <v15> --low <v15 built with --thr US=0.8,India=0.8,France=0.85>` | v15 |
| `v16/fr_unit_adds` | 297 | `france_shared_addr.py --mode unit` | v15b |
| `v16/fr_acr_city_shared_adds` | 709 | `france_shared_addr.py --mode acr_city` | v15b |
| `v17/legal_tie_adds_v17c` | 735 | `legal_tie.py --gate_run nr` | v16 |

Every script above takes `--data <D>` plus the listed arguments; its docstring gives the full command.
`legal_tiebreak_adds`: an address-less record whose core name (as a set of words) belongs to 2 or 3 S1 and whose normalised
legal form is on exactly one of them goes to that S1. `legal_tie.py` is the later, refined version of the same rule.

**Phonetic route** (`phonetic_route_*.py`): Double Metaphone codes of the first two name words + the house number.
**Same-name route** (`samename_route_*.py`): the same name with the spaces removed, at a shared house number or city word.
Both are for US/India records no other route proposed. Their candidates are scored with the e5-base round-2 cross-encoder + a
small XGBoost, gated on clean validation, and added to a base build (v2h: its `matching_results.tsv` and `candidate_pairs.tsv`);
`phonetic_samename_merge.py` combines the two into one route file.
```
python src/phonetic_route_candidates.py --data <D> --work work --out phonetic_cand
python src/phonetic_route_score.py --data <D> --work work --ce_dir ce_b2 --ce_train ce_data/train.parquet,ce_data2/train.parquet \
    --cand phonetic_cand --base_matching v2h/matching_results.tsv --base_candidates v2h/candidate_pairs.tsv --out phonetic_score
python src/phonetic_route_build.py --data <D> --scored phonetic_score --base_matching v2h/matching_results.tsv \
    --base_candidates v2h/candidate_pairs.tsv --out phonetic_build
python src/samename_route_candidates.py --data <D> --work work --split train --out samename_cand
python src/samename_route_candidates.py --data <D> --work work --split test --out samename_cand
python src/samename_route_score.py --data <D> --work work --ce_dir ce_b2 --ce_train ce_data/train.parquet,ce_data2/train.parquet \
    --cand samename_cand --phonetic_cand phonetic_cand --phonetic_score phonetic_score --phonetic_build phonetic_build \
    --base_matching v2h/matching_results.tsv --base_candidates v2h/candidate_pairs.tsv --out samename_score
python src/phonetic_samename_merge.py --data <D> --phonetic phonetic_score/route_adds_test.parquet \
    --samename samename_score/route_adds_test_v2.parquet --base_matching v2h/matching_results.tsv \
    --claimed v4/matching_results.tsv --out phonetic_samename_adds.parquet
```

## 7. Final build of the submitted file (`avg_ce4_v17c`, public LB 0.989828, the file in `output/`)
Decode of the two runs with the cross-encoder ensemble (rounds 3, 4, 4b, Qwen; the French cross-encoder for France),
US/India thresholds 0.8, France 0.87, the France generator rules and the route files, then first match for empty US/India
S1 (blend >= 0.5); two more route steps are appended.
```
R=src/artifacts/routes
python src/final_build.py --data <D> --runs nr,nr10 \
  --ce nr/ce_test.parquet:1,nr10/cebase_ce_test.parquet:1,ce_base4/ce_test.parquet:1,ce_base4b/ce_test.parquet:1,qwen_ce_test.parquet:1 \
  --ce_fr nr/ce_test_france.parquet,nr10/ce_test_france_new.parquet \
  --rules A2P,A2F,A2N,HN_A,DP,E2,OOC,HNK --collapse_legal --first_min US=0.5,India=0.5 --first_skip_ea_ties \
  --extra_pairs $R/v11/native_adds_q90.parquet,$R/v11/phonetic_samename_adds.parquet,$R/v13/fr_inv_adds.parquet,$R/v13/fr_acr_adds.parquet,$R/v13/fr_fuzzy_street_adds.parquet,$R/v13/legal_tiebreak_adds.parquet,$R/v13/fr_inv_city_adds_f.parquet,$R/v13/fr_acr_city_adds.parquet,$R/v13/fr_inv_legal_adds.parquet,$R/v15/fr_acr_shared_adds.parquet,$R/v15/fr_thr85_safe_adds.parquet \
  --out v15b                                                   # 5,838,236 matches
BER_TEST_S1=<D>/test/test_source1.tsv python src/append_routes.py v15b v16 $R/v16/fr_unit_adds.parquet,$R/v16/fr_acr_city_shared_adds.parquet
python src/legal_tie.py --data <D> --claimed v16/pred.parquet --gate_run nr --out legal_tie_adds_v17c.parquet   # = $R/v17/legal_tie_adds_v17c.parquet (735 pairs)
BER_TEST_S1=<D>/test/test_source1.tsv python src/append_routes.py v16 v17c $R/v17/legal_tie_adds_v17c.parquet   # 5,839,939 matches = output/
```
`v17c/` then holds `matching_results.tsv` and `candidate_pairs.tsv` (= `output/`). `candidate_pairs.tsv` is the stage-1
output of both runs plus the pairs accepted by the routes: 4.43 pairs per S1, 1.31x the final matches.
Check with the challenge's validator: `python3 utils/validate_submission.py --matching output/matching_results.tsv
--candidate output/candidate_pairs.tsv --test-dir <D>/test --check-ids` -> PASS.

## Source layout (src/)
- `ber/normalize.py`, `ber/translit.py`: normalisation (accents, OCR digits, legal forms, street types EN/FR, house numbers)
  and the learned Indic->Latin dictionary
- `ber/blocking.py`, `ber/dense.py`: TF-IDF (sparse_dot_topn) and dense (exact blocked top-k on GPU) retrieval
- `ber/features.py`: pair / competition / decoy-signature / global features, word log-odds fitting
- `ber/pipeline.py`: stages (incl. the reverse-name route), candidate union + pruning, decoders, prior-shift thresholds
- `ber/io.py`, `ber/metric.py`: TSV reading / writing, macro F0.5
- `run.py` (stages norm, block, dense, namerev, train, test), `fit_translit.py` (Indic dictionary)
- `ce_data.py` + `ce_train.py` (cross-encoders), `llm_ce.py` (Qwen LoRA), `fit_pseudo_odds.py` (French decoy words)
- `sagemaker_pipeline.py`, `sagemaker_llm.py`: the AWS entry points of step 4 and the Qwen run
- `native_route.py` (India native-script route), `phonetic_route_*.py` + `samename_route_*.py` + `phonetic_samename_merge.py`
  (phonetic and same-name routes), `france_name_replaced.py`, `france_thr_safe.py`, `france_shared_addr.py` (France routes),
  `legal_tie.py` (legal-form tie-break): the route and rule files of step 6
- `final_build.py` (decode of the final build), `append_routes.py` (the last two route steps), `blend.py` (the single-run
  decode we used until 26 Sep, superseded by `final_build.py`)
- `artifacts/`: `indic_dict.json`, `odds_extra_france_iso.parquet`, `routes/` (step 6)
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
            z.write(f"{root}/artifacts/{v}/{f}.parquet", f"{base}/src/artifacts/routes/{v}/{ZIP_NAME.get(f, f)}.parquet")
    z.writestr(f"{base}/README.md", readme)
    z.writestr(f"{base}/requirements.txt", "\n".join(req) + "\n")
    if a.output:
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            z.write(os.path.join(a.output, f), f"output/{f}")
    if a.doc:
        z.write(a.doc, "Documentation_template.md")
print(a.out, [i.filename for i in zipfile.ZipFile(a.out).infolist()][:40])
