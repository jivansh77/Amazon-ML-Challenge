# Experiment log

This log feeds the methodology document. All numbers are measured on the full training set unless marked "dev". "Dev" means the 10% cluster sample: S1 entities together with their matches, plus 10% of the unmatched records.

## Blocking recall

Recall is the share of the 7,638,365 true (S1, S2/S3) pairs that land in the candidate set.

| Route | @5 | @10 | @20 | @30 | Time (Kaggle, 4 CPU / 2×T4) |
|---|---|---|---|---|---|
| Word TF-IDF on name core + address, df cap 1% | 0.829 | 0.919 | 0.943 | 0.953 | train 86 min, test 54 min |
| Dense `multilingual-e5-small` on "name \| address", exact top-k on GPU | 0.852 | 0.940 | 0.953 | 0.958 | ~2 h for train + test, including encoding 24M records |

**Union of both routes:**

| Candidates kept per route | Union recall | Candidates per S1 |
|---|---|---|
| TF-IDF 30 + dense 30 | 0.9831 | 53.2 |
| TF-IDF 30 + dense 20 | 0.9817 | 43.8 |
| TF-IDF 20 + dense 20 | 0.9795 | 34.2 |
| TF-IDF 15 + dense 15 | 0.9761 | 24.8 |
| TF-IDF 10 + dense 10 | 0.9697 | 15.5 |

### Reverse dense route: top S1s for each S2/S3 record (`ber-dense2`)

This route targets name-only S2/S3 records whose generic name ties with many lookalikes in the forward direction.

| Candidates | Recall | Pairs per S1 |
|---|---|---|
| Reverse top 1 alone | 0.957 | ~4.7 |
| Forward dense top 20 | 0.953 | 20 |
| Forward top 20 + reverse top 1 | 0.9671 | 20.3 |
| Forward top 20 + reverse top 2 | 0.9713 | 22.7 |
| Forward top 20 + reverse top 3 | 0.9739 | 25.8 |

**Three-route union on full train:**

| TF-IDF + dense + reverse | Recall |
|---|---|
| 30 + 30 + 0 | 0.9831 |
| 30 + 30 + 1 | 0.9857 |
| 30 + 30 + 2 | 0.9867 |
| 25 + 25 + 2 | 0.9859 |
| **20 + 20 + 2** | **0.9848** |
| 20 + 20 + 3 | 0.9857 |
| 15 + 15 + 2 | 0.9832 |

**First leaderboard point:** `ber-xgb-test` scored 0.9801 on validation and **0.9705 on the public LB**. The gap is about 0.01; the likely causes are France (no labels) and the denser test pool.

The first run was killed (out of memory) inside the recall report, after all files had been written. Kaggle keeps no outputs from failed runs, so the kernel wrapper now always exits cleanly.

### What the TF-IDF route misses (362k pairs)

The misses are 61% India. Among them:
- 24% have an empty S2/S3 address and a generic name that ties with 30 or more lookalikes.
- Some addresses consist entirely of individually common tokens. Address bigrams would fix this (+0.5% recall on dev, but 2.5× slower).
- Some names are websites without word breaks. A compact-name token fixes this.

### Dead ends

- **Rare-key join blocking.** Each record keeps its rarest tokens plus address bigrams, then records are hash-joined on those keys. It reached only 0.92 recall @30 on dev, and ran out of memory at full scale.
- **Char 3-gram TF-IDF on names.** 0.70 recall on dev, because generic names crowd it out.
- **Absolute df caps on TF-IDF (300 or 1000).** They destroy recall: city names and frequent tokens do matter.

## Matching model

Validation is on 10% of the model-training S1 entities, hashed by S1 id. Candidates come from the full S2/S3 pool, and singletons are included. The metric is macro F0.5.

| Run | Candidates | Model | Blocking recall | Oracle F0.5 | **Val F0.5** | US | India | Singletons | Matched | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| ber-s2gr2-train | TF-IDF 20 ∪ dense 20 ∪ **reverse dense 2** (recall 0.985) | Two-stage XGBoost + global competition features + house-number features, negative sampling 0.3 | 0.985 | – | **0.9837** (stage 1: 0.9814) | 0.9859 | 0.9804 | 0.9738 | 0.9843 | expected-F0.5 decoder |
| ber-s2-train | TF-IDF top 30 ∪ dense top 30 | **Two-stage XGBoost**: stage 1 on 30% of S1; stage-1 score context (rank/margin/sum per S1 and per candidate over all pairs) feeds stage 2, trained on a disjoint 30%; easy-negative sampling 0.3 | 0.9832 | 0.9947 | **0.9831** (stage 1 alone on the same rows: 0.9794) | 0.9856 | 0.9794 | 0.9741 | 0.9836 | expected-F0.5 decoder |
| ber-combo-train | TF-IDF top 30 ∪ dense top 30 | single-stage XGBoost + global competition features, negative sampling 0.5 | 0.9832 | 0.9949 | **0.9809** | 0.9833 | 0.9773 | 0.9839 | 0.9807 | thr 0.75 + exclusivity |
| ber-exp-feat | TF-IDF top 20 ∪ dense top 20 | LightGBM + house-number diffs + **global name/address competition features** (`g_mix_rank_c`, `g_mix_margin_c` become the top features) | 0.9795 | 0.9938 | **0.9801** | 0.9829 | 0.9760 | 0.9828 | 0.9799 | thr 0.70 + exclusivity |
| ber-exp-xgb | TF-IDF top 30 ∪ dense top 30 | **XGBoost on GPU** (lossguide, 255 leaves, eta 0.08), 63 features including the new house-number diffs, 30% of S1 (31.8M pairs), best iteration 1020, ~10 min on a T4 | 0.9832 | 0.9949 | **0.9799** | 0.9825 | 0.9761 | 0.9826 | 0.9798 | thr 0.75 + exclusivity |
| ber-exp-union | TF-IDF top 20 ∪ dense top 20 | same model, 30% of S1 (22.7M pairs) | 0.9795 | 0.9938 | **0.9772** | 0.980 | 0.973 | 0.976 | 0.977 | thr 0.70 + exclusivity (expected-F0.5 decoder: 0.9769) |
| ber-exp-tok | TF-IDF top 30 | LightGBM, 67 features, 30% of S1 (19.9M pairs), 1500 rounds | 0.953 | 0.984 | **0.9660** | 0.972 | 0.957 | 0.974 | 0.966 | thr 0.70 + exclusivity (expected-F0.5 decoder: 0.9658) |

**ber-exp-union:** submitted first to the leaderboard. Its test predictions give 94.3% of S1 at least one match. Its top features are the candidate-side dense-score competition features: `dense_score_margin_c` and `dense_score_rank_c`.

**ber-exp-tok test predictions:**
- 93.6% of S1 get at least one match.
- Matches per matched S1: France 3.0, India 3.37, US 3.5. France looks plausible despite having no labels.

**Top features by gain:**
- The candidate-side competition features dominate: `tok_score_margin_c`, `tok_score_rank_c`, `tok_score_gap_c`. They describe how this S1 compares with the other S1s competing for the same S2/S3 record.
- These are followed by the house-number features `num_tset` and `num_first_ratio`, then the name similarities.
