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

**Public leaderboard probes:**

| Upload | Public LB |
|---|---|
| ber-xgb-test (0.9801 on validation) | 0.9705 |
| ber-s2-test (0.9831 on validation) | 0.970 |
| same scores, thr 0.30 | 0.956 |
| same scores, thr 0.90 + exclusivity | 0.973 |
| ber-big-test + prior-corrected per-country thresholds (US .90 / India .85 / France .97), 4.35 candidates per S1 | **0.9757** |

Conclusion: on test we are precision-limited, and validation gains that come from recall do not transfer.

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
| **ber-decdup-train** | TF-IDF 15 ∪ dense 15 ∪ reverse 2, stage-1 filter (top 15, p ≥ 0.005) | as ber-dec + **decoy duplication** (near decoys within rank 5 get a virtual copy: 6.1M pairs; train AND validation see ~2× decoys, like test), 35% + 35% | 0.9829 | 0.9947 | **0.9871 on test-like validation** (stage 1: 0.9847) | 0.9890 | 0.9844 | **0.9951** | 0.9867 | thr 0.80 |
| **ber-dec-train** | TF-IDF 20 ∪ dense 20 ∪ reverse 2, stage-1 filter (top 15, p ≥ 0.005) | two-stage XGBoost + global features + **decoy-signature features** (word log-odds with FR map, signed house-number shift), 45% + 45%, negative sampling 0.2, no drop | 0.9849 | 0.9953 | **0.9862** (stage 1: 0.9839) | 0.9879 | 0.9836 | 0.9822 | 0.9864 | expected-F0.5 decoder with exclusivity |
| ber-big-train | TF-IDF 20 ∪ dense 20 ∪ reverse 2, then **stage-1 filter (top 15, p1 ≥ 0.005) → ~4.3 candidates per S1** into stage 2 | two-stage XGBoost + global features, 45% + 45% of S1, negative sampling 0.25, drop 19% | 0.9846 (after the filter) | 0.9952 | **0.9837** under density simulation (stage 1: 0.9811) | 0.9857 | 0.9807 | 0.9748 | 0.9842 | expected-F0.5 decoder |
| ber-tri-train | as dropA + **triangle consistency features** (candidate vs the S1's anchor match) | – | 0.985 | 0.9954 | 0.98302 (dropA: 0.98297), **no gain** | 0.9849 | 0.9802 | 0.9710 | 0.9837 | expected-F0.5 decoder |
| ber-dropA-train | same as s2gr2, **19% of S1 dropped** (test-like density, orphaned S2/S3) | two-stage XGBoost + global features | 0.985 | 0.9954 | **0.9830** under density simulation (stage 1: 0.9807) | 0.9850 | 0.9799 | 0.9736 | 0.9835 | expected-F0.5 decoder |
| ber-dropB-train | as dropA + stage-1 filter top-5 / p≥0.01 | – | 0.937 (the filter is too aggressive) | 0.9887 | 0.9779 | – | – | – | – | expected-F0.5 decoder |
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

## Model comparison on identical data (Colab T4, `ber-dump` features)

Setup: single-stage model, 15% of S1, density simulation, negative sampling 0.3; 3.95M training pairs and 0.98M validation pairs.

| Model | Best val F0.5 | Time |
|---|---|---|
| **XGBoost** (lossguide, 255 leaves) | **0.97986** @ thr 0.75 | 2 min |
| CatBoost (depth 8, lr 0.08, 6000 iterations, did not converge) | 0.97835 @ thr 0.70 | 9 min |
| Blend 0.9 / 0.1 | 0.97967 | – |
| Blend 0.8 / 0.2 | 0.97971 | – |
| Blend 0.5 / 0.5 | 0.97943 | – |

| Random forest (XGBoost RF mode on GPU, 300 trees, depth 16) | 0.96855 @ thr 0.90 | 2 min |
| Random forest (500 trees, depth 20) | 0.96927 @ thr 0.90 | 6 min |
| Blend XGB 0.9 / RF 0.1 | 0.97949 | – |
| Blend XGB 0.8 / RF 0.2 | 0.97933 | – |

Blending does not help. The errors are systematic (hard decoys), not variance. XGBoost is kept alone.

## Why validation does not transfer to the LB: decoy density (analysis on 25 Sep)

**Train's unmatched S2/S3 records are decoys, not orphans.**
- 72% of them are very close to an existing S1 (similarity > 0.92), against 91% for true matches.
- They are deliberately generated near-duplicates.

**Test has about twice as many decoys per S1.**
- Records per S1: 5.5–5.8 in test vs 4.68 in train. The share of S1s with 6 or more nearby records rises from 30% to about 50%.
- Confident predictions (p ≥ 0.97) stay at about 3.2–3.3 per S1, the same as validation.
- Borderline pairs (0.5–0.97) are 40–150% more frequent than on validation, so the precision of each band collapses on test.

**France is the most ambiguous country.**
- The median best-vs-second-best S1 margin is 0.035, against 0.071 for the US.
- 49% of France's S1s fall in the < 0.05 margin buckets, against 10% for the US.

**What does not explain the gap:**
- Re-weighting validation by the ambiguity mix brings the estimate to 0.981, still far above the LB.
- Weighting decoy false positives 3× does not reproduce the LB ordering.
- Multi-assigned records: 272 out of 5.9M.

**Prior-shift correction (`scripts/prior_thresholds.py`):**
- Take TP per S1 per band from validation (France uses the US/India average) and divide by the observed test pairs per S1 in that band. That gives an estimated test precision per band.
- A band is included when its precision is ≥ 0.78 (≈ F*/(1+β²)).
- Result on the full-recipe model: **US 0.93, India 0.90, France 0.98**.

## Decoy signature (train, best-S1 similarity ≥ 0.93)

| | Decoys | True matches |
|---|---|---|
| Same first house number | **10%** | **79%** |
| Same legal form | 49% | 74% |
| Address near-identical | 32% | 65% |

Decoys are "branches" of the S1 business: the same name plus an ADDED qualifier word, and the house number shifted upward by a small step.

**Learned word log-odds (held-out S1 slice):**
- Most decoy-like added words: holdings, care, clinic, east, west, north, south, valley, public, exports, ridge, overseas, health, metro, highland, summit, coastal, central, eastgate, uptown, downtown, midtown…
- Most match-like added words: doing, known, nee, as, dba, formerly, fka (alias phrases), plus generator typos (lnfrastructure, prviate, 6reat).
- France S2/S3 names over-represent développement, participations, associés, holding, distribution, groupe, services. These are the French qualifiers, so a hand-written EN→FR map transfers the scores.

**Runs:**
- `ber-dec-train`: best recipe + decoy features + stage-1 filter (top 15, p ≥ 0.005).
- `ber-decdup-train`: the same + decoy duplication (test-like density in both training and validation).

## Cross-encoder (`ber-ce`, Kaggle T4)

**Setup:**
- Model: `intfloat/multilingual-e5-small` (MIT) fine-tuned as a cross-encoder on "name | address" pairs.
- Training data: 2.23M hard pairs from training S1s outside the big model's validation set, 24% positives. Negatives are the candidates within rank 5, plus 35% of the rest.
- Training: 55-minute time cap (1.49M pairs seen), max length 128, lr 5e-5, fp16.
- The saved model (`ce_model`) scores any model's uncertain band (0.02 ≤ p < 0.998).

**On the big model's validation band (75k pairs):**

| Scorer | AUC |
|---|---|
| Stage-2 model | 0.9657 |
| Cross-encoder | 0.9559 |
| **Mean of the two** | **0.9768** |

The two are complementary.

**Blend on big's validation (logit average, w = weight of the stage-2 model):**

| w | Val F0.5 |
|---|---|
| 1.0 (big alone) | 0.9840 |
| 0.8 | 0.9866 |
| **0.6** | **0.9874** |
| 0.5 | 0.9871 |

Linear averaging is a little weaker (best 0.9862).

**big + CE test file** (`matching_results_big_ce_percountry.tsv`): per-country thresholds US 0.70 / India 0.80 / France 0.97, 3.33 matches per S1.

## dec + cross-encoder (`ber-cescore-dec`)

The saved cross-encoder scored the dec model's uncertain band:

| Band | Pairs |
|---|---|
| Validation | 74,473 |
| Test | 1,385,388 |

**AUC on the dec validation band:**

| Scorer | AUC |
|---|---|
| Stage-2 model | 0.9651 |
| Cross-encoder | 0.9383 |
| **Mean of the two** | **0.9708** |

**Blend on dec's validation.** The 1,335 validation S1s (1.39%) that the cross-encoder saw in training are excluded, so the numbers are clean.

| Blend | Best threshold F0.5 | F0.5 decoder |
|---|---|---|
| dec alone | 0.98651 | 0.98655 |
| linear, w = 0.6 | 0.98757 | 0.98718 |
| logit, w = 0.8 | 0.98839 | 0.98821 |
| **logit, w = 0.6** | **0.98862** (thr 0.8) | 0.98837 |
| logit, w = 0.5 | 0.98824 | 0.98753 |

**Test file** `matching_results_dec_ce_percountry.tsv`:
- Per-country thresholds: US 0.80 / India 0.80 / France 0.97.
- 5.75M pairs (3.32 per S1). Passes the official validator against `candidate_pairs_dec.tsv`.
- Agreement with other files (share of identical rows): 90.4% with big_percountry (LB 0.9757), 95.2% with dec_percountry.

## France is where the leaderboard gap is (26 Sep analysis)

Per-S1 profile of the dec model's stage-2 pairs:

| | Confident (p ≥ 0.998) | Band (0.02–0.998) | Mid (0.3–0.9) |
|---|---|---|---|
| Val US / India | 3.20 / 3.03 | 0.79 / 0.75 | 0.107 / 0.110 |
| Test US / India | 3.24 / 3.06 | 0.81 / 0.77 | 0.122 / 0.128 |
| **Test France** | **2.91** | **1.04** | **0.251** |

- US and India test look like validation, with only ~15% more mid-band pairs, spread evenly over S2 and S3.
- France has 2.3× the mid band.
- On the LB, moving France from thr 0.90 to 0.97 (and India 0.90 → 0.85) gained +0.0027. That means France's [0.90, 0.97) pairs were mostly wrong.

**Why France:** the decoy-word odds are learned on US/India labels. French qualifiers score 0 ("unknown word"), so French lookalike branches stay uncertain.

| Pairs where the candidate ADDS the word | Pairs | Confident + | Confident − | Uncertain |
|---|---|---|---|---|
| "france" | 36.5k | 2.6k | 24.7k | 9.1k |
| "sainte" | 990 | 0 | 630 | 360 |
| "lille" | 254 | 0 | 239 | 15 |

Other French decoy patterns: a category word swapped at the same address (Comite → Musique, Club → Amicale, Ecole → Lycee), a city name added, and "EI" added.

**Transductive fix (`fit_pseudo_odds.py`, `--odds_extra`):**
- Confident test predictions (blend ≥ 0.99 / ≤ 0.05) serve as labels, and word log-odds are fitted on near-duplicate names.
- Sanity check on US test against the training odds for the same words: extra-word correlation 0.69, sign agreement 81% for strong words. Pseudo < −3 implies train < −2 in 97% of words.
- Magnitudes are ~2× too large (regression slope 0.47), so the French scores are halved.
- Missing-word odds are unreliable (sign agreement 46%) and are not used.
- 101 new French extra-words. The most decoy-like: sainte, lille, ateliers, nantes, jean, ei, bordeaux, notre, departemental, pierre, energie, communale, marie, calais, residence, ehpad, musique, maison, sport, france (−2.1 after halving).
- Match-like words are generator typos: farmacie, clbu, teatre, uion.

`ber-dec-test2` re-scores test with these words.

**Cross-encoder self-training for France (`ber-ce-fr-train`):**
- Continues the CE from `ber-ce` with lr 2e-5 on 300k confident French test pairs (50/50) plus 290k US/India training pairs.
- It then scores France's band [0.003, 0.9995).

**Who is right when the stage-2 model and the CE disagree** (clean US/India validation band):

| Case | Pairs | Share that are true matches |
|---|---|---|
| p < 0.6 and CE > 0.99 | 944 | 84% |
| p < 0.3 and CE > 0.99 | 440 | 74% |
| p > 0.9 and CE < 0.3 | 1362 | 80% |
| Blend in [0.85, 0.97) | 5776 | 94.6% |

### Results of the France runs (26 Sep, early)

**`ber-dec-test2`** (test re-scored with the 101 French extra-words):

| Effect | Pairs |
|---|---|
| France pairs changed by > 0.01 | 86k |
| France pairs changed by > 0.2 | 17.6k |
| "+ France" pairs dropped by the stage-1 filter | 23k of 36.5k |

- Pure additions collapse, e.g. "club projets" → "club projets france": 0.986 → 0.016.
- The model is non-monotonic in the new score: substitutions ("ase culturelle" → "ase france") rose from 0.04 to 0.93.
- Band totals barely move.
- The new words also touch ~95k US/India pairs (club, sport, bar, residence… occur there). The probe files therefore keep the old US/India scores.

**`ber-ce-fr-train`** (cross-encoder continued on 300k confident French test pairs + 290k US/India training pairs, lr 2e-5, 4.6k steps):
- Validation-band AUC 0.9377, against 0.9383 before: no US/India damage.
- On France's band the mean CE drops from 0.53 to 0.32. 29% of pairs the old CE scored ≥ 0.99 fall below 0.5.
- What gets rejected: category-word swaps at the same address (Loisirs → Fetes, Club → Groupement), "Cie" → "Et Fils", same name at a different street with the same number, "+ France".
- Clear errors: abbreviation expansions such as "DV Ets" → "DV Établissements" (0.10).
- The pseudo-labels carry the stage-2 model's view (the confident-negative set includes pairs where it overrode the CE), so the adapted CE mostly learns that view for France.

**France pair budget** (pairs per S1 per band, after exclusivity):
- US/India test pairs per band ≈ validation true matches per band. Their thresholds are consistent.
- France has 3.05–3.09 pairs per S1 in the top band vs 3.33, and about 2× the pairs in every band from 0.7 to 0.98.

**Earlier LB pair:** big thr 0.90 → per-country (FR 0.97, IN 0.85) gained +0.0027. India's part is worth ~+0.0001, so the removed France [0.90, 0.97) pairs were only ~20% true matches. France probably has fewer matches per S1, and its lower bands are mostly decoys, so France thresholds stay high.

**Probe files** (US/India identical to `dec_ce_percountry`; only France differs):

| File | France rows changed vs dec_ce |
|---|---|
| `matching_results_dec_ce_frwords.tsv` (French words, France thr 0.97) | 1.9% |
| `matching_results_dec_ce_frwords_frce.tsv` (+ French CE, France thr 0.97) | 5.4% |

## decdup is broken on test (26 Sep): do not upload any decdup file

The duplication added virtual copies only to records that match no S1 (label-derived). In training and validation, every nearby decoy therefore had an exact twin. The model learned the shortcut "no twin → real match".

On validation the shortcut still works, because validation also has the twins:
- On the 48k validation S1 with no virtual copies, decdup beats dec (0.9888 vs 0.9876; +CE 0.9895 vs 0.9893).
- A dec + decdup average reaches 0.9900.

Those S1s are the easy ones, with no decoy nearby. Test has no twins, so decdup accepts decoys:

| | dec | decdup |
|---|---|---|
| Test S1 with best p ≥ 0.8 | 94.1% | **99.1%** |
| Test S1 predicted empty | 5.9% | **0.9%** |
| Training S1 with no match | 5.6% | 5.6% |

- On 35,344 test S1s, dec's best candidate is below 0.3 while decdup's is at least 0.9 (15.6k US, 13.9k India, 5.9k France). The reverse happens 13 times.
- Those S1s are almost certainly singletons, which score 0 when anything is predicted: about −0.02 on the leaderboard.
- The earlier symptoms were the prior-shift thresholds at 0.99 in every country and 3.53 matches per S1 at thr 0.80.

**Lesson:** a density simulation must not be label-conditional. Dropping S1s (dropA) was the leak-free variant, and it did not help.

## France threshold: two estimates disagree, so the LB decides

**Label-shift EM (Saerens):**
- Isotonic calibration of the blended score on validation (after exclusivity; positive share 0.856).
- Then per test country, EM on the pair prior, and the threshold where the corrected probability reaches 0.78.

| Variant | US | India | France |
|---|---|---|---|
| dec_ce | prior 0.90, thr 0.78 | prior 0.92, thr 0.74 | prior 0.82, **thr 0.84** |
| + French words | 0.78 | 0.74 | 0.80 |
| + French words + French CE | 0.78 | 0.74 | 0.86 |

**Per-band count estimate (`prior_thresholds`):** France 0.97–0.98.

The EM assumes France's score distribution given the label equals validation's. The band count assumes France has as many true matches per S1 per band as US/India. Neither holds for sure.

**Probes on dec_ce, France threshold only (US/India unchanged):**

| File | France threshold |
|---|---|
| `dec_ce_fr99` | 0.99 |
| `dec_ce_percountry` | 0.97 |
| `dec_ce_fr93` | 0.93 |
| `dec_ce_fr85` | 0.85 |

Each LB difference is the net value of one France band.

**Upload plan, 26 Sep 4pm:**
1. dec_ce_percountry
2. dec_ce_frwords_frce
3. dec_ce_fr99
4. dec_ce_fr93
5. dec_ce_fr85

dec_ce_frwords (words only) is kept for 27 Sep if needed.

## French words, v2: calibrated, not halved (26 Sep)

On validation the model's response to the learned score of an added word is calibrated and sharply non-linear. For pairs whose candidate adds exactly one word:

| Word score | Pairs | True matches |
|---|---|---|
| Unknown word (mostly generator typos) | 67k | 88% |
| (−1, 0] | 29.5k | 76% |
| (−2, −1] | 5.2k | 66% |
| (−3, −2] | 577 | 55% |
| **(−4, −3]** | 6.3k | **3%** |
| ≤ −4 | 0.7k | 4–11% |

v1 halved the pseudo scores, which put "france" at −2.1, in the ambiguous zone. In confident French pairs "+ france" is ~10% match.

**v2 (`--calib isotonic`):**
- Run the same pseudo-label recipe on the labelled countries' test pairs (US + India, 5.55M confident pairs).
- Compare with their training odds word by word (437 words) and fit a monotone map.

| Pseudo score | −8 | −6 | −4 | −3 | −2 | −1 | 0 | +1 |
|---|---|---|---|---|---|---|---|---|
| Training scale | −5.16 | −5.07 | −3.45 | −2.56 | −2.35 | −1.30 | −0.79 | +1.07 |

After calibration:
- "france", "club", "amicale", "musique" → −3.45.
- "sainte", "lille", "nantes", "ateliers", "ei", "bordeaux" → −5.07.
- Typos such as "farmacie" → +1.11.

`ber-dec-test3` re-scores test with v2 (CPU kernel).
