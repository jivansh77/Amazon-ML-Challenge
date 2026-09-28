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

**`ber-dec-test3`** (French words v2, calibrated):

| File | Words | CE for France | France rows changed vs dec_ce | France S1 predicted empty |
|---|---|---|---|---|
| dec_ce (baseline) | – | original | – | 6.57% |
| `dec_ce_frv2` | v2 | original | 2.95% | 6.65% |
| `dec_ce_frv2_frce` | v2 | French | 6.40% | 6.79% |

- US/India are identical in all three.
- Candidate file: `candidate_pairs_dec_frv2.tsv` (4.13 per S1).
- Both files replace the v1 France files in the upload plan.

## Swapped-slice twin (`ber-decswap-train`, `--swap_ab`): no gain

- Stage 1 trained on dec's stage-2 slice and stage 2 on dec's stage-1 slice.
- The first run ran out of memory while scoring all training pairs with stage 1; `--chunk_s1 90000` fixed it.
- Its own validation: 0.9859. Averaged with dec on dec's 94k clean validation S1:

| | No CE | + CE (w 0.7) |
|---|---|---|
| dec alone | 0.98651 | 0.98873 |
| Logit average (0.3–0.7) | 0.98663 | 0.98880 |

The gain is ~+0.0001, within noise. The stage-2 model is saturated for this feature set, so the twin is not taken to test.

## Bigger cross-encoder: multilingual-e5-base (`ber-ce-base`)

- Training: 150-minute cap on a T4 at 193 pairs/s (1.74M pairs seen), lr 3e-5, same pairs as the small CE.
- Scoring: validation band in 2 min, test band (1.39M) in 39 min.

| AUC on dec's validation band | Stage-2 model | CE | Mean of the two |
|---|---|---|---|
| Small CE (e5-small) | 0.9651 | 0.9383 | 0.9708 |
| **Base CE (e5-base)** | 0.9651 | **0.9453** | **0.9730** |

**Blend F0.5 on clean dec validation** (CE-training S1 excluded):

| Blend | Val F0.5 |
|---|---|
| Model alone | 0.98651 |
| + small CE (w 0.6 / 0.7) | 0.98862 / 0.98873 |
| **+ base CE (w 0.6, thr 0.8)** | **0.98905** |
| Small + base three-way | ≤ 0.98902 |

The optimum is flat for w in 0.6–0.65 and thresholds 0.75–0.8.

**New probe set (all with the base CE; US/India identical across the five):**
- `dec_cebase`, `dec_cebase_fr99`, `dec_cebase_fr93`, `dec_cebase_fr85`: the base CE for France too.
- `dec_cebase_frv2_frce`: France uses v2 words + the French-adapted small CE.

`dec_cebase` vs `dec_ce_percountry`: 98.3% of rows are identical.

## LB 26 Sep

| Upload | Public LB |
|---|---|
| `dec_cebase` (dec + e5-base CE, US/IN 0.80, FR 0.97) | **0.982855** |
| Previous best (big_percountry) | 0.975703 |

The gain is +0.0072. If US/India score their validation 0.989 on test, France is at about 0.92, so it is still the weak spot.
| `dec_cebase_frv2_frce` (France: v2 words + French CE, FR 0.97) | **0.983051** (+0.0002 vs baseline, France only) |

## Unseen-country stand-in: train on US only, validate on India (`ber-usonly-train`)

| India validation, 37,437 S1 | Best F0.5 | Best threshold |
|---|---|---|
| dec (trained with India labels) | 0.984 | 0.70 |
| **US-only model** | **0.936** | 0.85–0.90 (flat 0.80–0.93) |

US itself barely moves (0.9880 → 0.9876).

- An unseen country loses ~5 points, consistent with France's implied LB F0.5 of ~0.92. Its best threshold also rises, as the France probes showed.
- Errors at the best threshold: FP 5.6k vs 0.6k (9×), FN 10.7k vs 5.0k (2×).
- The band is huge: 4.5 pairs per S1 in 0.02–0.998, against 0.75 when India is seen. India is a harsher stand-in than France (Indic scripts).
- Adding the base CE (which did see India, so this is optimistic; only 16% of the band has CE scores): 0.936 → 0.951 at w 0.6 and 0.952 at w 0.4. More CE weight helps an unseen country.

Next: `ber-usonly-st` tests `--self_train`, i.e. pseudo-labels on India from the US-only stage 2 and a refit.

### Self-training on the stand-in (`ber-usonly-st`): no gain

- Pseudo-labels on 715k India S1 outside validation: 2.46M pairs, 97.8% accurate against the hidden labels.
- Stage 2 was refit on US labels plus India pseudo-labels.

| | Best F0.5 |
|---|---|
| India, base | 0.9376 (thr 0.9) |
| India, self-trained | 0.9358 (thr 0.95) |
| US | unchanged |

Confident pseudo-labels teach nothing about the uncertain pairs (confirmation bias), so `ber-selftrain-fr` is not used for France.

**Next:** a US-only cross-encoder (`ber-ceus-train`, e5-small) scores the stand-in's validation band (442k pairs). This gives an honest measure of how much a CE, and which blend weight, helps an unseen country.

### Honest CE on the stand-in (`ber-ceus-train`: e5-small trained on US pairs only, 2.47M pairs)

India validation band (442k pairs of the US-only model), excluding the CE's training S1:

| | Model AUC | CE AUC |
|---|---|---|
| India | 0.956 | 0.940 |
| US | 0.997 | 0.995 |

| Unseen India, blend weight w (model) | Best F0.5 | At threshold |
|---|---|---|
| Model alone | 0.9332 | 0.9 |
| **w 0.7** | **0.9565** | 0.7 |
| w 0.6 | 0.9527 | 0.6 |
| w 0.5 | 0.9453 | – |

For US, w 0.6 is best (0.9898). The CE is the biggest lever for an unseen country (+2.3 points).

**Gain per added pair vs its precision** (labelled stand-in, w 0.6):

| Band | Precision | Gain per added pair |
|---|---|---|
| [0.93, 0.97) | 0.959 | +0.124 |
| [0.90, 0.93) | 0.942 | +0.114 |
| [0.85, 0.90) | 0.916 | +0.102 |
| [0.80, 0.85) | 0.885 | +0.085 |
| [0.70, 0.80) | 0.815 | +0.058 |
| [0.60, 0.70) | 0.699 | +0.008 |

Break-even is ~0.68–0.70.

**What this means for France:**
- The fr93 LB gain per added pair was +0.0175 (16,372 pairs). On this scale France's [0.93, 0.97) band is only ~0.72–0.80 precise.
- France is more decided (4× fewer mid-band pairs than the stand-in) but less precise at a given score.
- [0.90, 0.93) is therefore at or below break-even: keep France at 0.93.

## Base CE continued on fresh S1 (`ber-ce-base2`)

- Setup: 2.02M new pairs from 160k training S1 never seen by any CE, lr 2e-5.
- Validation-band AUC: 0.9545 (base 0.9453).
- Clean-validation blend: **0.98934** at w 0.6 / thr 0.8, and 0.98942 at thr 0.75. Base was 0.98905.

New safe upload `dec_cebase2_frv2_frce_fr93`:
- France is identical to `dec_cebase_frv2_frce_fr93` (calibrated words + French CE, 0.93).
- US/India use base2; 99.1% of their rows are unchanged.

`ber-cefr-base2` adapts base2 to France (pseudo pairs + 5% fresh pairs, 40 min).

### French adaptation of base2 (`ber-cefr-base2`)

- Validation-band AUC falls from 0.9545 to 0.9504, the usual adaptation cost on US/India.
- On France its bands nearly match the small French CE. At 0.93 the kept pairs differ by only ~6.5k of 800k (0.8%).
- Expected LB effect is about ±0.0001, so France keeps the LB-tested small French CE.

## Why France scores lower: compressed embedding neighbourhoods

Dense top-6 neighbours per S1 (test), median gaps in dense score:

| | rank 1 − rank 4 | rank 4 − rank 6 |
|---|---|---|
| US | 0.031 | 0.023 |
| India | 0.033 | 0.014 |
| **France** | 0.028 | **0.009** |

Training has 0.041 / 0.020 (US) and 0.039 / 0.011 (India). French generic names ("Lille Club", "Lille Amicale") crowd together in e5 space.

The strongest features are dense/route margins (rdense_score_margin_c, g_mix_margin_c), so a French true match looks less certain. Part of this is a scale shift, not real ambiguity.

**Tests on the stand-in (CPU kernels):**
- `ber-us-cnorm`: `--country_norm`, route scores as within-country quantiles before the margins are computed.
- `ber-us-nodense`: `--drop_feats dense`, no dense-derived features.

Baseline for unseen India with the model alone: 0.9332 (thr 0.9).

### Stand-in: no dense features (`ber-us-nodense`)

| Unseen India | Model alone | + US-only CE (w 0.7) |
|---|---|---|
| Base | 0.9332 | 0.9565 |
| No dense features | 0.9350 | 0.9553 |

US is unchanged. Rejected.

### Stand-in: per-country quantile route scores (`ber-us-cnorm`)

| Unseen India | Model alone | + US-only CE (w 0.7) |
|---|---|---|
| Base | 0.9332 | 0.9565 |
| Country-normalised | 0.9351 | 0.9539 |

US is unchanged. Rejected: the CE already carries the transferable signal.

**AWS (26 Sep):** SageMaker and EC2 GPU quotas are 0 in all 7 regions checked. Increase requests (1 each, us-east-1) are PENDING for ml.g6e.xlarge, ml.g6e.12xlarge, ml.g5.2xlarge, ml.g5.12xlarge and ml.p4d.24xlarge training jobs. `scripts/llm_ce.py` (LoRA Qwen2.5 pair classifier) is ready for them.

## 26 Sep: importance weighting, CE round 3, LLM cross-encoder on Colab A100
- **Importance weighting (ber-us-iw)** on the unseen-India stand-in: model 0.9337 (base 0.9332), with the US-only CE 0.9525 (base 0.9565). Rejected.
- **CE round 3 (e5-base continued on ce-data3)**: band AUC 0.9567 (round 2: 0.9544); clean val F0.5 with w=0.6: 0.98952 vs 0.98942 for round 2. Blending rounds 2 and 3 does not help (0.9893).
- **LLM cross-encoder**: Qwen2.5-7B-Instruct + LoRA r16 (bf16, sequence-classification head) on a Colab Pro A100 40GB,
  trained on the round-2 CE pairs (36 pairs/s, 150 min cap), scores the dec val band and the test band (dec ∪ France v2).
  `scripts/llm_ce.py` now scores in text-length order (little padding) and saves the adapter every 1500 steps.
- **LB: `dec_cebase3_frv2_frce_fr90` = 0.984427** (previous best 0.983051). US/India use CE round 3 (val estimate about +0.0004);
  France threshold 0.97 → 0.9 changed about 16.4k French S1, mostly by adding a second match. Implied precision of those
  additions: about 0.85–0.9 (break-even about 0.73). The stand-in calibration (0.72–0.80) was too pessimistic.
- **LB: `dec_cebase3_frv2_frce_fr87` = 0.984515** (fr90: 0.984427). 3,092 French rows changed; implied precision of the
  0.87–0.9 additions about 0.80 (0.9–0.97: about 0.85–0.9; break-even 0.73). Further lowering is near break-even.
- **Qwen2.5-7B LoRA, 217k pairs (Colab A100, 100 min):** val band AUC 0.930 (e5-base round 3: 0.957); every blend
  lowers clean val (best 0.98940 vs 0.98952). Too little training; rerun on all 2.2M pairs on AWS 4x L40S.
- **Qwen test-set cut** (score only pairs with model+e5 blend in [0.05, 0.995)): identical val F0.5 to scoring all.

## Reverse-name route for records without an address (AWS `ml.g5.12xlarge`, full pipeline)
- Validation misses: 72% of the remaining F0.5 loss is "S1 with some true matches missing"; ~half of the missed
  true pairs were never candidates, and 64% of those are S2/S3 records with an EMPTY address (3.3% of records).
- `namerev` stage: for every S2/S3 record without an address, top-5 S1 by char 3-gram TF-IDF on the name
  (same country); offline recall of those misses: 53% at 5, 58% at 10. Route pairs keep full context features
  (the Kaggle run with them was OOM-killed at 82M pairs; the AWS machine has 192 GB).
- Blocking recall 0.9849 -> **0.9888**, oracle F0.5 0.9953 -> **0.9964**.
- Clean val (94,236 common S1): model alone 0.98649 -> **0.98696**; + e5-base CE round 3 (w 0.6)
  0.98950 -> **0.99003**, with CE scores on only 73% of the new band so far.
- Full CE coverage (AWS job's e5-base round-3 scores on the new band): clean val **0.99007** (old 0.98950, +0.00057).
- New upload `nr_cebase3_frce_fr87` (+ `candidate_pairs_nr.tsv`, 4.26 per S1): same recipe as the fr87 best
  (w 0.6, US/India 0.8, France 0.87 with calibrated French words + French CE); ~48k S1 differ, mostly added matches.
- The AWS job's last step (French CE) failed on a tokenizer saved by transformers 5.0 vs 4.57.1 in the job; the
  51,936 new French band pairs were scored locally on CPU instead.
- **LB: `nr_cebase3_frce_fr87` = 0.98553** (previous best 0.984515, +0.00102). The reverse-name route is the
  largest single LB gain since the cross-encoder; the LB gain exceeds the val gain (+0.00057).

## France generator rules (Kavya's label-free finding, 27 Sep)
- France true matches change the house number ~1% (US 12%, India 24%); French noise suffixes (fils, groupe,
  developpement, associes) were systematically rejected (train word odds call them decoys, -4 to -7).
- **LB: `nr_fr87_kv_HN_A` = 0.986665** (+0.00114 over 0.98553): +25,063 suffix pairs with the house number kept,
  -3,943 claimed pairs with a changed number and blend < 0.998 (faithful port of her `sh-hn` kernel).
- After the fix France has 3.21 predicted matches per S1 vs US 3.38 / India 3.35 (generator is country-invariant):
  ~42k matches still missing. The largest unclaimed class keeps the house number 89-99% (like matches) with
  organisation words (club, comite, amicale, ecole, union...); in labelled US/India val, organisation words added at
  the same number are matches ~100% (association, society, council, federation...), pure decoy words
  (enterprises, trading, ventures) are 0% even with the number kept.
- Probes: `kv_HN_A_C1` (+23,829 pairs, 46 organisation words, France -> 3.30/S1), `kv_HN_A_C2france` (+5,424 '+france').
- **LB: `kv_HN_A_C1` = 0.983453** (-0.0032): the organisation-word adds are decoys in France (category swap at the
  same number). Removing the 1,319 already-claimed category-swap pairs is the consistent follow-up (`kv_D`,
  est. +0.00015, not submitted).

## France-supervised cross-encoder (AWS `ml.g5.2xlarge`, e5-large, US/India + LB-confirmed France classes)
- Trained on 693,557 pairs; France pseudo-val = 25,144 LB-confirmed suffix matches + 24,288 category-swap decoys,
  with the words groupe / club / ecole / amicale held out.
- AUC on training words 1.000, on **held-out words 0.069** (inverted, like the self-trained French CE 0.059);
  US/India val band 0.9481 vs e5-base round 3 0.9526. It only memorises the word lists the rules already
  apply. Rejected.

## France noise profile (edit signatures, label-free)
- 35 noise ops (legal form kept/dropped/added/swapped, name typo/abbreviation/drop/add/reorder, case, junk,
  empty address, house number, street words, address components) on 380k US/India val pairs and 932k France
  candidates; France confident rate bias-corrected by the US/India confident/true ratio.
- Apparent France differences (name typos 1.1% vs 5.6%, empty address 0.3% vs 4%) are the model's own bias:
  the claimed-band France typo pairs are plain OCR-noise matches ("Loisrs", "Shotkan"). Legal-form swap (0.09%
  vs 1.7-19%) touches only ~380 claimed-band pairs. No new rule.
- Empty-address records whose name equals 2+ S1 names are true matches 98.5% in train but claimed ~0% in test
  (every country): chains with orphan copies (test has 5.76 records per S1 vs 4.67 in train), undecidable;
  matches per S1 per source are spread (1+1 12%, 1+2 11%, 2+1 10%...), so counts cannot disambiguate.

## Synthetic French cross-encoder (AWS `ml.g5.2xlarge`, e5-base round 3 continued)
- 560k pairs: 300k US/India + 260k France (80k real confident matches, 60k suffix adds with the number kept,
  50k category swaps, 20k decoy words +/- number shift, 20k number shifts, 30k real low-score pairs). Words never
  added in training: groupe, associes, club, ecole, amicale, comite, amis, sportive.
- France pseudo-val: trained words AUC 1.000; held-out: associes accepted 72.5% (AUC 0.954 vs the held-out
  category swaps), held-out category swaps rejected (<=9%), **groupe rejected (1.3%)** (read as the trained decoy
  word "groupement"); held-out AUC overall 0.516. US/India val band AUC 0.9509 (base3 0.9527); blend with base3
  0.98999 vs 0.98996 (noise).
- As the France CE instead of the French CE: +12,325 / -10,830 France pairs; the adds are the suffix adds the rules
  already make, the removals are 5,507 empty-address pairs (a bias from the real low-score negatives; empty-address
  records are true matches 97.7% in train), 3,005 word swaps (category swaps = kv_D, plus unverified decoy-word
  assumptions) and 1,694 changed numbers (= HN_A). Nothing validated beyond the rules. Rejected.

## Qwen LoRA runs, 27 Sep
- Colab A100 run #2 (new-route bands, lr 1e-4, bs 16, resumed from the 313k-pair adapter) **diverged** at step
  ~34,600 (01:15 UTC, ~550k pairs): loss 0.07 -> 0.56 (constant prediction) and stayed there; the 10-minute syncs
  overwrote the last good checkpoint. Stopped (42 compute units left) to keep the A100 for scoring.
- AWS `ml.g5.12xlarge` run g5b (old-route bands, lr 1e-4, 4 x bs 6) is healthy (loss 0.02-0.03 at step 30,800);
  training cap 08:30 UTC, then it scores the old-route val band + test cut; new-route pairs need a rescore with its adapter.

## Final build (`scripts/final_build.py`)
- Reproduces `nr_cebase3_frce_fr87` (0.98553) and `kv_HN_A` (0.986665) / `kv_D` exactly (0 pair differences).
- Top-5 + top-10 route average (mean p over the runs that scored a pair), e5-base round-3 CE on the union band:
  clean val (94,368 S1) 0.98994 -> **0.99007** at the leaderboard thresholds (0.75: 0.98996 -> 0.99010).
- Candidate `avg_D` (+ French CE for 31,857 new France band pairs, A2 + HN_A + D): validator PASS, 11,322 S1
  differ from `kv_HN_A`, 4.41 candidates per S1 (was 4.26).
- e5-base round 4 (Kaggle P100, new-route bands): `base4` (base3 + ce4 data) val-band AUC 0.9542, `base4b`
  (base3b + ce3 data) 0.9550 (base3 0.9527). Clean val at 0.8, route average: base3 0.99011, base4 0.99018,
  **base3 + base4 + base4b (equal logit average) 0.99019** (top-5 route + base3 = 0.98998, i.e. +0.00021).
- Candidate `avg_ce4_D` (route average + 3-CE ensemble + French CE + A2/HN_A/D): validator PASS; vs `kv_HN_A`
  12,600 S1 differ; vs `avg_D` only US/India change (3,181 S1).

## France empty-address records (rule E, 27 Sep)
- Records without an address whose name clearly points to one S1 (token-sort margin >= 15 over the 2nd candidate):
  claimed 84-97% in US/India but 55-94% in France. Unclaimed France ones with the same core words have stage-2 p
  median 0.973 (75% >= 0.87); the French CE pulls them under the threshold. A2 never covers them (it needs a
  house number).
- Rule E (France only): unclaimed empty-address record, clear best S1, same core words and stage-2 p >= 0.87, or a
  French noise suffix added (nd <= 1, na = 1). Adds 1,342 + 498 pairs (sampled pairs all look like matches).
  Candidate `avg_ce4_DE`. Expected about +0.0001 (not LB-tested).
- Synthetic CE v2 (`yoddhas-ce-syn2-fr-0927-0433`): all words, decoy words only with a number shift, no
  Compagnie, +40k empty-address matches and +20k empty-address category swaps, real low-score negatives only with
  an address.
- Synthetic CE v2 as the France CE (with rules A2/HN_A/D/E after): +14,612 empty-address France pairs (France would
  claim ~81% of its empty-address records vs 59% in the US: over-claiming chains/orphans), -2,113 multi-word
  changes; US/India val band AUC 0.9517 (base3 0.9527). Rejected.
- Kavya's synthetic CE (`kavyachetwani/syn-fr-scores`, 341k of our 480k France CE pairs) as the France CE: +3,760 /
  -7,606 France pairs; removes clear matches (1,543 same-core pairs such as "Pharmacie Sainte" / "Pharmacie
  Sainte SCI" at the same address, 1,067 OCR-typo pairs) and ~1,400 more changed-number pairs than HN_A; its own
  US/India val is 0.98862 vs 0.98942. Rejected.

## Label-free France census (27 Sep morning)
- Candidate coverage per S1: France 3.91 records vs US 3.84 / India 3.73 (no blocking gap).
- Unclaimed in-candidate records per S1: France 0.70 vs US 0.54; the excess is category swaps (+0.098),
  "+france" (+0.025) and different names (+0.048): known decoy classes, not missed matches.
- 5,672 unclaimed France records with the same core name and number are generic "City + Category" names on a
  different street (different businesses); claimed France same-name pairs almost all share the street.
- The current blend's val and test band densities now match for US/India (ratio ~1.0 in every band >= 0.7),
  so US/India likely score their val (~0.990) on the LB and the gap is France.
- Generator "rename" op: 1.6% of US/India true matches get an invented single-token name at the same address.
  France has the same rate of invented-name + same-number candidates (0.077 per S1, US 0.077) but claims 71%
  of those records vs 91% (US). Rule "add unclaimed invented-name, same number, same street" is only 19% precise
  on US/India val (the generator also makes invented-name decoys at the same address): val -0.00215. Not used.
- Twin test (is a record with added word w accompanied by another record with the same w on the same S1?):
  France category decoys 0.2%, suffix matches 3.4%, '+france' 3.8% - but on US/India val labels the signal does not
  hold (decoy word "enterprises": 2% matches, 36% twins; corr(match rate, twin rate) over words -0.30). Not used.
- French vocabulary the normaliser mishandles (legal form "EI" not in LEGAL, "Ste" -> suite, "Dr" -> drive): rare
  (EI in 1.6% of France S1 names, Sainte/Docteur ~0.5% of addresses) and their S1s have the usual claims per S1
  (EI 3.26, Docteur 3.19, Sainte 3.19 vs 3.21 overall). No recall gap; not worth a pipeline rerun.
- "+country" word, with labels: among 3,059,843 India true matches, the record NEVER adds "india" to its S1's
  core name (0). 'S1 name + india' records do exist (35k), and the matched ones belong to a different S1 whose name
  already contains India. So the generator's match noise never adds the country word: France '+france' records
  (6,302 same-number single adds) are not matches of the S1 without "France"; the model rejects them (0.9%
  claimed) and D drops the claimed ones. Settled; no change.

## France suffix operation, position signature and number perturbation (27 Sep, v2 rules)
- **The suffix operation.** In US/India labels, "drop one S1 word + add Center/Services/Service/Partners" at the same house
  number is 97-99% true matches, and the added word is appended AFTER the legal form ("Great Indo Infra Pvt Ltd" -> "Great Indo
  Pvt Ltd Services"; 88-92% after the legal form, the rest reordered). The same operation exists in France with a French list:
  fils, groupe, associes, developpement (the A2 words) and also **france, services, cie**. Position of the added word among
  France same-number one-word swaps (share after the legal form): fils 0.995, associes 0.996, developpement 0.89, france 0.88,
  groupe 0.84, services 0.84, cie 0.79; when before the legal form, these words sit at the very front ("France Mamzelles SAS";
  france 99.6%, developpement 100%). Category swaps are **in place** (club/comite/amicale/ecole 0.13-0.16 after, ~1% front) -
  and so are **centre, compagnie, service, federation** (0.10-0.15), which the model claims at 39-84% because Center/Service
  are match words in the US. "+france" swaps: 4,846 same-number pairs (like groupe 4,317, associes 4,090), same dropped-word
  profile as the A2 words, "France" essentially never in the middle (19 of 6.3k); the old India argument does not apply (no
  "+india" record exists in train at all, and the US/India suffix list is Center/Services/Service/Partners).
- **Out-of-candidate suffix records.** 7.4k France records with the suffix operation (same number + street + city as exactly
  one S1) are in NO candidate list: the stage-1 filter drops them because the model reads the French suffixes as decoy words,
  so A2 could never reach them.
- **Number perturbation vs decoy shift.** US/India labels, same core name + same street + changed first number: with the legal
  form added or swapped it is a decoy (US 11-28% match: "Patel Seafood" -> "Patel Seafood Co | 120 -> 129"), with the legal
  form kept/dropped/absent it is a match (US 97%, India 98%). Decoys shift the number UP by 1-20 (US 85%); matches have a
  smaller number (79%) or >100. A record number of "1" is a match 97.7-100%. France follows the same decoy profile (33k
  legal-changed pairs: 90% shifted up by 1-20, claimed 0%), and an extra "France" (S1 already "(France)") behaves like a legal
  qualifier ("In Fetes (France) SARL | 10" -> "In Fetes (France) France SARL | 11"). HN_A (LB-tested only together with A2)
  dropped ~1.7k keep-qualifier pairs with the match profile; the kept-qualifier + (smaller or >100) class is HNK.
- A second France decoy operation: category -> Centre/Compagnie/Federation/Service with a shifted number ("Crapahute Club SAS |
  4" -> "Crapahute Compagnie SAS | 7"), like India's "South Consultants | 6" -> "South Partnership | 8".
- Checked, no rule: legal-form swaps with same name and number (2,848; 2,698 on a different street), misattribution between
  same-name S1s (0), unclaimed same-core same-number records (97% on a different street), empty-address records (12.7k of the
  19.6k unclaimed tie between 2+ same-name S1s), the 0.80-0.87 France band after DP (mixed: empty-address, invented names,
  swaps, acronyms).
- Matches per S1 (train truth, identical for US and India): mean 3.46, P(1) 5.4%, P(2) 17.0%. Test predictions: US 3.38,
  France 3.21 (kv_HN_A) -> 3.25 (v2); France's excess of k=1/k=2 S1s is the recall gap the v2 adds target.

**v2 rules in `scripts/final_build.py`** (France only; A2/HN_A/D/E unchanged, the LB file reproduces with 0 pair differences):
- A2F: +France/Services/Cie suffix adds (unclaimed record, best S1 by name + address, same number, appended/front/last).
- DP: drop claimed in-place category swaps over the derived French category vocabulary (dropped-word count >= 40, 164 words),
  not suffix words, not abbreviations/typos; replaces D (D also dropped the "+france" suffix records).
- E2: E with the v2 suffix words. OOC: out-of-candidate suffix records, added to the candidate file too (legal form not
  swapped, all address numbers equal, same city, unique S1). HNK: see above.

| Build (avg_ce4 = top-5 + top-10 route average, CE base3 + base4 + base4b, French CE incl. 31,857 new band pairs) | France pairs | vs kv_HN_A |
|---|---|---|
| avg_ce4_DE (previous candidate: A2, HN_A, D, E) | 832,962 | France +3,171 / -2,728 |
| **avg_ce4_v2** (A2 25,833, A2F 5,764, HN_A -4,016, DP -2,559, E2 2,096, OOC 6,760) | 844,442 | France +15,873 / -3,950 |
| **avg_ce4_v2h** (+ HNK 2,853: 652 number->1, 1,158 >+100, 1,043 smaller) | 847,295 | **LB 0.98809** |

US/India are identical in all three (avg_ce4: all-val F0.5 0.98997 -> 0.99010). Validator PASS with --check-ids. Files in
Kaggle dataset `jvmusic/ber-france-v2`. Label-free estimates: A2F +0.00025, DP +0.0003, OOC +0.00025, E2 +0.0001, HNK
+0.0002, avg_ce4 US/India +0.0002.

- **LB: `avg_ce4_v2h` = 0.98809** (previous best 0.986665, +0.00143; label-free estimate was +0.0013). US/India part (avg_ce4)
  is ~+0.0002 by validation, so the France v2 rules gave ~+0.0012 (France F0.5 about +0.008).

## After LB 0.98809: census against US/India labels, v3 (27 Sep, 08:10 UTC)
- Structural census (address relation x name operation x qualifier change, record-level best S1) comparing France claim
  rates with US/India true rates. Checked and ruled out as big levers: same-name records on a different street (France's are
  generic "City + Category" names on genuinely different streets), empty-address records (train: 97.7% of empty-address
  records are true matches, 99.5% when one S1 has the same name; France already claims 96% of those; the rest are ties
  between same-name S1s), per-source counts (France is ~0.06 short in both S2 and S3), empty France S1 (their same-address
  records are category-swap decoys of singletons), US/India val errors (6,257 FN vs 394 FP, dominated by empty-address ties).
- First match for empty S1 (best exclusive candidate with p >= t): val +0.00003..+0.00012 over t = 0.5..0.75 but only
  57% precise (47 of 83 adds right, 35 on true singletons); not used.
- v3 fixes (France only): A2P (A2 with the position check: in-place Groupe is a category swap, -~250), A2N (suffix operation
  on records whose house number was dropped, street words of the S1 in the record's address, same city: +1,278; US/India
  95-97% true for that class), DP abbreviation guard fixed (club~clinique / amis~amicale were treated as abbreviations) and
  dotted legal forms collapsed (S.A.R.L. no longer counts as added words): DP 2,559 -> 2,702.
  `avg_ce4_v3` vs `avg_ce4_v2h`: France +1,382 / -403, US/India identical; validator PASS. Estimate +0.0001.
- **Qwen2.5-7B LoRA (AWS g5b, 1.9M pairs, training stopped 08:30):** dec val band AUC 0.9547 (stage-2 0.9651; e5-base
  round 3 0.9567). Added to the avg_ce4 CE ensemble on val (all 95.9k val S1): 0.990098 -> 0.990125 / 0.990162 / 0.990154 /
  0.990148 for Qwen weight 0.5 / 1 / 2 / 3 - at most +0.00006, below the +0.0001 bar; not used. Its test cut covers only 6%
  of the LB-confirmed France suffix matches (their blend is ~0.007, the cut starts at 0.05), so it cannot be checked on
  the France classes either.
- **Final candidate: `avg_ce4_v3`** (US/India = avg_ce4, France = v3 rules).

## After 0.98809: remaining-loss audit and the native-script route (27 Sep, 09:00-11:00 UTC)
- Where US/India lose F0.5 (val, 96,116 S1 with candidates; 0.99010): 73% of the loss is missed matches on S1 that have other
  matches (9,751 FN), 19% is 179 non-singleton S1 predicted empty; FPs are small (254 + 13 on singletons). Val S1 without any
  candidate (half of the singletons) are not in the val file, so all-S1 val is ~0.9904; US/India test predicted k-distributions
  equal val (US 3.38 vs 3.37, India 3.35 vs 3.35), so France is at about 0.975 on the LB (~4,000 S1-units below US level).
- Missed val records: empty-address in candidates 5,366, not in candidates 1,618; other not in candidates 1,837 (native
  script 618, invented single-token names 608, websites 156, other 456); other in candidates 1,191.
- IDs carry no generation order (Spearman S1 id vs matched id 0.0008 / -0.0002).
- Empty-address records: 97.7% are true matches in train and the test rate per S1 equals train (US 0.167), but only 51-59% are
  claimed in test (France 51%) and ~55% in val: the rest are ties. A name-only assignment model over the full train S1 pool
  (char-trigram kNN + rapidfuzz + margins, 2-fold) is 98.8% precise on 44% of all empty-address records, but its picks among
  the records the pipeline leaves unclaimed are only 52-70% precise (val -0.00003..-0.001); none are outside the candidate
  lists. Legal-form tie-breaker for same-name ties (unique canonical legal form, nsn 2-3): 95 val adds at 79%, +0.00003. Not used.
- France class census against US/India true rates: the France noise profile differs (house numbers rarely change, 13x more
  acronym names, which are 99.5% matches in US/India); unclaimed France same-name records at another street are generic
  "City + Category" names on different streets; claimed pure adds / swaps are suffix-list words, typos and English
  organisation words (matches in US/India). Unclaimed below-threshold classes are 35-62% precise on US/India val. No rule.
- **Native-script route (`scripts/native_route.py`, India).** Transliterate with the pipeline normaliser, top S1 of the country
  by name (char 3-gram TF-IDF, 0.6) + address (word TF-IDF, 0.4): the owner is in the top 20 for 98.1% of owned train records
  and first for 96.0%. A classifier on the top-5 pairs (train labels, 2-fold) is 99.8% precise at q >= 0.9 on all records.
  Val (exact, India val S1): adds for unclaimed records at q >= 0.9 are 91.9% precise overall, but pairs the pipeline had
  already scored and rejected are only 15% right; restricted to out-of-candidate pairs: 346 adds, 97.7% precise, val F0.5
  +0.00033 (India 0.98926 -> 0.99009; q >= 0.8: +0.00034, q >= 0.95: +0.00031). Test: 8,480 out-of-candidate adds at
  q >= 0.9 (1.05% of India S1 vs 0.9% on val; test has 26% more native records per S1 than train, i.e. more decoys).
  Expected LB about +0.0003-0.0004.
- `avg_ce4_v4` = `avg_ce4_v3` + the 8,480 native adds (`--extra_pairs`, also added to the candidate file); the v3 part
  reproduces with 0 pair differences; validator PASS with --check-ids.
- **General out-of-candidate route (AWS `yoddhas-route-oocand2-0927-1123`, ml.r5.24xlarge; the first run was stopped: the
  uncapped kNN over 9.2M records would not finish in 3 h, the 96 vCPUs ran like ~8-9 local cores on this sparse product).**
  Same method for non-native, non-empty-address US/India records (name trigrams capped at df 2%: 2.6x faster, owner in top 5
  98.8%); train side = 600k random records + records near validation S1 (reverse kNN) + their true matches, 2-fold classifier.
  Validation (out-of-candidate pairs, unclaimed records): q >= 0.9 493 adds, 53% precise (US 35%, India 70%), -0.00034;
  q >= 0.98 218 adds, 78%, +0.00002. Blocking already finds the clean Latin-script matches; what is left outside the candidate
  lists is mostly decoys (invented-name "same building" decoys, branches). Not used; no France route adds either.
- Other checks: unclaimed empty-address records with a unique same-name S1 are 38% precise on val (-0.0003); E-rule-like adds
  (same core, stage-2 p >= 0.87) 76% on val (E2 is neutral); invented-name records in the candidate lists are 80% true and the
  model already claims ~95% of the true ones (decoys differ by unit/suite); main CE ensemble instead of the French CE for France:
  +4,224 / -2,185 France pairs (adds same-name records on other streets, drops acronyms and typos): not used.
- **France probe `avg_ce4_v5_frprobe`** = v4 + France threshold 0.85 + 819 same-name / kept-qualifier records with the house
  number up 1-20 (US 81%, India 99% true for that signature; blend >= 0.5) + 371 empty-address same-name records with stage-2
  p >= 0.97 and a clear best S1: France +2,135 / -1 vs v4, US/India identical; validator PASS. Expected effect about +-0.00005.
- **Kavya's `kavyachetwani/ber-final-merge-v1`** (v2h + 11,577 US/India adds, built on a proxy blend dec + base2, val 0.98942):
  R3 phonetic route (Double Metaphone + house number, 8,015 India adds on records v2h leaves unclaimed) and family completion
  (3,562 in-candidate pairs rescored with CE evidence against the S1's accepted records; its own gate failed). Against v4:
  4,686 R3 pairs are identical to the native-route adds (same record, same S1; 11 records differ). R3's other 3,319 adds on
  records v4 leaves unclaimed: 2,023 native-script records where the native classifier prefers another same-name S1 and gives
  the R3 pair q ~0.02 (generic names such as "Balaji Finance" with only "H.NO 6, DELHI" as address; the R3 S1 carries the
  record's legal form in 1,406 of 1,630 disputed cases vs 969 for the classifier's pick, i.e. roughly 80% precise, worth about
  +0.00002) and 1,296 Latin-script records. The family adds are pairs the submitted blend scored and rejected (US: 548 with
  blend 0.2-0.5, 145 with 0.05-0.2); the stronger blend already claims half of what family completion adds on the proxy (test
  0.24% of S1 vs val 0.45%), and rejected in-candidate pairs are the least precise class on our val. Not merged.
- **Kavya's final_merge_v2** (v1 + R6: same-name compact key with a shared house number or city word, 6,818 adds, 99.5%
  India; her val 284 adds at 88.7%, +0.00017 on the proxy): 2,966 R6 pairs are identical to v4's native adds; 2,230 native
  records v4 leaves unclaimed (native classifier agrees on the S1 for 1,021 with median q 0.65, prefers another S1 for 1,209)
  and 1,611 Latin-script records (generic Indian names whose record carries the S1's unit number under the "Block X / Door No"
  noise, websites). From the two validations (her R3+R6: 603 val adds at ~93%; native route: 346 at 97.7%) her extra adds are
  roughly 87% precise, about +0.0001 LB; not validated on our blend.
- **`avg_ce4_v6_kv`** = v4 + her 7,156 R3/R6 adds on records v4 leaves unclaimed (outside the candidate lists; family
  completion not included). **`avg_ce4_v7_kv_fr`** = v6 + the v5 France bundle. Both validator PASS.
- **LB: `avg_ce4_v6_kv` = 0.989052** (v2h 0.98809, +0.00096; expected ~+0.0006: v3 France ~+0.0001, native route ~+0.0004 by
  val, R3/R6 extras ~+0.0001). The India out-of-candidate routes transfer at least as well as validation suggests.
- India in-candidate misses on val (355) are mostly invented names at the S1's address and small number changes (203 -> 204,
  B-81 -> B-83); addresses with a prepended "Block / Door No / Plot" number are only 100 of them. No rule.
- **`avg_ce4_v8_kv_fam_fr`** = v6 + Kavya's family-completion adds (3,559 on records unclaimed in v6) + the France bundle
  (+2,135 France). Probe for the last two submissions; validator PASS.
- **Kavya's list of unused val-positive items.** Qwen in the CE ensemble at weight 1: val 0.990098 -> 0.990162 (+0.00006; US
  +0.00012, India -0.00002; rebuilt val blend reproduces the stored one exactly); Qwen test scores cover 26% of India's and 41%
  of the US band (France keeps the French CE). Legal-form tie-breaker (nsn 2-3, record has a legal form): val +0.00003; 1,258
  test adds (US 578, India 384, France 296). First match for empty S1: not used (57% precise on val, every error is a full
  singleton loss, and test has twice the decoys). R3 disputed native records: already in v6. `final_build.py --extra_pairs` now
  keeps one S1 per record across route files (earlier files win).
- **`avg_ce4_v9s_safe`** = v6 + Qwen + tie-breaker (val-positive only); **`avg_ce4_v9_all`** = v9s + family completion +
  France bundle. Both validator PASS, no record on two S1.
- **Val bias found (hidden owners).** In the val set, a record owned by a non-val S1 looks unclaimed (its owner is not scored),
  so any rule that adds unclaimed records is charged with false positives that cannot happen on test, where the owner is
  present. Rule K (Kavya: best unclaimed candidate in [0.55, 0.8) for S1 with k <= 2, match profile) on US/India val: raw 750
  adds at 58% (-0.00057); with non-val-owned records removed 466 at 94% (+0.00054); without empty-address records only 74-81
  adds at 82-89% (+0.00004-0.00006). The empty-address part is an artifact of the correction (same-name ties: the owner does
  not reliably win), so Rule K is small for US/India. Global US/India threshold on corrected val: address records 0.8 -> 0.5-0.7
  +0.00014-0.00028 (keep 0-10% of non-val-owned records), empty-address records +0.0011-0.0014 (artifact). Not applied.
- **France invented-name records (the France blocking gap).** Full train labels: a single-token record with no name overlap at
  the same house number + street key as exactly ONE S1 is that S1's match 94.2% (US) / 94.9% (India); the non-matches are
  street-key collisions across different cities (exact address: 99.9%). Test claim rate of that class: US 94.2%, India 92.7%
  (= base rate; their leftovers are the cross-city collisions: val precision of unclaimed ones 32%), **France 59.7%**. Of the
  5,198 unclaimed France records, 5,135 are in the same city and only 507 were ever candidates (blend median 0.77 < 0.87);
  4,691 were never proposed by blocking. Same invented-name vocabulary as US/India (Onyxwex, ZEPHBRIXUMBRA...). Added as a
  France rule (5,135 pairs on ~5,020 S1; 1,750 of those S1 had k <= 2). Expected about +0.0002-0.00025.
- **`avg_ce4_v10s`** = v6 + Qwen + tie-breaker + France invented-name adds; **`avg_ce4_v10fr`** = v10s + France bundle. No family
  completion in either (357 of its pairs are claimed by the Qwen-augmented decoder itself). Both validator PASS.
- **LB: `avg_ce4_v10s` = 0.989412** (v6 0.989052, +0.00036; Qwen + tie-breaker ~+0.00005 by val, so the France invented-name
  rule gave ~+0.0003, more than estimated).
- **Name-type census at unique S1 addresses** (train match rate vs test claim rate US / India / France, after v10s): same 99.6%
  (99.8 / 96.7 / 100), drop 99.9%, web 95.7% (97 / 94 / 97), suffix 99.3% (99 / 96 / 98), invented 93.6% (93 / 92 / 94 after the
  rule), **acronym 100% (99.8 / 99.8 / 83.4)**; swap1 91% and multi 60% are France's category-swap decoys (47% / 10%), excluded.
  France acronyms: 1,730 unclaimed, 1,727 in the same city, only 228 ever candidates (blend median 0.78); e.g. "CG" for
  "Chasseurs Groupe SARL" at the same street and number.
- **`avg_ce4_v11`** = v10s + 1,727 France acronym adds; validator PASS, no family completion. Expected about +0.0001.
  `scripts/france_name_replaced.py` reproduces both France rules; the exact pairs used are in `artifacts/v11/` with the build
  command.

## Address-relation census and the per-city France gap: v12, v13 (27 Sep, 13:45-14:20 UTC)
- **Census coverage.** The 08:10 structural census crossed address relation x name operation x qualifier, but only for pairs
  already in the candidate lists. The v11 census covered records outside the lists, but only at the exact address of one S1.
  This round adds records outside the lists with other address relations:
  - **Same street key as exactly one S1, different house number** (train rates per country vs test claims): France has only
    ~3.4k such records, against ~290k US and ~245k India, because France copies keep the number. The large France classes
    are the known decoy signature: number up 1-20 with a suffix, swap or added word, and 0% claimed. US train rates for those
    classes are 0-7%. No gap.
  - **Name similarity at the exact S1 address** (leetspeak-normalised token_sort_ratio bins x name type): classes where
    France claims less are category-swap decoys, e.g. unclaimed "Calais Ecole" -> "Calais Centre", "Lille Club" -> "Lille
    Ecole" (claimed ones are OCR typos). Where US multi-word <55 matches are aliases ("Zetadova formerly Eye Center Inc"),
    France claims 99.9% of alias names ("dba / fka / formerly ..."), same as US / India. France has no "***" names.
  - **Zero-overlap two-word names at a unique address** (US 12% matches = every token OCR-corrupted): France's 33k are
    different businesses at the same address ("Bordeaux Ecole SAS" vs "YKT Compagnie"). No gap.
  - Name-replaced records with no house number at a unique street + city S1: 13 (not added).
  - Using every numbered address component instead of the first (apartment / residence parts first): 53 more France
    records, no train check, not added.
- **Fuzzy street (v12), train check.** A single-token acronym / invented record on the same house number + city (all
  non-numeric address parts) as one S1, whose street key differs by a typo (ratio 85-99, no other S1 >= 60), is that S1's
  match 95.2% (US, 983) / 86.1% (India, 208); exact street: 91.5% / 97.8%. Test claims of the class: US 90.7%, India 85.3%,
  France 53.5% (v10s) -> 96.4% (v12).
- **Per-city uniqueness (v13).** The v10s/v11 rules made the S1 unique on (house number, street key) across all of France.
  The same number + street in another city is common (Rue Voltaire, Bd Victor Hugo), so ~1k clear matches were skipped
  ("TC" for "Tamarii Compagnie SARL" at 283 Rue Degland, Lille).
  - Train, record at the (number, street, city) of exactly one S1 whose number + street exists in another city: acronyms
    100% (US 242, India 1,512), invented 94.2% (US 1,226) / 98.0% (India 3,743).
  - Test claims of that class: US 91-100%, India 96%, France 41% (invented) / 76% (acronym).
  - `--unique_by city` finds 1,825 invented + 844 acronym records unclaimed in v12. That is more than the census's 1,167
    because the rule's city match also accepts the département in place of the region ("Loire-Atlantique" for "Pays de la
    Loire", "Nord" for "Hauts-de-France"). 179 of them already met the country rule: 114 invented sat below the France
    threshold (blend 0.86) when the v10s adds were computed, and 65 acronyms were missed by the earlier initials-order bug.
- **Web handles.** Handles (@x, #x, ...com) with no S1 word inside and not the S1's acronym are 30% (India) / 51% (US) matches
  in train (with an S1 word: 99.9-100%). Six are dropped from the new invented adds ("@badunion" for "Centre Medical
  Relais"). The LB-confirmed v10s adds are left unchanged.
- **Invented word + legal form.** One core word ("Nexaria Co"), in no France S1 name, at a unique S1 address (all non-numeric
  address parts). Train 95.8% (US 406) / 98.6% (India 358); test claims US 95.4%, India 95.2%, France 47.2% (84 unclaimed).
  Another real word + legal form is 0% (excluded).
- **`avg_ce4_v12`** = v11 + 1,181 fuzzy-street adds (764 acronyms, 417 invented): 5,834,113 matches.
- **`avg_ce4_v13`** = v12 + 1,819 invented + 844 acronym per-city adds + 84 invented + legal-form adds (France +2,747, nothing
  removed): 5,836,860 matches, validator PASS, no record on two S1s, no family completion. 2,570 S1 touched (45 had no
  claim).
  - Expected about +0.00015 over v12, scaling from the LB-measured +0.0003 for the 5,135 v10s invented adds; v12 about
    +0.00007 over v11.
  - `scripts/france_name_replaced.py` (`--unique_by city --drop_foreign_handles`, `--mode fuzzy_street`,
    `--mode inv_legal`) reproduces all four new route files exactly. `artifacts/v13/` has every route file and the build
    command.

## Hidden-owner correction calibrated to test; first match for empty S1 (v14) (27 Sep, 14:20-14:55 UTC)
- **Calibration.** The hidden-owner "corrected val" removes every record owned by a non-val S1. That assumes test has no such
  unowned look-alikes. Checked directly: per 1,000 S1, count the records whose best score falls in a band. On val, split them
  by owner (true match / unowned decoy / another val S1 / non-val S1 = "hidden"); on test, count them all. The share of hidden
  records val must keep to match test's count is the calibrated keep share.
  - True-match density agrees at the top: best score >= 0.99, address records: test 3,322 vs val true matches 3,314 (US) and
    3,316 vs 3,311 (India).
  - In the address bands 0.5-0.6 / 0.6-0.7 / 0.7-0.8, test has as many unowned records as the RAW val or more. Keep share:
    US 0.36 / 0.76 / 1.06, India 0.17 / 0.28 / 1.31. Empty-address bands look like the corrected val (0-0.26, India 0.7-0.8
    0.92), but there the correction hides same-name ties whose owner is present on test.
  - Implied test precision of address records scored 0.5-0.8 is only 54-70% (corrected val said ~80%).
- **Options re-scored** (val F0.5 deltas; raw / corrected / calibrated):
  - address threshold 0.8 -> 0.7: -0.00005 / +0.00017 / -0.00005;
  - address threshold 0.8 -> 0.6: -0.00038 / +0.00021 / -0.00020;
  - Rule K on address records only (k <= 2): -0.00021 / +0.00006 / -0.00008 (61% precise);
  - Not applied: their corrected-val gain came from the removed unowned records.
- **First match for empty S1** (best candidate whose best S1 it is, blend >= t): t = 0.5 gives +0.000096 / +0.00038 / +0.000315
  (calibrated: 63 adds, 48 right, 15 on true singletons). For an empty S1 a wrong add costs only when the S1 is a true
  singleton, so the break-even is ~0.5, not ~0.73. Lower t is negative on raw val: t = 0.4 -0.00015, t = 0.3 -0.00078 (156 of
  246 adds on true singletons). Applied at t = 0.5 for US/India only. France is left to the France empty-S1 audit, because the
  France empty S1s with a high unclaimed score are mostly category swaps that DP dropped.
- `--first_min` in `final_build.py` now runs after the France rules and the route files. It only fills S1s still empty at the
  end, with records nobody claimed, one per S1 and one S1 per record. (Run before the route files, it took 2 records a route
  had assigned.)
- **`avg_ce4_v14`** = v13 + 904 first-match pairs (US 456, India 448; 4-CE blend median 0.65): 5,837,764 matches, validator PASS,
  no record on two S1s, France unchanged. Empty S1 rate now US 5.72%, India 5.73%, France 6.00%. Expected about
  +0.00008-0.00027 over v13 (raw vs calibrated val x 0.85 US/India share).
- Full-band Qwen job (`yoddhas-llm-qwen7b-g6e-all-0926-2053`) has been pending for capacity since 26 Sep 20:53 UTC and never ran;
  v13/v14 include the g5b Qwen scores.
- **Count-prior tie resolution (Kavya's research note), checked on full train** (every owner present, like test; val cannot
  measure ties because the same-name twin is almost never a val S1). There are 99k empty-address records whose name is
  shared by 2+ S1s, and one of those S1s owns the record 92.7% of the time. For 2-way ties with unequal address-match counts:
  - the S1 with 0 address matches owns the record only 18-29% of the time, against 61-73% for the other S1. It is usually
    a true singleton, so the proposed "give n=0 members a tied record" rule is wrong;
  - with both S1s at >= 1, the smaller one owns it 46-70% of the time (the note's direction, but below its own 0.75 bar).
  - No-go.
- **Effect on v14.** 187 of v14's 904 first-match picks are such empty-address same-name ties. On raw val they come out even
  (29 picks: 15 right, 13 on true singletons; +0.000012). On calibrated val they look positive only because the correction
  hides the twin.
  - **`avg_ce4_v14b`** = v14 without them (`--first_skip_ea_ties`): v13 + 717 US/India pairs, 5,837,577 matches, validator
    PASS. Val first-match delta: raw +0.000084, calibrated +0.000177.

## France empty-S1 audit; v15 (27 Sep, 15:05-15:20 UTC)
- **Empty S1 after v14b:** France 15,558 (6.00%), US 5.73%, India 5.75% (train singleton rate 5.58%).
- **Records at the empty S1's own number + street + city:** 29% of France empty S1 have at least one, vs 6.7% (US) and 5.4%
  (India). Almost all are claimed by, or belong to, another tenant of a multi-tenant address (a "Maison des Associations"
  can hold 100+ S1s), e.g. "Motoamis.Com" -> "Moto Amis SARL". Not a gap.
- **Best unclaimed candidate (blend >= 0.3, 1,124 France empty S1):**
  - ~430: records without an address or house number, mostly generic-name ties. Train: an empty S1 rarely owns a tie.
  - ~250: high-scoring in-place category swaps and same-name records with the number shifted, i.e. the France decoys that
    DP / HN_A drop.
  - A few web handles and suffixes.
  - No clean class. Adding these back would undo LB-validated rules without labels.
- **Web handles spelling an S1's full name:** France claims 96.6% (US 94.5%, India 96.9%). No gap.
- **Acronyms at a shared address**, exactly one tenant with those initials: train 99.7% (US 334) / 100% (India 346); test
  claims US 98.4%, India 99.4%, France 78.2%. 371 France records unclaimed; 7 of them go to S1s that were empty.
  `--mode acr_shared`.
- **`avg_ce4_v15`** = v14b + 371 France acronym adds: 5,837,948 matches, validator PASS, no record on two S1s.
- **France threshold 0.87 -> 0.86 / 0.85** (the same build otherwise; DP / HN_A still run): +428 / +782 France pairs.
  - Of those, 114 / 221 are same-name ties with no address and 113 / 216 are at addresses shared by 2+ S1s, i.e. coin flips
    on S1s that already have matches (wrong ~ -0.2, right ~ +0.1). Blanket lowering is ~0 EV; not applied.
  - **`avg_ce4_v15b`** = v15 + the 288 safe 0.85 adds (`scripts/france_thr_safe.py`): no-address records with a name unique
    to that S1 (68) and S1s alone at their address (220). 5,838,236 matches, validator PASS. Expected about +0.00001-0.00002.

## France gap audit in a fresh container; shared addresses: sub-numbers and the city key; v16 (27 Sep, 15:35-16:00 UTC)
- **Where France is short.** Predicted matches per S1 (v15b): France 3.315, US 3.385, India 3.375 (train truth 3.46 in both
  countries, whatever the S1's name-twin status). By the number of S1s sharing the S1's core name (nsn): France 3.379 / 3.321 /
  3.242 for nsn 1 / 2 / 3+, US 3.435 / 3.349 / 3.290. France has 43% of its S1s in nsn >= 3 (US 28%): generic "City + Category"
  names.
- **Empty-address records** (per 1,000 S1, unclaimed; train truth 95-99% matches in every tie class): France 24.7 (no S1 with
  that core name) + 7.3 (2-way name tie) + 46.5 (3+-way) = 80.6, US 66.9, India 60.6. The France excess is ties between
  same-name S1s, which carry no signal (count prior: no-go, above). Claimed 2-way ties are informed picks (legal form, word
  order), not coin flips, so none are dropped.
- **Records near unique-name France S1s** (sharing a core token with df <= 2): the unclaimed ones are the known decoy classes
  (in-place category swaps at the same number, suffix / qualifier / legal form with the number shifted up, same name in
  another city, "Participations / Holding / Distribution" qualifiers on another street). Records at the S1's number + city with a
  typo'd street are claimed 99.9% when the name is the same; the unclaimed ones are category swaps. No new class.
- **Precision side:** claimed France pairs where another S1 with the record's exact core name sits at the record's exact
  address: 3 (misassignments are not a lever).
- **Shared addresses, sub-numbers.** Single-token records (invented names, acronyms) at an address with 2+ tenants cannot be
  placed by the name. Train (all owners present): when the record carries a unit (Unit / Suite / Apt / # ...) or a French
  sub-number (bis / ter / A / B) and exactly one tenant at the same number + street + city has the identical unit + sub-number,
  that tenant owns the record 98.8% (US, 768) / 96.3% (India, 217); without a unit on the record 62% / 80% (not used). At
  unique addresses a unit mismatch is still 83-89% true, so no existing adds are removed. Test unclaimed: US 0, India 10,
  France 287 (e.g. "CSA | 10BIS R. Alexis Maneyrol" -> "Collège Sainte Azureen | 10 Bis Rue Alexis Maneyrol").
- **v15 acronym rule, city key.** v15 keyed the address on the set of all non-numeric parts, so a record naming the
  departement ("Loire-Atlantique", "Nord", "Gironde") instead of the region, or dropping it, never met its S1. With the city
  recognised from the city list: 709 more unclaimed acronyms at shared addresses with exactly one tenant holding those initials
  (v15 had 371), 671 after skipping sub-number / unit conflicts and the records the unit rule already places.
- **`avg_ce4_v16`** = v15b + `artifacts/v16/fr_unit_adds.parquet` + `artifacts/v16/fr_acr_city_shared_adds.parquet` (the same
  `--extra_pairs` semantics: unclaimed records only, one S1 per record, earlier file wins): +968 (France 958, India 10),
  5,839,204 matches, validator PASS with --check-ids. Expected about +0.00003 over v15b (~970 adds at ~97-99%).
- **Also checked after v16 (no change):**
  - Unclaimed empty-address records whose core name belongs to exactly one S1 (full-train uniqueness): train owner = that S1
    96.7% (US) / 98.1% (India), but the unclaimed ones on val are 42% precise (blend 0.7-0.8: 73-77%). The model's rejections
    are informed. France has 552 such records; not added.
  - Typos at a unique France address, names equal after fuzzy word alignment (train, unique S1 address, no word added or dropped):
    word typo US 97.5% / India 94.5%, a substituted letter in a short token 99.9% / 99.3%, a letter DELETED from a short token
    93% / 74%, a letter INSERTED into a short token (acronym "OVN" -> "OVNX") only 71.5% / 33% (a decoy operation). France claims
    98.4% / 99.2% / 72.6% / 37.7% of those classes, i.e. at the train truth rates. The France "suffix + typo" records ("KU Club
    SAS" -> "KCU SAS Groupe") are nearly all acronym-letter insertions: decoys.
  - One-letter substitution in a short token at the same address: train 94% / 88%; 127 unclaimed in France (~+0.000003).
  - Claimed France pairs with the same house number but a street whose name words all differ (<60): 222 (about a third are
    address-component-order false alarms). Train: same name + number with a different street is 0.5-4% true when the city also
    differs; same-city cases are rare in US/India. Removing ~150 would be worth ~+0.00001; not done.
  - Records at shared France addresses with exactly one same-name tenant: 99.999% claimed. Tail of k: train truth never exceeds
    11 matches per S1, and neither does any test prediction.

## Density audit against train truth; legal-form tie-break; v17 (27 Sep, 16:15-17:20 UTC)
- **Per-class density check (label-free test audit).** True matches per S1 of any structural class should be the same in train
  and test (same generator, random S1). For US/India, claimed pairs per 1,000 S1 in test vs true pairs per 1,000 S1 in train,
  by name relation (same core / subset / overlap / disjoint) x address relation (empty / no number / number+street / number only /
  street only / neither): every class with an address agrees within +-2 per 1,000 (e.g. US same core + number + street 860.9 vs
  861.6; changed-number classes by direction and legal-form change also agree, e.g. US lower number + legal form kept 24.5 vs
  24.2). The whole US/India shortfall is address-less records: US -64 per 1,000 (same core -31, overlap -26, subset -7), India -58.
  Precision cannot be read off the densities, but the recall side of US/India is at the tie ceiling.
- **Address-less records with no exact-name S1** (typos): train owner has a unique name only ~50% of the time (US 39.7k of 78.5k,
  India 22k of 46k); the model claims ~48% of them in test, i.e. the resolvable part.
- **Other tie signals checked on train, all coin flips:** file row order (Spearman 0.001, rows shuffled), per-S1 source split
  (n2/n3 spread 1-4), name similarity to the twin's other copies in the same source (48% / 53%), full-name identity (74% US).
- **Test decoys come in clusters.** Groups of 2+ unclaimed records with the same core name + number + street: US 38.8 per 1,000
  S1 in test vs 1.4 unmatched in train, France 20.9 (decoy "branches" with several copies; legal form changed, number shifted).
  Claimed records whose branch twin is unclaimed at a changed number: US 995, India 713, France 394; not removed (in train 55% of
  changed-number true matches share their number with a sibling, mostly in the same source).
- **Legal-form tie-break.** Address-less record carrying a legal form, core name shared by 2+ S1s, exactly one of them with that
  legal form (canonicalised). Train precision by the number of same-name S1s without any legal form (n0): n0 = 0 95.1% (US 3,810)
  / 96.7% (India 365); n0 = 1 90.3% / 91.0%; n0 = 2 83.9% / 83.5%; n0 >= 3 56-61%. France copies never swap the legal form
  (claimed same-name pairs: swapped 0.02%, US 2.6%; only 150 same-address swapped records), so France uses n0 <= 2 and also
  case B (the record's form is on none of the twins, exactly one twin has no form; US 80% / India 39% in train because they swap).
  Test adds on records unclaimed in v16: France 483 (A) + 199 (B), US 89, India 52 = 823 (US/India already claim most of these).
- **`avg_ce4_v17`** = v16 + `artifacts/v17/legal_tie_adds.parquet` (`scripts/legal_tie.py`): 5,840,027 matches, validator PASS
  with --check-ids. Expected about +0.00002-0.00003 over v16.
- **Rejected-subset check (val, 17:25 UTC).** The tie-break adds only records the model left unclaimed, so its all-records train
  precision overstates it. On val, among Case A records the decoder did not claim: US n0 = 0 81.8% (11), n0 = 1 82.6% (46),
  n0 = 2 72.7% (33); India 50% (2) / 37.5% (8). The model's claimed acronym-letter-insertion pairs are 97.9% right on val (its
  picks inside that class are informed), so France's are not dropped.
- **`avg_ce4_v17b`** (recommended) = v16 + `artifacts/v17/legal_tie_adds_v17b.parquet`: v17's adds without India (52) and without
  the France adds whose stage-2 p < 0.5 (57; the model clearly rejected them, most France adds have p 0.84-0.97 and were pulled
  under the threshold by the French CE only). 714 pairs (France 625, US 89), 5,839,918 matches, validator PASS.
- **v17c** (not uploaded; `artifacts/v17/legal_tie_adds_v17c.parquet`): the tie-break gated on stage-2 p >= 0.5 in every country.
  On val the US rejected-subset precision is 88.2% (51) with p >= 0.5 vs 33% (6) below. France 618, India 38, US 79; about
  +0.000005 over v17, so **v17 is the final** (the zip's `output/` holds v17).
- **Final package** `Yoddhas_submission.zip` (`scripts/make_code_zip.py ... --output <v17 dir> --doc Documentation.md`): the code
  folder now carries `final_build.py`, every route/rule script, `append_routes.py`, all route pair files, and Kavya's R3/R6 Kaggle
  kernels (`third_party/kavya_r3r6/`: rt-miss, rt-score, rt-build, rt-miss2-tr, rt-miss2-te, rt-score2, plus sh-hn; they use the
  internet only to pip-install metaphone / sparse_dot_topn / rapidfuzz, inputs are the challenge data and our own runs).
  `append_routes.py` on the v15b build reproduces the v17 matching file byte-for-byte (candidate file: same 7,670,616 pairs).
- **Final submission (28 Sep).** v17c was uploaded after all and is the final file: public LB **0.989828**. The package's
  `output/` holds v17c (5,839,939 matches, 7,670,609 candidate pairs), and the README and Documentation.md document v17c.
- **Package clean-up (28 Sep).** The code folder ships only what the submitted build reads:
  - the six R3/R6 kernels as `src/routes_r3_r6/`, without HANDOFF.md, the Kaggle kernel-metadata files or the sh-hn France
    kernel (its rules live in `final_build.py`);
  - `odds_extra_france_iso.parquet` only;
  - the 14 route files named in the final_build / append_routes commands, without the duplicate v11/v13 copies, the v14
    first-match files, the v17 / v17b tie-break files or the route READMEs.

  Rerunning the README's append_routes -> legal_tie -> append_routes steps from the extracted zip on the v15b build reproduces
  both output files byte-for-byte; the regenerated tie-break file holds the same 735 pairs in another row order. Validator PASS.
- **Package: exact end-to-end README (28 Sep).** Every step of the final build now has its command in the zip's README,
  recovered from the Kaggle notebooks that ran it, `sagemaker_pipeline.py` and this log:
  - Cross-encoder training sets: the S1 samples were matched against our files (100% of S1): `ce_data` = seed 0 drawn outside
    the validation S1 of the earlier `big` run (`--neg_rate 0.25 --drop_s1 0.19`), `ce_data2` seed 5, `ce_data3` seed 9,
    `ce_data4` seed 13 (each excluding the earlier sets, drawn from `work`). Rounds 3/3b/4/4b: lr 1.5e-5, 170/150/300/300 min.
  - The French cross-encoder's band is the France band [0.003, 0.9995) of the plain `work` test scores (all 392,479 pairs
    match), not of the French-words re-score as the old README said. `nr/ce_test_france.parquet` = those scores + 56,129
    newly scored France pairs (53,436 of them the `nr_fr` band [0.02, 0.998) not scored before).
  - The top-5 run needs `--cap_rname 5` (the route is off by default); the old README's run commands lacked it.
  - Qwen (AWS, `yoddhas-llm-qwen7b-g5b`): settings from this log (lr 1e-4, 4 x bs 6, round-2 pairs, stopped 08:30 UTC).
    peft is pinned to 0.21.0, the newest release (15 Sep 2026) when the job pip-installed it unpinned. The AWS tools were not
    available in this session, so the jobs' own argument lists were not read.
- **Kavya's R3/R6 kernels as scripts (28 Sep).** `rt-miss` / `rt-miss2-tr` + `rt-miss2-te` / `rt-score` / `rt-build` / `rt-score2`
  are now `scripts/r3_candidates.py`, `r6_candidates.py --split`, `r3_score.py`, `r3_build.py`, `r6_score.py`: inputs as
  arguments instead of `/kaggle/input` globs, `--out` instead of `/kaggle/working`, no runtime pip installs; the logic is
  line-for-line hers (the originals stay in `third_party/kavya_r3r6/`). Her merge kernel (final-merge) is not accessible,
  so `scripts/r3r6_merge.py` implements it for R3 + R6 (R3 prob >= 0.70, R6 >= 0.55, records unclaimed in the base v2h,
  one S1 per record by probability, then records unclaimed in v4). Checks on her real outputs:
  - `r3_build.py` on her rt-score output reproduces her r3_v1 files (same pairs; 573 rows list the ids in another order, as
    her own group_by does) and the same metrics (8,015 adds on 6,835 S1, validator PASS).
  - `r3r6_merge.py` gives R3 8,015 / R6 6,871 / merged 14,826 (her report: R3 8,008 + R6 6,818 = 14,826 after exclusivity
    with family completion) and 7,155 of the 7,156 pairs of `kv_r3r6_extras`. The 7,156th (S1-350491974, S2-172056214) is a
    family-completion pair: the shipped file = every pair final_merge_v2 adds outside the v2 candidate lists, on records v4
    leaves unclaimed (7,156 exactly; = v6_kv - v4), and one family pair lies outside those lists.
- **`legal_tiebreak_adds.parquet` (v9) has no script.** It was made by a notebook snippet that was never committed. The file
  follows the rule "address-less record, word-set core name shared by 2-3 S1, legal form on exactly one of them" (every pair
  has the same word-set key and nsn 2-3), but no variant tried reproduces it exactly (closest: the pipeline's name_legal
  against v6, 35 missed / 167 extra). The README says so; `legal_tie.py` is the refined rule.
