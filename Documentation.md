# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Yoddhas  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve every Source 1 business against ~10M Source 2/3 records with a four-step pipeline:
1. **Blocking.** Three complementary retrievers (word TF-IDF, multilingual dense retrieval, and a reverse name route for records without an address), unioned and cut by a learned first-stage model to **4.41 candidates per S1**, keeping **98.9% of true matches**.
2. **Scoring.** A two-stage XGBoost model scores the candidates using similarity, competition and "decoy-signature" features. Two pipeline runs (reverse-name route top 5 and top 10) are averaged.
3. **Uncertain cases.** An ensemble of three fine-tuned multilingual-e5-base cross-encoders re-scores the pairs the model is unsure about (a French-adapted e5-small does it for France).
4. **Decoding.** One exclusivity-aware decode with per-country thresholds, then label-free generator rules for France measured on the test data itself.

Two findings drove most of the gain:
- **The unmatched records are generated lookalikes ("decoys").** These are branches of the S1 business with an added qualifier word and a shifted house number. We model that signature explicitly.
- **France has no training labels.** We learn its decoy vocabulary from confident test predictions, and we measure how France's data generator differs from US/India directly on the test records (house numbers almost never change on a French match, French noise suffixes, category-word swaps are decoys). No external data and no hand labelling.

Validation macro F0.5 (US/India, held-out S1): **0.9902**. Public leaderboard: **0.98667** (best submitted), final file described here.

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale.**

| | S1 | S2 + S3 | Countries |
|---|---|---|---|
| Train | 2,206,821 | 10,320,219 | US 1.32M, India 0.88M |
| Test | 1,732,544 | 9,969,589 | India 810k, US 663k, **France 259k** (unseen in training) |

**Ground truth.**
- 7,638,365 true pairs: 3.46 matches per S1 (1–5 per source).
- 5.6% of S1 are singletons.
- Every S2/S3 record matches at most one S1 (**exclusivity**).
- The country label always agrees within a true pair, so we block per country.

**Noise in true matches.**
- Names: legal-suffix changes (Pvt Ltd / Private Limited, SARL / S.A.R.L.), dropped words, initials ("BS College" → "BC"), typos and OCR-like substitutions (0→o, 1→l), DBA / "t/a" / "formerly" phrases, and word-order swaps.
- 18% of Indian S2/S3 names are in Indic scripts (Devanagari, Tamil, …).
- Addresses: abbreviations (Rd/Road, R./Rue, AV/Avenue), reordered components, "No"/"Nº" prefixes, leading zeros ("02134"), missing parts.
- About 3% of addresses are empty.

**The key insight: decoys.** 26% of training S2/S3 records match no S1, and they are not random. 72% of them are near-duplicates of an existing S1 (name similarity > 0.92). They are deliberately generated "branches" of the business:

| Among near-duplicate candidates | Decoys | True matches |
|---|---|---|
| Same first house number | **10%** | **79%** |
| Same legal form | 49% | 74% |
| Address near-identical | 32% | 65% |

A decoy typically adds a qualifier word to the S1 name:
- English: Holdings, Care, Clinic, East/West/North/South, Central, Downtown, Valley, Exports, International, India, American…
- French: Ateliers, Sainte, Groupe, Participations, a city name, "France"…

The house number usually shifts upward by a small step.

**Train/test shift.** Test has more S2/S3 records per S1 (5.5–5.8 vs 4.7). The shift that matters is in the scores, and it sits almost entirely in France.

Pairs per S1 kept after exclusivity, by score band:

| | ≥ 0.99 | 0.70–0.99 |
|---|---|---|
| US test | 3.35 | ≈ validation's true matches, band by band |
| India test | 3.34 | ≈ validation's true matches, band by band |
| **France test** | **3.08** | **about 2×** |

US/India also have only ~15% more pairs in the raw 0.3–0.9 band.

- France's best-vs-second margin is half that of the US.
- A leaderboard pair confirms the cost: raising France's threshold from 0.90 to 0.97 gained +0.0027. The removed French pairs were only ~20% correct.
- The leaderboard is therefore precision-limited in France. A validation score over-states the leaderboard by ~0.008, and most of that gap is France.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (sparse + dense retrieval), then a two-stage gradient-boosted classifier, then a transformer cross-encoder on uncertain pairs, then a global exclusivity decode.

**Core Innovation:** Explicit modelling of the generated-decoy signature:
- Learned log-odds of the words a candidate adds or drops, and a signed house-number shift.
- Transductive learning of the same word scores for France, the country without labels.
- Prior-shift-corrected per-country thresholds that estimate each score band's precision on the test population itself.

---

## 3. Candidate Generation (Blocking)

All blocking runs **within country**. Every step is vectorised and chunked; the whole test set blocks in about 1.5 hours on a 4-CPU / 1-T4 Kaggle machine.

- **Blocking keys used:**
  1. **Word TF-IDF.** Name core (legal forms and titles removed) plus address tokens, over normalised text. Tokens in more than 1% of records are dropped. Top-30 cosine neighbours per S1 via `sparse_dot_topn`.
  2. **Dense retrieval.** `intfloat/multilingual-e5-small` (MIT, 118M parameters) embeddings of "name | address", exact top-k by blocked GPU matrix products. It handles transliterations, abbreviations and reordering that share no tokens.
  3. **Reverse dense.** The top-2 S1 for each S2/S3 record. This recovers name-only records whose generic names tie with many lookalikes in the forward direction.
  4. **Reverse name route.** For every S2/S3 record **without an address** (3.3% of records, but 64% of the true pairs the first three routes missed), the top-k S1 of the same country by character 3-gram TF-IDF on the name. Two runs use k = 5 and k = 10.
  5. **Learned filter (stage 1).** The union of TF-IDF top 20, dense top 20, reverse top 2 and the reverse name route is scored by the stage-1 XGBoost model. Stage 2 keeps each S1's top 15 with p₁ ≥ 0.005.

- **Candidate pairs generated:** **7,637,679 test pairs (4.41 per S1)**: the union of the pairs the two pipeline runs score. This is the exact set the final model scores, and it is what `candidate_pairs.tsv` contains.

- **How we ensured true matches were not lost.** Recall was measured on the full training ground truth for every route and cut-off:

| Candidates | Recall |
|---|---|
| TF-IDF top 30 | 0.953 |
| Dense top 30 | 0.958 |
| TF-IDF 20 ∪ dense 20 | 0.9795 |
| + reverse top 2 | 0.9848 |
| after the stage-1 filter (validation) | 0.9849 |
| + reverse name route (top 5), after the filter | **0.9888** |

The filter removes ~88% of the pairs at no measurable recall cost. The oracle macro-F0.5 of the candidate set (a perfect scorer on these candidates) rose from 0.9953 to 0.9964 with the reverse name route (0.9967 with k = 10).

Text normalisation before blocking and matching:
- Accent stripping, lowercasing, OCR digit fixes, merging of spaced initials ("S.C.I." → "sci").
- Removal of junk tags, DBA splitting, legal-form extraction (EN + FR + IN).
- Street-type canonicalisation (EN + FR), house-number extraction.
- A **learned Indic→Latin word dictionary**: 1,362 words mined from training pairs by co-occurrence × Jaro-Winkler against a rule-based transliteration.

---

## 4. Matching Model

**Features used (≈90):**
- **Name features.**
  - Token-set, partial and plain ratios; Jaro-Winkler; Levenshtein.
  - IDF-weighted token overlap (max / sum / missing mass); acronym match; legal-form equality or missing.
  - Name-in-address containment.
- **Address features.**
  - Token-set ratio, IDF overlap, word-only similarity.
  - House number: first-number equality, Levenshtein, ratio, log-abs-difference, count of numbers.
  - Empty-address flags.
- **Decoy-signature features.**
  - For the words the candidate name **adds** or **drops** relative to the S1 name: min and sum of learned log-odds, word counts, and the signed house-number shift.
  - The log-odds come from a held-out 10% S1 slice that neither model stage trains on.
  - English→French equivalents inherit scores. French words unseen in training are learned transductively (see below).
- **Retrieval and competition features.** TF-IDF and dense scores and ranks. Rank, gap and margin of each score **within the S1's candidate list and within the candidate's competing S1s**; global name/address similarity ranks over all pairs; "twin" counts. The candidate-side margins are the strongest features, because each record belongs to at most one S1.
- **Stage-2 context.** The stage-1 probability and its rank, gap, sum and count above thresholds, per S1 and per candidate.

**Model type:**
- **Two-stage XGBoost** (GPU, hist, lossguide, 255 leaves, eta 0.08, early stopping). Stage 1 trains on 45% of the training S1, stage 2 on a disjoint 45%, and 10% is held out for validation. Easy negatives are subsampled (rate 0.2) with compensating weights.
- **Cross-encoder.**
  - `intfloat/multilingual-e5-base` (MIT, 278M parameters) fine-tuned as a pair classifier on "name | address" text.
  - Trained in rounds on hard pairs of fresh training S1 that are never in the validation split. Each round is 160k S1 and 2.2M pairs (24% positives), 150–300 GPU minutes, and starts from the previous checkpoint (lr 3e-5 → 2e-5 → 1.5e-5).
  - Lineage: round 1 → round 2 → round 3 (new S1 slice, seed 9) and round 3b (seed 13) → round 4 (round 3 on the seed-13 slice) and round 4b (round 3b on the seed-9 slice). The final US/India model is the **equal logit average of rounds 3, 4 and 4b**.
  - The first small-model version (e5-small) is kept for France, where it is adapted to French (see below).
  - It re-scores pairs in the uncertain band 0.02 ≤ p < 0.998 (19% of pairs).
  - It is blended on the logit scale with weight 0.6 for XGBoost (chosen on validation).
  - AUC on the uncertain band: XGBoost 0.962. The cross-encoder rises from 0.938 (e5-small) to 0.945 (e5-base) to 0.953–0.955 (rounds 3–4b). Their blend reaches 0.972, so the two are complementary.
- **Two pipeline runs averaged.** The whole pipeline runs twice, with the reverse name route at k = 5 and k = 10; a pair's probability is the mean over the runs that scored it (clean validation +0.00013).
- Larger or different models were measured and not kept: e5-large (band AUC 0.948), Qwen2.5-7B LoRA at 217k pairs (0.93).
- CatBoost (0.9784) and random forest (0.9693) were weaker than XGBoost (0.9799) on identical features, and blending them did not help.
- All models are MIT/Apache-licensed and far below 8B parameters.

**Threshold selection method:**
1. **Exclusivity.** Each S2/S3 record is kept only for its highest-scoring S1.
2. **Per-country thresholds, corrected for prior shift.** For each score band (0.5, 0.7, 0.8, 0.85, …, 0.99):
   - The expected true matches per S1 come from validation. France, which has no labels, uses the US/India average.
   - They are divided by the observed test pairs per S1 in that band. The result estimates the band's precision on test.
   - A pair improves macro F0.5 only if its precision exceeds ≈ F*/(1+β²) ≈ 0.78. Each country's threshold is the lowest band edge above which every band clears 0.78.
   - Result on the first model: US 0.80, India 0.80–0.85, France 0.97. With the decoy features and cross-encoders the validation and test band densities now agree for US/India (ratio ~1.0 in every band ≥ 0.7), so US/India use the validation optimum **0.80**.
3. **Cross-check with a label-shift estimate.** Calibrate validation scores (isotonic), run EM on each country's test pair prior (Saerens et al.), and find where the corrected probability reaches 0.78.
   - It agrees for US/India (~0.74–0.78).
   - It suggests a lower France threshold (~0.84).
   - The two estimators assume different things about the unlabelled country. The France threshold was therefore settled with public-leaderboard probes that change only French predictions: 0.97 → 0.93 → 0.90 → **0.87**, each an improvement.

**Transductive decoy words for unlabelled countries** (`fit_pseudo_odds.py`):
1. Confident test predictions (blended p ≥ 0.99 as match, ≤ 0.05 as non-match) serve as labels for France. The same word log-odds are fitted on near-duplicate names.
2. We keep the 101 words the training odds do not know and halve their scores (on US test, the same procedure reproduces the training scores with correlation 0.69 at about 2× magnitude). Test is then re-scored.
3. The words found are exactly the French decoy qualifiers: sainte, ateliers, a city name (lille, nantes, bordeaux, calais), jean, ei, departemental, residence, ehpad, musique, maison, france…
4. Generator typos (farmacie, clbu, teatre) come out as match-like.
5. **The effect on France:** 23k "+ France" lookalikes leave the candidate set, e.g. "club projets" vs "club projets france" goes 0.99 → 0.02. The words are applied to French S1 only; US/India scores are unchanged.

**Cross-encoder adapted to France** (self-training, `ce_data.py --parts pseudo` + `ce_train.py --init_dir`):
- The cross-encoder continues training for 30 minutes on 300k confident French test pairs (50/50) plus 13% of its original training pairs, at lr 2e-5.
- Validation AUC is unchanged (0.938), so there is no damage to US/India.
- On France it learns to reject swapped category words at the same address ("Biserica Loisirs" vs "Biserica Fetes") and same-name records at a different street.

**France's generator, measured on the test data** (`scripts/final_build.py`, all rules label-free):

The data generator treats countries identically in structure (on train, US and India agree to 4 decimals on matches per S1 and singleton rate), but France's noise vocabulary and rates differ, so a model trained on US/India errs on France in predictable directions. We measure each noise dimension on France's own confident predictions, correct it by the US/India ratio of confident-test rate to true-train rate, and compare:

| Measured on test | US (true) | India (true) | France |
|---|---|---|---|
| House number changes on a true match | 12.4% | 23.7% | **~1.1%** |
| Noise suffix words | center, services, partners, incorporated | same | **fils, groupe, développement, associés, cie, services, frs** |
| Decoy qualifier words | holdings, care, clinic, … (shift the house number) | same | centre, service, fédération (shift the number ~60%) and **category swaps** (club → école, same number) |

Rules applied after decoding, France only:
1. **A2 (suffix adds).** An unclaimed record whose name drops ≤ 1 word of its best S1 (by name + address similarity), adds exactly one French noise suffix (fils, groupe, développement, associés) and keeps the house number is matched to that S1. The training word odds call these words decoys (−4 to −7), so the model rejected them.
2. **HN_A (changed house number).** A claimed France pair whose first house number differs and whose blended probability is below 0.998 is dropped: French matches change the number only ~1% of the time.
3. **D (category swaps).** A claimed France pair that replaces one S1 word by a category word (club, comité, amicale, école, …, or "France") at the same number is dropped. A leaderboard probe that *added* 23.8k such pairs lost 0.0032, confirming they are decoys. On India's labels, a true match never adds the country word to the name (0 of 3.06M), which settles "+ France".
4. **E (empty addresses).** A France record without an address, unclaimed, whose name clearly points to one S1 (token-sort margin ≥ 15 over the second candidate) is matched when the names share every core word and the XGBoost probability alone is above the France threshold, or when the only change is a French noise suffix. US/India claim this record type 82–97% of the time; for France the French cross-encoder had vetoed it.

Probes the leaderboard rejected, and ideas measured and dropped, are listed in section 5.

---

## 5. Results & Error Analysis

**Validation** (held-out 10% of training S1 in the stage-2 slice, full candidate pool, singletons included):

| System | Val F0.5 (macro) |
|---|---|
| TF-IDF-only blocking + LightGBM | 0.9660 |
| TF-IDF ∪ dense + XGBoost | 0.9799 |
| Two-stage XGBoost + reverse dense + competition features | 0.9837 |
| + decoy-signature features | 0.9865 |
| + cross-encoder blend (e5-small) | 0.9886 |
| + e5-base cross-encoder | 0.9891 |
| **+ second e5-base round (final)** | **0.9893** |

- **Public leaderboard:**

| Submission | Public LB |
|---|---|
| First model | 0.9705 |
| + thr 0.90 + exclusivity | 0.9730 |
| + per-country thresholds | 0.9757 |
| + decoy features + e5-base cross-encoder | 0.9829 |
| + France fix | 0.9831 |
| France threshold 0.93 (on the baseline) | 0.9830, vs 0.98286 at 0.97 |
| + cross-encoder round 3, France threshold 0.90 | 0.98443 |
| France threshold 0.87 | 0.98452 |
| + reverse name route (k = 5) | 0.98553 |
| + France rules A2 + HN_A | **0.98667** |
| Probe: + 23.8k French category-swap pairs (C1) | 0.98345 (rejected: they are decoys) |
| Final (route average + cross-encoder ensemble + rules D and E) | [final] |

**Unseen-country stand-in** (train on US only, validate on India):
- F0.5 falls from 0.984 to 0.933. That confirms France's implied ~0.92 is a transfer problem, not bad luck.
- A US-only cross-encoder recovers 2.3 points (0.957).
- Self-training on pseudo-labels recovers nothing (0.9376 → 0.9358).
- The same stand-in calibrates the value of an added pair against its precision (break-even ≈ 0.70). It shows France's band [0.93, 0.97) is only ~0.75 precise, so France stops at 0.93.
- **Common false positives (wrong merges):**
  - Generated branches that change a category word at the same address ("Rayon Comite SAS" vs "Rayon Musique SAS"); rule D removes the single-word ones.
  - Branches that add a qualifier the model has not seen.
  - Same-name records at a different street with the same house number.
  - Records with an empty address and an exact name, which could be a branch or the entity itself.
- **Common false negatives (missed matches):**
  - Brand/trade-name records that share only the address ("Pessac Ecole SAS" vs "NOVIQUO"). The generator renames 1.6% of true matches this way, but it also creates invented-name decoys at the same address: adding the unclaimed ones is only 19% precise on validation.
  - Heavy abbreviations or initials.
  - Empty-address name-only records with generic names (the largest blocking miss: 24% of TF-IDF misses).
  - Indic-script names without a dictionary entry.
- **Where the validation loss sits** (final US/India validation, 1 − F0.5 = 0.0114):

| Error type | Share of the loss | Note |
|---|---|---|
| S1 with some matches found and some missed | 71% | 46% of the missed pairs never reached the candidate set |
| Matched S1 predicted empty | 16% | |
| Wrong merges on S1 that have matches | 11% | |
| Wrong merges on singletons | 2% | |

  The final system is recall-limited on US/India. France has the same structure but no labels; its remaining errors are spread over pairs where several noise operations combine.

**What did not work** (each measured, then dropped):

| Idea | Result |
|---|---|
| Rare-key hash blocking | 0.92 recall @30 on dev; ran out of memory at scale |
| Char-3-gram TF-IDF | 0.70 recall on dev |
| Absolute document-frequency caps | Destroy recall |
| Triangle-consistency features (candidate vs the S1's anchor match) | 0.98302 vs 0.98297 |
| Test-density simulation by dropping 19% of S1 | 0.9830 vs 0.9837 |
| Decoy duplication (virtual copies of decoy records) | See below |
| CatBoost / random forest / blends with XGBoost | ≤ XGBoost alone |
| Expected-F0.5 decoder on the LB | 0.970 vs 0.973 for a plain threshold. It trusts validation calibration, which the test prior shift breaks |
| Stage-3 stacker on the model + CE scores | +0.0003 on validation; not worth the complexity |
| Importance weighting of training S1 towards the test mix | no gain |
| France cross-encoder trained on the leaderboard-confirmed French classes | memorises the word lists: AUC 1.0 on seen words, 0.07 on held-out words |
| Synthetic French pairs (generator ops applied to French records) for the cross-encoder | re-learns the rules; beyond them it encodes wrong assumptions (e.g. empty address = decoy) |
| e5-large cross-encoder | band AUC 0.948; +0.00002 in the ensemble |
| Qwen2.5-7B LoRA cross-encoder at 217k pairs | band AUC 0.93, lowers the blend |
| "Twin" counts of the same added word as a decoy signal | contradicted by US/India labels |

**Decoy duplication in detail:**
- Virtual copies were made only of records that match no S1. A twin therefore identified a decoy, which leaks the label.
- It looked +0.0016 better on validation.
- On test it matched 35k S1 that the base model rejects: only 0.9% of S1 predicted empty, against 5.6% singletons. Never submitted.

---

## 6. Conclusion

Entity resolution at this scale is won in the tails. Candidates are cheap to retrieve: three routes reach 98.9% recall with 4.4 pairs per S1, and a dedicated route for address-less records was the largest late gain. The hard part is telling a business from its generated branches, especially in a country without labels. Modelling the decoy signature, cross-encoders for the uncertain band, and measuring the unseen country's generator on its own test records (rather than self-training on the model's confident guesses) each gave measurable gains.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Role |
|---|---|
| `src/ber/normalize.py` | name/address normalisation, legal forms, Indic transliteration |
| `src/ber/translit.py` | learns the Indic→Latin word dictionary from training pairs |
| `src/ber/blocking.py` | TF-IDF blocking per country (sparse_dot_topn) |
| `src/ber/dense.py` | multilingual-e5 encoding + exact blocked top-k on GPU |
| `src/ber/features.py` | pair, competition, decoy-signature and global features; word log-odds fitting |
| `src/ber/pipeline.py` | stages, candidate loading/pruning, decoders, prior-shift thresholds |
| `src/ber/metric.py`, `src/ber/io.py` | macro F0.5; TSV reading/writing |
| `src/run.py` | stages norm / block / dense / train / test |
| `src/ce_data.py`, `src/ce_train.py` | cross-encoder data, training and scoring |
| `src/fit_pseudo_odds.py` | French decoy words from confident test predictions |
| `src/blend.py` | model × cross-encoder blend, per-country thresholds (used for the single-run files) |
| `src/final_build.py` | final decode: averages the two runs, cross-encoder ensemble, French cross-encoder, per-country thresholds, France rules A2 / HN_A / D / E; writes both output files |
| `src/artifacts/` | learned Indic→Latin dictionary, calibrated French word odds |

The exact commands are in `README.md`: normalise + block, dense retrieval, reverse name route + train, test (and France re-score), cross-encoders, then `final_build.py`.

### B. Additional Results

See `README.md` for the run times. The full experiment log, including the blocking-recall tables and every model run, is summarised above.
