# Plan: Business Entity Resolution

This plan is based on EDA of the real dataset (Kaggle `jvmusic/ml-challenge`). All numbers below were measured on the actual files.

## 1. What we have to do

For every **Source 1 (S1)** record in test, output the set of **S2/S3** record IDs that describe the same business, or an empty set.
- **Score:** macro F0.5, computed per S1 and then averaged.
  - A singleton scores 1.0 if we predict empty, and 0.0 if we predict anything.
  - Precision counts twice as much as recall.
- **Deliverables:**
  - `matching_results.tsv`, which is scored.
  - `candidate_pairs.tsv`, the exact set the model scored. It must be a superset of the matches.
  - Runnable code, a README and requirements.
  - The filled-in documentation template.
- **Rules:**
  - No external data or lookups.
  - The final model must be MIT or Apache-2.0 licensed, with at most 8B parameters.
  - Country is an open set: the test data includes **France**, which has no training labels.

## 2. EDA findings that shape the design

| Fact | Number | Consequence |
|---|---|---|
| Train size | S1 2.21M, S2 5.03M, S3 5.29M | Scale matters. Use polars, vectorised rapidfuzz and FAISS. Avoid Python loops over pairs. |
| Test size | S1 1.73M (India 810k, US 663k, **France 259k**), S2 4.89M, S3 5.08M | France is 15% of the test S1 records, so it strongly affects the leaderboard. |
| Singleton rate (train) | **5.6%** (identical for US and India) | Most S1 records do have matches, so recall matters a lot. Singletons still need a proper no-match gate. |
| Matches per S1 | mean 3.46, 0–11. S2: 0–5, S3: 0–6 | Many-to-one: S2 and S3 each contain duplicates of the same business. We output sets, not a single best match. |
| S2/S3 record linked to >1 S1 | **0** | Strict exclusivity. Each S2/S3 record belongs to at most one S1, so assignment can be done **per S2/S3 record**. |
| S2/S3 records matched to nothing | ~26% | Many distractors. |
| Country mismatch inside true pairs | **0** | Safe to block within the same country label. This is a data-driven block, not a hard-coded list. |
| Test S2+S3 per S1 | 5.8 (test) vs 4.7 (train) | The test pool is about 23% denser. Expect more distractors, so thresholds tuned on train may be slightly optimistic. |
| Train/test overlap | 0 IDs, 0 exact records | No leakage to exploit. |
| Exact normalised-name joins | 20.6M pairs, only 1.67M true | Generic names such as "Primary Care Group" repeat across cities. The **name alone is useless without the address**. |
| True pairs with name `token_set_ratio` < 50 | **10%** | These are the hard positives (next table). |
| True pairs with address sim < 50 | 4.6% (mostly a null address) | |
| True pairs where **both** are < 50 | 0.04% | Name and address cover for each other, so the model must learn to trust whichever one is informative. |
| Indic-script names (India S2/S3) | **18%** (Devanagari, Bengali, Telugu, Tamil, Kannada, Gurmukhi) | Transliteration or a multilingual encoder is essential for India. |

Hard-positive types among the low name-similarity pairs:
- Indic script: `Tirupati Tech Private Limited` ↔ `तिरुपति टेक प्राइवेट लिमिटेड`
- Random trade names with the same address: `Kapishwar Trading Pvt Ltd` ↔ `Belozeta`
- Acronyms: `Physical Therapy Crystal Associates` ↔ `PTCA`
- Websites: `Heritage Foundation LLC` ↔ `heritagefoundation.com - 7334891395`

Other noise types:
- DBA prefixes, e.g. `Solectosol t/a Secure Quetta LLC`.
- Titles, e.g. `Mr`, `Dr`, `Shri`, `M/s`.
- Injected accents and OCR swaps, e.g. `5ecure`, `D0ts`, `ÍNFÓRMATION`.
- Junk tags, e.g. `(ID: 93609)`, `#58665`, `<<`, `--`.
- Shuffled tokens.
- State names in native script or as abbreviations, e.g. `UP`, `उत्तर प्रदेश`, `Iowa`/`IA`.
- Shuffled address components.
- Perturbed house numbers, e.g. `12781` ↔ `1258`, `46` ↔ `96`, `0327` ↔ `327`.

Decoys:
- Most same-name non-matches are at a **completely different city or street**.
- France test data covers only a few cities (Bordeaux, Lille, Nantes, Pessac…) with generic names ("Sociale Ecole SASU"), so the street and house number will carry most of the signal there.
- French abbreviations appear: `R.` → Rue, `Av`, `Bd`, `Allée`. So do French legal forms: SARL, SAS, SASU, EURL, SCI, SA.

## 3. Approach

### 3.1 Normalisation, independent of country
- NFKD accent stripping and casefolding.
- Map OCR confusables (0→o, 5→s, 1→l) to produce a "shape" variant.
- Strip junk tags (IDs, `#nnn`, phone numbers, brackets).
- Split out the DBA part (`t/a`, `dba`) and use **both** sides.
- Strip titles and legal forms. Keep the legal form as a separate field for a feature.
- Split website domains into word candidates.
- Build acronyms.
- **Transliterate Indic scripts to Latin** with a rule-based, MIT-licensed library. We can also learn the mapping from train pairs.
- Parse the address into house number(s), street tokens, city, state (canonicalised) and postcode.
- Use one abbreviation map covering EN and FR (Rd/St/Ave/R./Av/Bd…). It is hand-written, so it is not external data.

### 3.2 Blocking: several routes, unioned, run within the same country
1. **Address key:** (house number, rarest street token) and (street tokens, city). Blocking on the address is what recovers the random-trade-name, Indic and acronym cases.
2. **Name TF-IDF:** char 3–4-gram TF-IDF on the normalised and transliterated name, with sparse top-K restricted to the same city/state where possible and globally otherwise.
3. **Dense kNN:** multilingual embeddings (`multilingual-e5-small`/`base`, MIT) of `name | address`, with FAISS top-K. This runs on a Kaggle GPU; ~10M records takes well under an hour.
4. Take the union and cap it per S1.

Target recall on validation: 99% or more. We will report recall@K and the reduction ratio for the documentation.

### 3.3 Pair scorer: LightGBM
Features:
- Name similarities (ratio, token_set, partial, Jaro-Winkler, char TF-IDF cosine) on each name variant: raw, clean, transliterated, DBA side, acronym, domain.
- IDF-weighted token overlap, which separates "Sharma Traders" from "Gupta Traders".
- Address similarities, plus exact or near house-number match, street overlap, city/state/postcode agreement, and missing-field flags.
- Dense cosine.
- Source (S2 vs S3).
- **No country feature.** All features are pure similarities, so they transfer to France.

### 3.4 Context and relational features: stage 2
Using out-of-fold stage-1 scores, add:
- The rank and margin of this candidate within its S1.
- The rank and margin of this S1 among all S1s competing for the candidate. Exclusivity makes this very strong.
- Mutual-best flag.
- How many strong candidates the S1 has.
- Support between S2 and S3, i.e. whether this candidate is also similar to the S1's other confident matches.

### 3.5 Decision layer, optimised directly for macro F0.5
1. **Assign each S2/S3 record to at most one S1**: its argmax, if the calibrated probability is high enough. This enforces exclusivity.
2. For each S1, pick the set that maximises **expected F0.5**, with "empty" as an explicit option, using calibrated (isotonic) probabilities.
3. Tune this against a simple threshold baseline on the exact metric, and keep whichever wins.

### 3.6 Optional boost: cross-encoder
If time allows, add a multilingual cross-encoder (`xlm-roberta-base` or `mdeberta-v3-base`, MIT) fine-tuned on hard pairs. Run it only on the uncertain band, and feed its score to the GBDT as a feature.

### 3.7 France without labels
- Use only country-agnostic features, plus FR abbreviations and legal forms in the normaliser.
- **Leave-one-country-out** check: train on US and score India, and the reverse. This measures how well features transfer.
- Look at the **predicted** matched fraction and score distribution for France in test, and compare them with US and India.
- If allowed, fit unsupervised statistics (IDF) on test records so French tokens get proper weights.

### 3.8 Validation
- Hold out about 10% of S1 records as validation, but always search the **full** S2/S3 pool, as the test setup does.
- Report macro F0.5 overall, for singletons vs matched records, and per country.
- A stress test for the denser test pool: remove some S1 records from the validation set so that their matches become distractors.

## 4. Compute
- **Kaggle notebooks:** 30 GB RAM, a T4/P100 GPU, and 12 h sessions. Used for embeddings, the full test run and heavy experiments.
- **Local (this container):** 4 CPU and 15 GB RAM. Used for EDA and development on samples.

## 5. Timeline (5 submissions per day, 15 in total)
- **Day 1:**
  - Build the end-to-end baseline: normalise → address + TF-IDF blocking → LightGBM → threshold.
  - Set up the validation harness.
  - Make the first submission.
- **Day 2:**
  - Add transliteration, dense blocking, stage-2 relational features and the F0.5 decoder.
  - Run the LOCO checks for France.
- **Day 3:**
  - Add the cross-encoder only if it helps on validation.
  - Ensemble and calibrate.
  - Build the final package, README and documentation.
