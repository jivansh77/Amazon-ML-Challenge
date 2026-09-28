# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Yoddhas  
**Team Members:** Jivansh Chawla (Team Leader), Tejashwini Gowda, Kavya Chetwani  
**Institution:** Thadomal Shahani Engineering College (TSEC), Mumbai  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve every Source 1 business against ~10M Source 2/3 records with a four-step pipeline. **Final public leaderboard: 0.989828** (first submission 0.9705).
1. **Blocking.** Three complementary retrievers (word TF-IDF, multilingual dense retrieval in both directions, and a name-only route for records without an address), unioned and cut by a learned first-stage model, plus a few targeted routes. This gives **4.4 candidates per S1** and keeps **98.9% of true matches**.
2. **Scoring.** A two-stage XGBoost model scores the candidates using similarity, competition and "decoy-signature" features. Two full runs are averaged.
3. **Uncertain cases.** An ensemble of fine-tuned cross-encoders (three multilingual-e5-base rounds and a Qwen2.5-7B LoRA pair classifier) re-scores the pairs the model is unsure about.
4. **Decoding.** An exclusivity-aware decode (every record goes to at most one S1) with per-country thresholds. It is followed by label-free France rules, a few extra routes, and tie-breaks for records the decode leaves unclaimed.

Three findings drove most of the gain:
- **The unmatched records are generated lookalikes ("decoys").** These are branches of the S1 business with an added qualifier word and a shifted house number. We model that signature explicitly.
- **France has no training labels.** We learn its decoy vocabulary (Ateliers, Sainte, city names, "France", …) from confident test predictions. We also read the generator's French operations off the test data (suffix words, category swaps, invented names and acronyms at the S1's address) and check every rule against the same structural class in the US/India labels. No external data is used.
- **Test is denser in lookalikes than training.** Thresholds and rules are checked against the test population itself: per-band precision estimates, and claimed-vs-true match densities per structural class.

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
- For France, label-free generator rules. Each rule is a structural class (e.g. "an acronym of the S1's name at the S1's house number + street + city"). It is accepted only if the same class is ≥ 94% true matches in the US/India training labels and France claims clearly fewer of it than US/India do.

---

## 3. Candidate Generation (Blocking)

All blocking runs **within country**. Every step is vectorised and chunked; the whole test set blocks in about 1.5 hours on a 4-CPU / 1-T4 Kaggle machine.

- **Blocking keys used:**
  1. **Word TF-IDF.** Name core (legal forms and titles removed) plus address tokens, over normalised text. Tokens in more than 1% of records are dropped. Top-30 cosine neighbours per S1 via `sparse_dot_topn`.
  2. **Dense retrieval.** `intfloat/multilingual-e5-small` (MIT, 118M parameters) embeddings of "name | address", exact top-k by blocked GPU matrix products. It handles transliterations, abbreviations and reordering that share no tokens.
  3. **Reverse dense.** The top-2 S1 for each S2/S3 record. This recovers name-only records whose generic names tie with many lookalikes in the forward direction.
  4. **Reverse-name route for records without an address.** For every S2/S3 record with an empty address, its top-5 S1 (top-10 in the second run) by character-3-gram TF-IDF on the name, within country. 64% of the validation misses that were never candidates were such records. This route raises blocking recall from 0.9849 to 0.9888.
  5. **Learned filter (stage 1).** The union of TF-IDF top 20, dense top 20, reverse top 2 and the reverse-name pairs is scored by the stage-1 XGBoost model. Stage 2 keeps each S1's top 15 with p₁ ≥ 0.005.
  6. **Targeted routes** for records the routes above never propose. Each is scored by its own classifier or rule, and its pairs are added to the candidate file:
     - India native-script route: Indic-script names transliterated, then name + address TF-IDF and a classifier.
     - R3: Double Metaphone of two name tokens + the house number.
     - R6: same-name compact key sharing a house number or city word.
     - France address-keyed routes: invented names and acronyms at the house number + street + city of exactly one S1; French suffix copies outside the candidate lists.

- **Candidate pairs generated:** **7,670,609 test pairs (4.43 per S1)**: the stage-1-filtered lists of the two runs plus the route pairs. This is the exact set that is scored, and it is what `candidate_pairs.tsv` contains. 46.9k S1 (2.7%) end up with no candidates.

- **Candidate-set size.**
  - 4.43 pairs per S1: median 4, 90th percentile 7, 99th percentile 11, maximum 20. By country: US 4.56, India 4.24, France 4.68.
  - That is only **1.31× the 3.37 matches per S1** we finally output.
  - The file has fewer pairs than there are S2/S3 records (0.77 per record).
  - Against the 6.7 × 10¹² same-country pairs, the reduction ratio is **99.9999%**, at 98.9% pair completeness on validation.
- **How the set is kept small.**
  - The retrievers' raw union is ~34 pairs per S1. The stage-1 model, a light XGBoost on cheap similarity and rank features, keeps each S1's top 15 with p₁ ≥ 0.005. This removes ~88% of the pairs, and recall moves from 0.9848 to 0.9849.
  - The targeted routes only propose records that no other retriever proposed, and a route pair enters the file only when it is accepted.
- **What `candidate_pairs.tsv` contains.** Exactly the pairs the matching stage runs on, written by `final_build.py` / `append_routes.py`. Every matched pair is in it.
  - The stage-1-filtered lists of the two runs, scored by the stage-2 XGBoost and the cross-encoders.
  - The pairs accepted by the targeted routes, scored by their own classifiers or rules.
- **Scalability.**
  - Blocking runs per country and keeps only top-k lists, so memory is O(N·k).
  - TF-IDF is a sparse matrix product with a 1% document-frequency cap: only pairs sharing a non-frequent token are ever compared (sparse_dot_topn keeps the top k per row). The reverse-name route does the same on character trigrams, for records without an address.
  - The dense search is exact top-k by blocked GPU matrix products. At this size, train and test together take about 2 hours on one T4, encoding included.
  - At billions of records the dense search would be replaced by an approximate nearest-neighbour index (e.g. FAISS IVF or HNSW) behind the same top-k interface. Nothing downstream changes.

- **How we ensured true matches were not lost.** Recall was measured on the full training ground truth for every route and cut-off:

| Candidates | Recall |
|---|---|
| TF-IDF top 30 | 0.953 |
| Dense top 30 | 0.958 |
| TF-IDF 20 ∪ dense 20 | 0.9795 |
| + reverse top 2 | 0.9848 |
| after the stage-1 filter (validation) | 0.9849 |
| + reverse-name route for address-less records (validation) | **0.9888** |

The filter removes ~88% of the pairs at no measurable recall cost. The oracle macro-F0.5 of the candidate set rises from 0.9953 to 0.9964 with the reverse-name route. The India native-script route, for example, finds the owner among its top 20 for 98.1% of owned training records; its added pairs are 97.7% precise on validation.

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
  - Trained in rounds on hard pairs of fresh training S1 that are never in the validation split. Each round is 160k S1 and 2.2M pairs (24% positives), 150–170 T4 minutes.
  - Each round starts from the previous checkpoint.
  - The first small-model version (e5-small) is kept for France, where it is adapted to French (see below).
  - It re-scores pairs in the uncertain band 0.02 ≤ p < 0.998 (19% of pairs).
  - It is blended on the logit scale with weight 0.6 for XGBoost (chosen on validation).
  - AUC on the uncertain band: XGBoost 0.965. The cross-encoder rises from 0.938 (e5-small) to 0.945 (e5-base) to 0.955 (e5-base, second round). Their blend reaches 0.975, so the two are complementary.
- **Final cross-encoder ensemble.** The submitted blend averages, on the logit scale, three e5-base rounds (AUC on the band 0.957 / 0.954 / 0.955) and a **Qwen2.5-7B-Instruct LoRA pair classifier**. Qwen is Apache-2.0 and 7.6B parameters: LoRA rank 16 with a sequence-classification head, trained on 1.9M hard pairs on 4× A10G. Its band AUC is 0.955, and adding it to the ensemble gives +0.00006 on validation.
- **Two-run average.** Two full pipeline runs, with the reverse-name route at top-5 and top-10, are averaged pair by pair before blending (+0.00013 on validation).
- CatBoost (0.9784) and random forest (0.9693) were weaker than XGBoost (0.9799) on identical features, and blending them did not help.
- All models are MIT/Apache-licensed and at most 7.6B parameters (limit: 8B).

**Threshold selection method:**
1. **Exclusivity.** Each S2/S3 record is kept only for its highest-scoring S1.
2. **Per-country thresholds, corrected for prior shift.** For each score band (0.5, 0.7, 0.8, 0.85, …, 0.99):
   - The expected true matches per S1 come from validation. France, which has no labels, uses the US/India average.
   - They are divided by the observed test pairs per S1 in that band. The result estimates the band's precision on test.
   - A pair improves macro F0.5 only if its precision exceeds ≈ F*/(1+β²) ≈ 0.78. Each country's threshold is the lowest band edge above which every band clears 0.78.
   - Result: US 0.80, India 0.80–0.85, France 0.97. Validation alone would pick 0.75–0.80 everywhere.
3. **Cross-check with a label-shift estimate.** Calibrate validation scores (isotonic), run EM on each country's test pair prior (Saerens et al.), and find where the corrected probability reaches 0.78.
   - It agrees for US/India (~0.74–0.78).
   - It suggests a lower France threshold (~0.84).
   - The two estimators assume different things about the unlabelled country. The France threshold was therefore settled with public-leaderboard probes that change only French predictions: 0.97 → 0.93 → 0.90 → 0.87, each an improvement.
4. **Final thresholds:** US 0.80, India 0.80, France 0.87.
   - An S1 still empty after decoding (US/India only) takes its best unclaimed candidate when the blend is ≥ 0.5. For an empty S1 a wrong add only costs when the S1 is a true singleton, so the break-even precision is ~0.5, not ~0.75.

**Transductive decoy words for unlabelled countries** (`fit_pseudo_odds.py`):
1. Confident test predictions (blended p ≥ 0.99 as match, ≤ 0.05 as non-match) serve as labels for France. The same word log-odds are fitted on near-duplicate names.
2. We keep the 101 words the training odds do not know and calibrate their scores. The same recipe is run on the labelled countries' test pairs (5.55M confident US + India pairs), and an isotonic map is fitted from those pseudo scores to the training scores of the same 437 words. On US test the pseudo scores correlate 0.69 with the training scores at about 2× magnitude. Test is then re-scored.
3. The words found are exactly the French decoy qualifiers: sainte, ateliers, a city name (lille, nantes, bordeaux, calais), jean, ei, departemental, residence, ehpad, musique, maison, france…
4. Generator typos (farmacie, clbu, teatre) come out as match-like.
5. **The effect on France:** 23k "+ France" lookalikes leave the candidate set, e.g. "club projets" vs "club projets france" goes 0.99 → 0.02. The words are applied to French S1 only; US/India scores are unchanged.

**Cross-encoder adapted to France** (self-training, `ce_data.py --parts pseudo` + `ce_train.py --init_dir`):
- The cross-encoder continues training for 30 minutes on 300k confident French test pairs (50/50) plus 13% of its original training pairs, at lr 2e-5.
- Validation AUC is unchanged (0.938), so there is no damage to US/India.
- On France it learns to reject swapped category words at the same address ("Biserica Loisirs" vs "Biserica Fetes"), the "+ Et Fils" branches, and same-name records at a different street.

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
| + second e5-base round | 0.9893 |
| + third e5-base round | 0.9895 |
| + reverse-name route for address-less records | 0.9901 |
| + top-5/top-10 run average, three e5-base rounds averaged | 0.9902 |
| **+ Qwen2.5-7B LoRA in the ensemble (final US/India model)** | **0.9902** (+0.00006) |

The France rules and routes cannot be measured on validation (no French labels). Each was checked against the same structural class in the US/India labels, and they were measured together on the public leaderboard.

- **Public leaderboard:**

| Submission | Public LB |
|---|---|
| First model | 0.9705 |
| + thr 0.90 + exclusivity | 0.9730 |
| + per-country thresholds | 0.9757 |
| + decoy features + e5-base cross-encoder | 0.9829 |
| + France fix | 0.9831 |
| France threshold 0.93 (on the baseline) | 0.9830, vs 0.98286 at 0.97 |
| French decoy words + French-adapted cross-encoder | 0.98305 |
| Third e5-base round, France threshold 0.90 | 0.98443 |
| France threshold 0.87 | 0.98452 |
| Reverse-name route for address-less records | 0.98553 |
| France suffix operation + changed-house-number rules | 0.98667 |
| France generator rules v2 (suffix position, category swaps, out-of-candidate suffix copies), run average + 3-CE ensemble | 0.98809 |
| India native-script route + R3/R6 routes | 0.98905 |
| France invented-name route, Qwen in the ensemble, legal-form tie-break (2–3 same-name S1s) | 0.98941 |
| **Final: France acronym / per-city / shared-address routes, first match for empty S1, extended legal-form tie-break** | **0.989828** |

**Unseen-country stand-in** (train on US only, validate on India):
- F0.5 falls from 0.984 to 0.933. That confirms France's implied ~0.92 is a transfer problem, not bad luck.
- A US-only cross-encoder recovers 2.3 points (0.957).
- Self-training on pseudo-labels recovers nothing (0.9376 → 0.9358).
- The same stand-in calibrates the value of an added pair against its precision (break-even ≈ 0.70). It put France's band [0.93, 0.97) at only ~0.75 precise. The leaderboard showed the stand-in was too pessimistic: the 0.87–0.97 French additions were ~0.80–0.90 precise, so France ends at 0.87.
- **Common false positives (wrong merges):**
  - Generated branches that change a category word at the same address ("Rayon Comite SAS" vs "Rayon Musique SAS").
  - Branches that add a qualifier the model has not seen.
  - Same-name records at a different street with the same house number.
  - Records with an empty address and an exact name, which could be a branch or the entity itself.
- **Common false negatives (missed matches):**
  - Brand/trade-name records that share only the address ("Pessac Ecole SAS" vs "NOVIQUO").
  - Heavy abbreviations or initials.
  - Empty-address name-only records with generic names (the largest blocking miss: 24% of TF-IDF misses).
  - Indic-script names without a dictionary entry.
- **Where the validation loss sits** (final US/India model, validation S1 with candidates, F0.5 = 0.9901):

| Error type | Share of the loss | Note |
|---|---|---|
| S1 with some matches found and some missed | 73% | mostly address-less records whose name belongs to several S1 |
| Matched S1 predicted empty | 19% | |
| Wrong merges | 8% | 254 on S1 with matches, 13 on singletons |

  The final system is recall-limited, and the remaining recall loss is structural. We compared claimed pairs per 1,000 test S1 with true pairs per 1,000 training S1, by name relation × address relation. Every US/India class with an address agrees within ±2 per 1,000. The only gap is address-less records: about 60 per 1,000 S1, almost all names shared by 2+ S1.
  - On the training labels, row order, IDs, the source split, the owner's other copies and the full name all pick the owner at chance level.
  - Only the legal form separates the owners, and we use it (section 6.4).

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

**Decoy duplication in detail:**
- Virtual copies were made only of records that match no S1. A twin therefore identified a decoy, which leaks the label.
- It looked +0.0016 better on validation.
- On test it matched 35k S1 that the base model rejects: only 0.9% of S1 predicted empty, against 5.6% singletons. Never submitted.

---


## 6. Final submission (`avg_ce4_v17c`, public LB 0.989828)

The submitted file builds on the pipeline above with four additions. Public LB history: `dec_cebase` 0.982855 → reverse-name
route 0.98553 → France generator rules 0.98809 → native-script and R3/R6 routes 0.989052 → France invented-name rule 0.989412
→ **final 0.989828**. The final step (+0.0004) adds label-validated France address routes, first matches for empty S1s and the
extended legal-form tie-break. The submitted file has 5,839,939 matches: 3.37 per S1 (US 3.39, India 3.38, France 3.32), and
5.8% of S1 are predicted empty.

**6.1 Reverse-name route and run average.** 64% of the validation misses that were never candidates are address-less records.
A `namerev` stage adds, for every S2/S3 record without an address, its top-k S1 by character-3-gram TF-IDF on the name. Blocking
recall rises from 0.9849 to 0.9888. Two full runs (top-5 and top-10) are averaged pair by pair.

**6.2 Cross-encoder ensemble.** The uncertain band is scored by three multilingual-e5-base rounds (base3, base4, base4b) and a
Qwen2.5-7B-Instruct LoRA pair classifier (Apache-2.0, 7.6B parameters). Their logits are averaged and blended with the XGBoost
logit (weight 0.6). France uses the French-adapted e5-small cross-encoder instead. Thresholds: US/India 0.8 (validation optimum),
France 0.87 (leaderboard probes 0.97 → 0.93 → 0.90 → 0.87).

**6.3 France generator rules (label-free).** France has no labels. We read the generator's operations off the test data and
checked every rule against the same structural class in US/India labels:
- **Suffix operation = match.** One S1 word is dropped and a French suffix is appended after the legal form (Fils, Groupe,
  Développement, Associés, France, Services, Cie). Rules A2P / A2F / A2N / E2 / OOC.
- **Decoys, dropped.** In-place category swaps (Club → Comité, 164-word vocabulary; rule DP). Changed house numbers with an added
  or swapped qualifier and a small upward shift (HN_A; the kept-qualifier profile is re-added by HNK).
- **Name-replaced copies with the address kept.** An invented single-token name or the S1's acronym. Train precision 94-100%;
  France's blocking missed most of them. Placed when the address (house number + street + city) identifies one S1 (per city,
  typo'd street, shared address with one matching acronym, or a matching unit / sub-number such as "bis", "ter", "B").
- **How a rule is admitted.** A rule adds (or drops) one structural class. Its true-match rate in the US/India training labels must be
  high: 94-100% for adds; decoy classes 0-7% for drops. France's claim rate for that class must also fall clearly below the
  US/India rate (e.g. acronyms at the S1's exact address: 100% true in train; claimed by US/India 99.8%, by France 83%).

**6.4 Extra routes and tie-breaks (records the decode leaves unclaimed).**
- India native-script route: transliterate, then name + address TF-IDF and a classifier; 97.7% precise on validation.
- Kavya's R3 phonetic route (Double Metaphone + house number) and R6 same-name compact-key route, each scored by the e5-base
  cross-encoder + XGBoost (`src/r3_candidates.py`, `r3_score.py`, `r3_build.py`, `r6_candidates.py`, `r6_score.py`,
  merged by `r3r6_merge.py`).
- First match for S1s still empty (US/India, blend ≥ 0.5; skips address-less same-name ties).
- **Legal-form tie-break.** For an address-less record whose name belongs to 2+ S1s, the one S1 carrying the record's legal
  form. Train precision 95-97% / 90-91% / 84% with 0 / 1 / 2 same-name S1s without a legal form.
  - France copies never swap the legal form (0.02% of claimed same-name pairs vs 2.6% in the US). There, the one twin without a
    form also takes a record whose form no twin has.
  - Only records the stage-2 model scores ≥ 0.5 are added. On validation, records the model had rejected are 88% right above
    that score and 33% below it.

**6.5 What limits the score.** For US/India, test claims per 1,000 S1 match train's true matches per 1,000 S1 within ±2 in every
name × address class that has an address. The remaining gap is address-less records whose name belongs to several S1s
(≈ 60 per 1,000 S1). On train we checked row order, IDs, the source split, sibling-name similarity and full-name identity; all
are at chance, and only the legal form separates the owners.

## 7. Conclusion

Entity resolution at this scale is won in the tails. Candidates are cheap to retrieve: three routes reach 98.9% recall with 4.4 pairs per S1. The hard part is telling a business from its generated branches, especially in a language without labels. Several steps each gave measurable gains:
- modelling the decoy signature;
- a cross-encoder ensemble for the uncertain band;
- correcting thresholds for the test population's decoy density;
- for France, reading the generator's operations off the test data and admitting a rule only when the same class is clean in the labelled countries.

The public leaderboard rose from 0.9705 to 0.989828. What remains is mostly structural. A record without an address whose name belongs to several S1s cannot be attributed from the data, except by its legal form. Validation scores must be read with the test prior shift in mind.

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
| `src/blend.py` | model × cross-encoder blend, per-country thresholds, writes both output files |
| `src/llm_ce.py`, `src/sagemaker_llm.py` | Qwen2.5-7B LoRA pair classifier (training / scoring) |
| `src/sagemaker_pipeline.py` | the AWS run of the full pipeline with the reverse-name route |
| `src/final_build.py` | final decode: run average, CE ensemble, French CE, thresholds, France rules, route files, first match |
| `src/native_route.py` | India native-script route |
| `src/france_name_replaced.py`, `src/france_thr_safe.py`, `src/france_shared_addr.py` | France name-replaced / threshold-safe / shared-address rules |
| `src/legal_tie.py` | legal-form tie-break for address-less same-name ties |
| `src/append_routes.py` | appends the v16 / v17 route files to the decoded build |
| `src/r3_candidates.py`, `src/r3_score.py`, `src/r3_build.py`, `src/r6_candidates.py`, `src/r6_score.py`, `src/r3r6_merge.py` | R3 phonetic and R6 same-name compact-key routes: candidates, scoring, build, merge |
| `src/artifacts/routes/` | every route / rule pair file used by the submitted build |

The exact commands are in `README.md`: normalise + block, dense retrieval, train (with the reverse-name route), test, cross-encoders, French words + re-score, then `final_build.py` and `append_routes.py` for the submitted file (section "Final build").

### B. Additional Results

See `README.md` for the run times. The full experiment log, including the blocking-recall tables and every model run, is summarised above.
