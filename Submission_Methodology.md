# Team Yoddhas: Business Entity Resolution (final public leaderboard 0.989828)

**Team:** Jivansh Chawla (Team Leader), Tejashwini Gowda, Kavya Chetwani  
**Institution:** Thadomal Shahani Engineering College (TSEC), Mumbai

## 1. Methodology used

The pipeline has five stages:
1. Retrieval-based blocking.
2. A two-stage gradient-boosted pair classifier.
3. A cross-encoder ensemble on the uncertain pairs.
4. An exclusivity-aware decode with per-country thresholds.
5. Label-free France rules and targeted routes for records the decode leaves unclaimed.

The key ideas:
- **Decoys.** 26% of training S2/S3 records match no S1, and 72% of those are near-duplicates of an S1. They are generated "branches": the S1 name plus a qualifier word, with the house number shifted upward. We model this decoy signature explicitly with learned log-odds of the words a record adds or drops and a signed house-number shift.
- **Exclusivity.** Every record belongs to at most one S1. This drives both the competition features and a decode that gives each record to its best S1.
- **Test/train shift.** Test holds more lookalikes than training (5.5-5.8 vs 4.7 records per S1). We estimate each country's threshold on the test population itself: validation matches per S1 in each score band, divided by test pairs per S1 in that band. Public-leaderboard probes confirmed the choice.
- **France (no training labels).**
  - French decoy words are learned from confident test predictions and calibrated on US/India.
  - A cross-encoder is adapted to French by self-training.
  - Generator rules are read off the test data. A rule is admitted only if the same structural class is ≥ 94% true matches in the US/India labels.
- **Validation.** 10% of training S1 are held out (full candidate pool, singletons included) and scored with macro F0.5. The final US/India model reaches 0.9902.
- **Constraints.** No external data. All models are MIT/Apache-2.0, at most 7.6B parameters.

## 2. Candidate generation / blocking strategy

All blocking runs within country, on normalised text: accents stripped, OCR digits fixed, legal forms extracted (EN/FR/IN), street types canonicalised, house numbers extracted, and a learned Indic→Latin dictionary of 1,362 words applied.
1. **Word TF-IDF** on the name core plus address (tokens in more than 1% of records dropped), top 20 S1 per record via sparse_dot_topn.
2. **Dense retrieval.** multilingual-e5-small embeddings of "name | address", exact top 20 per S1 on GPU. The reverse direction adds the top 2 S1 for each record.
3. **Reverse-name route for records without an address.** Top 5 (and top 10 in a second run) S1 by character-3-gram TF-IDF on the name. Recall rises from 0.9849 to 0.9888.
4. **Learned filter.** A stage-1 XGBoost keeps each S1's top 15 with p ≥ 0.005. It removes ~88% of the pairs with no measurable recall loss.
5. **Targeted routes:**
   - India native-script: transliteration, TF-IDF, classifier.
   - R3 phonetic: Double Metaphone plus house number.
   - R6: same-name compact key.
   - France address-keyed: invented names or acronyms at the house number + street + city of exactly one S1.

Result:
- **Size.** 7.67M candidate pairs, 4.43 per S1 (median 4, 99th percentile 11, maximum 20). That is only 1.31× the 3.37 matches per S1 in the final output, and fewer pairs than there are S2/S3 records (0.77 per record).
- **Quality.** Blocking recall 0.989; the oracle macro F0.5 of the candidate set is 0.9964. 2.7% of S1 have no candidates.
- **Reduction.** 99.9999% against the 6.7 × 10¹² same-country pairs.

`candidate_pairs.tsv` is exactly what the matching stage runs on: the stage-1-filtered lists (scored by stage 2 and the cross-encoders) plus the accepted route pairs.

**Scalability.**
- Blocking runs per country and keeps only top-k lists, so memory is O(N·k).
- TF-IDF compares only pairs that share a token held by fewer than 1% of records.
- The exact dense top-k runs on one T4 at this size. At billions of records it would be swapped for an approximate nearest-neighbour index (FAISS IVF/HNSW) behind the same top-k interface.

## 3. Model architecture and feature engineering

**Features (~90):**
- **Name similarities:** token-set, partial and plain ratios, Jaro-Winkler, Levenshtein, IDF-weighted token overlap, acronym match, legal-form match.
- **Address similarities:** token-set ratio and IDF overlap; house-number equality, distance and ratio; empty-address flags.
- **Decoy signature:** learned log-odds (from a held-out training slice) of the words the record adds or drops, word counts, and the signed house-number shift. An EN→FR word map transfers the scores to French.
- **Retrieval and competition:** retrieval scores and ranks, plus the rank, gap and margin of each score within the S1's candidate list and among all S1s competing for the same record. The competition margins are the strongest features. Global name/address similarity ranks.

**Models:**
- **Two-stage XGBoost** (GPU, hist, lossguide, 255 leaves).
  - Stage 1 trains on 45% of training S1 and also serves as the candidate filter.
  - Stage 2 trains on a disjoint 45% and adds the stage-1 context features. 10% is held out for validation.
  - Easy negatives are subsampled (rate 0.2) with compensating weights.
  - Two full runs (reverse-name top 5 and top 10) are averaged.
- **Cross-encoder ensemble** on the uncertain band 0.02 ≤ p < 0.998: three fine-tuned multilingual-e5-base rounds and a Qwen2.5-7B-Instruct LoRA pair classifier (rank 16, sequence-classification head). Their logits are averaged and blended with XGBoost (weight 0.6). France uses an e5-small cross-encoder adapted by self-training on confident French test pairs.
- **Decode:** exclusivity, then thresholds US 0.80 / India 0.80 / France 0.87. A US/India S1 still empty afterwards takes its best unclaimed candidate when the blend is ≥ 0.5.

**Validation macro F0.5 (US/India):**

| Model | Validation F0.5 |
|---|---|
| TF-IDF blocking + LightGBM | 0.9660 |
| + dense retrieval, XGBoost | 0.9799 |
| Two-stage XGBoost + decoy features | 0.9865 |
| + cross-encoder | 0.9893 |
| + reverse-name route, run average, cross-encoder ensemble with Qwen | 0.9902 |

## 4. Other relevant information

- **Public leaderboard:**

| Step | Public LB |
|---|---|
| First submission | 0.9705 |
| Per-country thresholds | 0.9757 |
| Decoy features + e5-base cross-encoder | 0.9829 |
| Reverse-name route | 0.98553 |
| France generator rules | 0.98809 |
| Native-script + R3/R6 routes | 0.98905 |
| France invented-name route, Qwen, legal-form tie-break | 0.98941 |
| **Final** | **0.989828** |

- **France generator rules:**
  - **Suffix operation = match.** One S1 word is dropped and a French suffix is appended after the legal form (Fils, Groupe, Développement, Associés, France, Services, Cie).
  - **Decoys, dropped.** In-place category swaps (Club → Comité), and an added qualifier with the house number shifted upward.
  - **Name-replaced copies.** Invented names and acronyms keep the S1's address, so they are placed when the address identifies exactly one S1 (also by a sub-number such as "bis" or "B").
- **Tie-break.** For an address-less record whose name belongs to 2+ S1s, we choose the one S1 carrying the record's legal form. Train precision is 95-97%. France copies never swap legal forms.
- **What limits the score.**
  - For US/India, test claims per 1,000 S1 match training true-match densities within ±2 in every name × address class that has an address.
  - The gap left is address-less records whose name belongs to several S1s (~60 per 1,000 S1).
  - Nothing in the data resolves those ties beyond the legal form: row order, IDs, source split and sibling names all pick the owner at chance level on training labels.
- **Tried and dropped:** rare-key hash blocking; char-3-gram blocking; CatBoost and random forests; decoy duplication (it leaks the label); an expected-F0.5 decoder; self-training the pair model on pseudo-labels; synthetic French cross-encoders.
- **Compute:** Kaggle (T4/P100); AWS SageMaker ml.g5.12xlarge and ml.g4dn.16xlarge for the two reverse-name pipeline runs and ml.g5.12xlarge for the Qwen LoRA; Colab A100 for the first part of the Qwen training.
- **Code:** `code/business_entity_resolution/`. Its README has the exact commands, and they regenerate the final matching file byte-for-byte from the build.
