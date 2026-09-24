# Research Brief: Business Entity Resolution (Amazon ML Challenge 2026)

Four parallel research tracks: past ER winners, deep entity matching, collective/graph ER, and F-measure optimisation plus France generalisation. They are merged here into ranked idea cards. `f05_decoder.py` next to this file is a tested implementation of the decision rule in card D1.

**How to use this:** hand one card to one agent session. Each card says what to build, why it helps, and how it can go wrong. Every run reports end-to-end macro F0.5 on the frozen folds **and** the leave-one-country-out (LOCO) score.

---

## 0. The 8 things that matter most

1. **The decision layer is the biggest lever.** Singletons score 1 or 0 with nothing in between. Each entity's prediction set should be chosen by maximising **expected F0.5, with "empty" as an explicit option** (card D1). Most teams will use one global threshold. Shopee-style threshold tuning alone gave +0.7 to 1.3 F1 in that competition.
2. **Blocking caps recall.** Use several routes and union them: char-n-gram TF-IDF/BM25, plus multilingual dense retrieval, plus a postcode block. Sparkly (lexical BM25) beats most deep blockers on its own, and adding dense retrieval on top gives large recall gains. The Foursquare champion reached a 0.979 recall ceiling with 2 routes × 100 candidates.
3. **Use a GBDT with per-S1 aggregate features, then stack a cross-encoder into it.** On Foursquare this went CatBoost 0.878 → mDeBERTa 0.907 → ensemble 0.911.
4. **Second-order (relational) features in a stage-2 GBDT.** Reverse rank, margin to the second-best S1, mutual top-1, and triangle support. Most of the Kaggle winners' post-processing gain came from these.
5. **Candidate-side exclusivity.** Each S2/S3 record goes to at most one S1. **Never use connected components or transitive closure**: they were consistently the worst in the FAMER benchmark, and one bad edge zeroes two entities.
6. **France is unseen.** Features must be pure similarities with no country tokens. Select everything on the **worse** of the two LOCO directions (US→IN, IN→US). Multilingual backbones only.
7. **Licensing traps.** Qwen2.5-**3B** (non-commercial), jina-embeddings-v3 (CC-BY-NC) and Jellyfish (CC-BY-NC) are out. Unidecode is **GPL**, so use stdlib NFKD or anyascii (ISC) instead. libpostal ships OpenStreetMap-trained data, which probably counts as external data.
8. **Calibration is what makes D1 work.** Fit isotonic regression on out-of-fold scores, and check it per country.

---

## 1. EDA gates: answer these first, because they decide which cards run

| Question | If yes / high | If no / low |
|---|---|---|
| What % of S1 entities have **no** match (overall and per country)? | The NIL classifier (D2) and empty-set decoding matter most | Recall matters more |
| Does any S2/S3 record appear under **more than one** S1 in ground truth? | Drop hard exclusivity (D3) and use soft margin features | Enable **D3** |
| Do S1s have **2+ matches from the same source**? | Many-to-one: skip UMC/Király | One-to-one per source: try **UMC** (C5) |
| What is the S2/S3 pool size per S1 in train vs test? | Test is denser → CV precision is optimistic; stress-test K | Folds are representative |
| Is there **exact overlap** between train and test records? | Ask the organisers before using it (the Foursquare champion gained 0.900 → 0.971 this way; it's a rules question) | Ignore |
| Do S2 and S3 records matched to the same S1 also look like each other? | Enable triangle rescue/veto (C4) | Skip |

---

## 2. Tier 1: build on Day 1 (the foundation that every other card plugs into)

### V1. Validation protocol [PRIORITY]
- **What:** Two schemes, both frozen.
  - (a) 5-fold grouped by S1. Each S2/S3 record goes in the fold of its matched S1; unmatched ones are spread randomly.
  - (b) LOCO: train US → validate IN, and train IN → validate US.
  
  Choose features, thresholds and calibration by the **worse** LOCO score. Report F0.5 split into entities with matches vs without.
- **Why:** LOCO is the only offline proxy for France. It also shows up features that don't transfer.
- **Cost:** 1h, CPU. **Pitfall:** US and IN are both mostly English, so LOCO won't reveal problems with accents or French word order (see card A3).

### N1. Country-agnostic layered normalisation [PRIORITY]
- **What:** Keep several variants of each field: raw, normalised, suffix-stripped, token-sorted.
  - Unicode: `unicodedata.normalize('NFKD')` and drop combining marks, or `anyascii` (ISC). Then casefold, map `&`→`and`/`et`, and strip punctuation.
  - Legal forms: `cleanco` (MIT) already knows SARL/SAS/SASU/EURL/SNC/SCI/SA, Pvt Ltd, LLC and so on. Strip them **but** keep a `legal_form_equal` / `legal_form_missing` feature, because "X Ltd" and "X Inc" can be different businesses.
  - Address abbreviation map:
    - EN: rd, st, ave, blvd, ln, nr, opp
    - FR: av→avenue, bd→boulevard, pl→place, imp→impasse, all→allée, fg/fbg→faubourg, ch→chemin, rte→route, r→rue
  - Landmark phrases ("near …", "opp …", "behind …"): strip them into a separate field.
  - Regex extraction: postal code (`\b\d{6}\b` or `\d{3}\s\d{3}` for IN, `\b\d{5}(-\d{4})?\b` for US/FR), house number, and city token.
- **Why:** This addresses every noise type listed in the problem statement, plus French at test time.
- **Cost:** 2 to 4h, CPU. **Pitfall:** Ask the organisers whether hand-written dictionaries count as "external data" (they almost certainly don't, but get it confirmed).

### B1. Multi-route blocking with union [PRIORITY]
- **What:** Run these routes and union them per S1. Store `route_bitmask` and the rank within each route as features.
  - (a) char 3-to-5-gram TF-IDF cosine (or BM25, Sparkly-style) on normalised name, and on name+address, top-K=50
  - (b) dense retrieval on `"query: {name} | {address}"` with multilingual-e5-base (MIT) or bge-m3 (MIT, dense+sparse), top-K=50, using FAISS
  - (c) exact postcode or city block, ranked by name token Jaccard, K=20
  - (d) optionally, a phonetic or transliteration key
  
  Report **recall@K per route and for the union**, and average candidates per S1 (the organisers look at recall ceiling and reduction ratio). Target ≥97 to 99%.
- **Sources:** Sparkly (VLDB'23, 92.5 to 100% recall at k≤50, beats DeepBlocker): https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf. DeepBlocker (deep+lexical union): https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf. Foursquare champion: https://jishu.proginn.com/doc/7780647801f904d12. 7th place (5 routes): https://future-architect.github.io/articles/20220720a/
- **Cost:** 3 to 6h. CPU, with a GPU for the embeddings (minutes on a T4).
- **Pitfalls:** Don't use a hard country filter, since the country label may be noisy. A larger K means more pairs to score, so find the knee of the recall/K curve. Fitting TF-IDF on train+test is a rules question (see §6).

### M1. Two-stage GBDT pair scorer [PRIORITY]
- **What:**
  - **Stage 1:** a cheap LightGBM that prefilters to about 20 to 40 candidates per S1.
  - **Stage 2:** LightGBM/CatBoost on about 100 features, each computed on every normalised variant:
    - Levenshtein, normalised Levenshtein, Jaro-Winkler, token_set_ratio, partial_ratio, difflib ratio, LCS, token Jaccard, char-TF-IDF cosine, dense cosine
    - length ratio, postcode equal/prefix/missing, house-number equal/missing, legal form equal
    - IDF-weighted overlap, and rare-token vs common-token-only overlap (to catch "Sharma Traders" vs "Gupta Traders")
    - **per-S1 aggregates:** max, min and mean of the key similarities over that S1's candidates; rank; difference and ratio to the best; candidate count
  
  Weight negatives about 1.5 to 2× (F0.5 is precision-heavy).
- **Sources:** https://future-architect.github.io/articles/20220720a/ , https://github.com/hobbitlab/Foursquare-Location-Matching , https://github.com/TheoViel/kaggle_foursquare . Splink term-frequency adjustment: https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html
- **Cost:** 4 to 6h, CPU (use `rapidfuzz`, which is fast).
- **Pitfalls:** Stage 2 must train on **out-of-fold** stage-1 scores. Never use country or raw tokens as features. The 7th-place team used num_leaves=4096 on millions of pairs; ours is smaller, so start around 63 to 255.

### D1. Expected-F0.5-optimal set decoder [PRIORITY] → `f05_decoder.py`
- **What:** For each S1, sort the calibrated probabilities. Compute E[F0.5] for each top-k set (k=1..n) and compare against the empty set, which scores exactly P(no match). Output the argmax. This is exact under independence: the optimal set is always a top-k (Ye et al. 2012, Thm 9). If D2 gives a separate `q0` = P(no match), the expected scores are rescaled. The implementation is tested against brute-force enumeration over all subsets and matches it exactly. It runs in about 8ms per entity at n=50, so cap n at the top-20 by probability to speed it up.
- **Intuition it encodes:** a single candidate needs p>0.5 to be output. For p=(0.9, 0.3) it outputs only the first. For p=(0.9, 0.8) it outputs both.
- **Sources:** Ye et al. ICML'12 https://arxiv.org/pdf/1206.4625 ; GFM (Dembczyński et al. NeurIPS'11) https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf
- **Cost:** Already written. Plugging it in takes about 30 minutes.
- **Pitfall:** It is only as good as the calibration (D2). Always compare it against the tuned-threshold fallback (D4) on out-of-fold and LOCO, and keep the winner.

### D2. Calibration + NIL ("has any match") classifier [PRIORITY]
- **What:**
  - Isotonic regression on out-of-fold pair scores (use Platt if there are fewer than about 1k rows). Break ties caused by isotonic steps using the raw score.
  - A per-S1 GBDT with target "has ≥1 true match". Features: top-1 p, the top1−top2 margin, number of candidates with p>0.3 and p>0.5, the top candidate's reverse rank, whether the top candidate prefers another S1, candidate count.
  - Calibrate it, and pass it to D1 as `q0`.
  - Check the reliability curves per country and on LOCO.
- **Sources:** Niculescu-Mizil & Caruana 2005, https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf. The entity-linking NIL-detection survey, https://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/TKDE14-entitylinking.pdf
- **Cost:** 2h, CPU. **Pitfall:** The singleton base rate may differ in France. Don't hard-code the prior. Consider a single temperature/shift term tuned on LOCO.

### D3. Candidate-side exclusivity (only if EDA confirms it holds)
- **What:** For each S2/S3 record c, keep only the edge to `argmax_s P(s,c)`, and only if P ≥ t **and** the margin over the second-best S1 is ≥ δ. Each S1 still collects 0..many records. If the margin is < ε and the addresses match (e.g. chain branches), keep both. Run this **before** D1.
- **Source:** Clean-clean bipartite matching benchmark (BMC), https://arxiv.org/pdf/2112.14030v3
- **Cost:** 1h, CPU. **Gain:** Medium to high on precision, because one shared false positive hurts two entities.

### D4. Tuned dynamic-threshold fallback (a baseline to beat)
- **What:** Predict empty if p1 < t0 (or q0 > t_e). Otherwise keep top-1, and add candidate j if p_j > t2 **and** p_j > r·p1. Grid-search on out-of-fold data with the exact metric, and pick using LOCO.
- **Source:** https://github.com/jingxuanyang/Shopee-Product-Matching. Note that Shopee's "min2" forces at least one match; **reverse that here** into a no-match gate.
- **Cost:** 1h.

---

## 3. Tier 2: Day 1 night to Day 2 (exploit and diversify)

### M2. Relational / second-order features and a stage-2 retrain [PRIORITY]
- **What:** From out-of-fold stage-1 probabilities, compute for each pair (s,c):
  - reverse rank of s among all S1s competing for c
  - mutual-top-1 flag
  - `P − max P(other S1, c)` and `P − max P(s, other c)`
  - number of S1s within 0.05 of P
  - c's best score and entropy across S1s
  - number of s's candidates with p>0.5
  - strong/normal/weak link type (CLIP-style: best for both ends / best for one / best for neither)
  - triangle support `max_{c'} min(P(s,c'), P(c,c'))` across S2↔S3
  
  Retrain the GBDT on the original features plus these.
- **Sources:** Shopee 4th place ("Rank2" reciprocal matching) https://masatakashiwagi.com/blog/kaggle-shopee-solution/ ; Foursquare 2nd place post-processing 0.907 → 0.946 https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ ; CLIP https://old.dbs.uni-leipzig.de/file/eswc_0.pdf
- **Cost:** 2 to 4h, CPU. **Pitfalls:** Leakage if these are computed on in-fold scores. Keep K the same between train and test.

### M3. Ditto-style cross-encoder, stacked into the GBDT [PRIORITY]
- **What:** Fine-tune **mDeBERTa-v3-base (MIT)**, xlm-roberta-base/large (MIT), or bge-reranker-v2-m3 (Apache) as the starting point.
  - **Input:** `COL name VAL … COL addr VAL … [SEP] COL name VAL … COL addr VAL …`. Optionally add a few key tabular similarities as text (the Foursquare champion did).
  - **Training:** max_len 128, lr 2 to 3e-5, batch 32 to 64, 3 to 4 epochs, binary cross-entropy. Negatives are **hard negatives from our own blocker's top-10**, at about 2:1 negatives to positives.
  - **Augmentation:** span delete, token shuffle, column drop, A/B swap. At test time, average both orderings. Optionally add FGM adversarial training and EMA (the champion used both).
  - Feed its out-of-fold probability into the GBDT as a feature. Do not use it as the final decider.
- **Sources:** Ditto https://arxiv.org/abs/2004.00584 (up to +9.8 F1, 96.5 F1 on company matching). Foursquare champion (0.878 → 0.911 with the ensemble) https://jishu.proginn.com/doc/7780647801f904d12 . AnyMatch data curation (keep GBDT errors, 2:1 ratio, attribute-level pairs) https://arxiv.org/html/2409.04073v1
- **Cost:** 1 to 3 GPU-hours on a T4 per fold set, 6 to 10h of total effort.
- **Pitfalls:** Pretrained matchers lose 22 to 61% F1 on unseen entities, so **validate on LOCO**. mDeBERTa fp16 on a T4 can produce NaN loss; switch to fp32 or XLM-R if that happens. Train 5 fold models for out-of-fold predictions, or use a single holdout plus refit if time is tight.

### M4. Zero-shot rerankers as features (robust for France)
- **What:** Score every blocked pair, without fine-tuning, with **Qwen3-Reranker-0.6B (Apache)**, using the instruction "Judge whether the Document is the same business entity as the Query", and/or bge-reranker-v2-m3. Add the scores as GBDT features.
- **Why:** These aren't fitted to US or India, so they should behave the same on France.
- **Sources:** https://huggingface.co/Qwen/Qwen3-Reranker-0.6B , https://huggingface.co/BAAI/bge-reranker-v2-m3
- **Cost:** About 1 T4-hour per ~1M short pairs. **Pitfall:** These are trained for relevance, not identity, so they score "same chain, different branch" pairs high. Pair them with the address features.

### B2. Contrastive bi-encoder for blocking (only if the recall ceiling is the bottleneck)
- **What:** Label connected components of known matches as entity IDs. Fine-tune multilingual-e5-base or bge-m3 with MultipleNegativesRanking/SupCon loss (temperature 0.07), using hard negatives from the same city/postcode taken from BM25 top-k. Then run FAISS kNN and re-rank by the average of neural similarity and Jaccard.
- **Sources:** R-SupCon https://arxiv.org/pdf/2202.02098.pdf (+3 to 4 F1 over Ditto). SIGMOD'22 winner https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/
- **Cost:** 1 to 2 GPU-hours.

### C4. S1-anchored triangle rescue and veto
- **What:** Score S2↔S3 pairs with the same model.
  - **Rescue:** add (s,b) if P(s,a)≥t_hi, P(a,b)≥t_hi and P(s,b)≥t_lo.
  - **Veto:** if P(s,a) and P(s,b) are both high but P(a,b) is very low, demote the weaker edge.
  
  Allow at most 2 hops. Never chain S2→S3→S2.
- **Sources:** TransClean https://arxiv.org/html/2506.04006 (+24 F1 from transitive consistency on multi-source data). 7th place Foursquare (UnionFind with betweenness pruning).
- **Cost:** 2h. Prefer turning this into M2 features over hard rules.

### N2. Transductive IDF on test (only if the rules allow it)
- **What:** Fit TF-IDF/IDF on train+test S2/S3 text, so French tokens ("sarl", "rue", "paris") get correct rarity weights.
- **Why:** Otherwise the IDF comes from US/IN statistics, which overweights matches on very common French tokens.
- **Cost:** 30 minutes. **Rules question**, so ask the organisers first.

---

## 4. Tier 3: Day 2 to 3, only if the ledger says they're worth it

| Card | Idea | Cost | Note |
|---|---|---|---|
| L1 | **Qwen3-4B (Apache) QLoRA judge** on the uncertain band only (e.g. 0.2<p<0.8). One forward pass, read the yes/no logit, r=16, 1 to 2 epochs, thinking mode off. Use it as a GBDT feature. | 4 to 8 T4-hours | Fine-tuning gave +17 F1 over zero-shot; structured-explanation targets added about +5. But cross-domain fine-tuning *hurt* Llama-8B by 11 F1, so gate it on LOCO. https://arxiv.org/html/2409.08185v1 |
| A3 | **Rule-based "French-ization"** of train pairs: &→et, Ltd→SARL/SAS, Street→rue with the number moved before it, 5-digit CP, dropped accents. Use it mainly as a **stress-test validation set**; train on it only lightly. | 2h | No paper backing, so it's a guess. It doesn't create country=FR→match shortcuts if the country field isn't a feature. |
| C5 | **UMC / Király / exact clustering**, only if EDA shows one-to-one per source | 1h | Best on the clean-clean benchmark (avg F1 ≈ 0.62). It hurts recall if the data is many-to-one. https://arxiv.org/pdf/2112.14030v3 |
| N3 | **Impute missing PIN/state** from the modal value for the same city within the provided data, plus "imputed" flags | 1 to 2h | hobbitlab team: 0.931→0.940 |
| M5 | **Final blend**: weight GBDT + cross-encoder + reranker scores by CV, then re-calibrate, then D3, then D1 | 2h | Champion weights were about 0.32 CatBoost / 0.29 XLM-R / 0.38 mDeBERTa |

---

## 5. Models: licence-checked shortlist (all ≤8B)

| Model | Params | Licence | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small / base | 0.1B / 0.3B | MIT ✅ | Bi-encoder blocking (needs the `query: ` prefix) |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.1B | Apache ✅ | Fastest bi-encoder |
| BAAI/bge-m3 | ~0.57B | MIT ✅ | Dense + sparse hybrid blocking |
| Alibaba-NLP/gte-multilingual-base / reranker-base | 0.3B | Apache ✅ | Bi-encoder / cross-encoder (trust_remote_code) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | Apache ✅ | Strongest multilingual small embedder |
| BAAI/bge-reranker-v2-m3 | 0.6B | Apache ✅ | Cross-encoder |
| Qwen/Qwen3-Reranker-0.6B | 0.6B | Apache ✅ | Zero-shot reranker feature |
| microsoft/mdeberta-v3-base | 0.28B | MIT ✅ | Ditto-style cross-encoder |
| xlm-roberta-base / large | 0.28B / 0.56B | MIT (re-check the model card) | Cross-encoder backbone |
| Qwen/Qwen3-4B, Qwen2.5-1.5B/7B-Instruct, Mistral-7B-v0.3 | 1.5 to 7B | Apache ✅ | LLM judge (uncertain band only) |
| ❌ Qwen2.5-**3B**-Instruct | 3B | Qwen Research, non-commercial | **Do not use** |
| ❌ jina-embeddings-v3, ❌ Jellyfish-7B/13B | – | CC-BY-NC | **Do not use** |
| ❌ Llama and Gemma family | – | Custom licences | **Do not use** |

**Libraries:** `rapidfuzz` (MIT), `cleanco` (MIT), `anyascii` (ISC), `faiss` (MIT), `lightgbm` (MIT), `catboost` (Apache), `sentence-transformers` (Apache). **Avoid `Unidecode` (GPL-2.0+) and `text-unidecode` (Artistic/GPL)**, and use stdlib NFKD instead. The rule strictly says "final model", but don't risk it in a reviewed package.

---

## 6. Ask the organisers now (Google Form)

1. Are hand-written abbreviation/legal-form dictionaries allowed? What about `cleanco`'s built-in term list, or the GLEIF ELF list (CC0)?
2. Is **libpostal** allowed? Its code is MIT, but its models are trained on OpenStreetMap/OpenAddresses data.
3. May we fit unsupervised statistics (TF-IDF/IDF, embedding normalisation) on the **test** S2/S3 records? Is pseudo-labelling on test allowed?
4. Does the MIT/Apache ≤8B rule apply to **every** model in the pipeline (embedders, rerankers) or only the "final model"? Do pretrained encoders count as "external data"?
5. If train and test records overlap exactly, may train labels be used?

---

## 7. Traps (don't do these)

- Connected components or transitive closure over S2/S3. It chains errors, and the FAMER benchmark shows it is consistently poor.
- Shopee's "always predict ≥1 / min2" trick. Here an empty prediction is worth 1.0 on singletons.
- Country (one-hot) or raw tokens as features, or hard-coded `{US, India}`.
- Stage-2 or relational features computed from in-fold scores.
- Trusting a dense cosine threshold on France: calibration shifts. Use ranks and margins, and let the GBDT decide.
- Picking a model on the public LB when CV or LOCO disagree.

## Coverage gaps

- Kaggle discussion pages rendered client-side, so the Foursquare 1st/2nd/3rd write-ups weren't read directly. Their numbers come from secondary summaries (the champion re:waiwai's Chinese write-up, the 7th-place blog, and Foursquare's own blog). Placements were reshuffled after a leak re-evaluation.
- Lewis '95, Jansche '07 and Waegeman '14 weren't opened. The D1 theory rests on Ye '12 and Dembczyński '11, both of which were read.
- HierGAT, Rotom, JointBERT and Sudowoodo were only skimmed.
