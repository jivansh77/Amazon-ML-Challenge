# Handoff: final_merge_v1 and final_merge_v2 (27 Sep 2026, Kavya's account)

## 1. What the file is
- **Base:** `matching_results_avg_ce4_v2h.tsv` from Kaggle dataset `jvmusic/ber-france-v2` (LB 0.98809).
- **final_merge_v1** is that base plus **US/India additions only**:
  - **0 removals.** The base's 5,807,199 pairs are all still there, so the file has 5,818,776 pairs.
  - **France rows are byte-identical to the base** in both the matching and the candidate file. A line-by-line comparison checked this.
  - **Official validator PASS with `--check-ids`:** 1,732,544 rows, 101,409 empty.
- **Files:**
  - Local: `~/Desktop/mlchallenge/runs/final-merge/final_merge_v1/matching_results.tsv` and `.../candidate_pairs.tsv`.
  - Kaggle dataset `kavyachetwani/ber-final-merge-v1` (private): `matching_results_final_merge_v1.tsv` and `candidate_pairs_final_merge_v1.tsv`.
  - Also the output of kernel `kavyachetwani/final-merge`, folder `final_merge_v1/`.
- **Candidate file** = `candidate_pairs_avg_ce4_v2.tsv` plus 8,016 ids:
  - the R3 route records, which were not in the base candidate lists;
  - 1 family pair.
  - Base order is kept and the new ids are appended.
- **final_merge_v2** = the same base + family + R3 + R6 additions (US/India only):
  - 0 removals vs the base; France byte-identical (matching and candidates); validator PASS `--check-ids`.
  - 5,825,585 pairs (+18,386), 16,155 S1 changed, candidate file +14,834 ids.
  - Files in the same dataset: `matching_results_final_merge_v2.tsv`, `candidate_pairs_final_merge_v2.tsv`. Local: `~/Desktop/mlchallenge/runs/final-merge-v2/final_merge_v2/`.
  - v2 is not a strict superset of v1: 9 v1 pairs (7 R3, 2 family) moved to another S1, because R6 proposed the same record for that S1 at a higher probability (one S1 per record).

## 2. Why this was done in PROXY mode
- The submitted blend can't be rebuilt from `kavyachetwani`:
  - the top-5 and top-10 reverse-name route runs are on AWS;
  - `ber-ce-base4` and `ber-ce-base4b` are 403 (so are `ber-ce-base` and `ber-ce-data3`).
- Every idea was therefore measured as a **paired delta on clean val** against a proxy blend:
  - proxy = `ber-dec-train` p + `ber-ce-base2` CE, logit w 0.6, exclusivity, thr 0.75;
  - clean val = dec val S1 minus every S1 in `ber-ce-data`/`ber-ce-data2`, i.e. 94,379 S1 (US 56,904, India 37,475);
  - the proxy reproduces exactly at 0.989421.
- On test, each idea was applied as a **delta edit of the submitted file**:
  - only records that the submitted file leaves unclaimed were added;
  - each record goes to one S1 (exclusivity, also across sources);
  - nothing the submitted file accepts was removed.

## 3. Components and their numbers (paired deltas vs the proxy, clean val)

**a) Family completion** (chat 2; kernels `fam-pre`, `fam-ce`, `fam-gate`)
- **Method:** band candidates are rescored by a depth-4 XGBoost. Its inputs:
  - the proxy score q;
  - ber-ce-base2 CE of (candidate, each already-accepted family member): max, mean, min;
  - house-number match with the members;
  - source coverage;
  - competing-S1 evidence;
  - string similarities.
- **Accuracy:** band AUC 0.824 (q) → 0.837. A q+CE-only XGB control gives ~0, so the gain comes from the family evidence.
- **Val at prob ≥ 0.75:** +0.000137.
  - fold0 +0.000107, fold1 +0.000167.
  - US +0.000166, India +0.000092.
  - Singletons 0.
  - 427 adds, precision 0.838.
- **Test:** 3,564 adds (US 2,127 / India 1,437) and 0 removals. 3,562 are in the merge; 2 lost their record to a higher-probability R3 pair.
- **Its own gate (+0.0003 on both folds) failed.** The test file was built on explicit request.

**b) R3 phonetic route** (chat 3; kernels `rt-miss`, `rt-score`, `rt-build`)
- **Method:**
  - New candidates = Double Metaphone (BSD) of 2 name tokens + house number, top-2 per S1.
  - Scored by the base2 CE + an XGBoost (OOF AUC 0.9997).
  - It recovers 16% of India's never-candidate true pairs (US ~0%).
- **Val at 0.70:** +0.000312 (fold A) / +0.000232 (fold B).
- **Singletons:** one singleton S1 gets a false add at every threshold 0.30–0.99 (fold B singleton accuracy −0.00085). Its gate failed on that clause only.
- **Test:** 8,015 adds at prob ≥ 0.70 (India 8,013, US 2) on 6,835 S1.
  - A spot check shows Indic-script copies at the same house number (Hindi/Kannada/Telugu/Tamil) and OCR typos.
  - 6,445 of the 8,015 have prob ≥ 0.99.

**c) R6** (chat 3; kernels `rt-miss2-tr`, `rt-miss2-te`, `rt-score2`)
- **Method:** same-name compact key, restricted to a shared house number or city word; R3 candidates excluded. Scored by the base2 CE + an XGBoost (OOF AUC 0.9945).
- **Its own gate** (both folds ≥ +0.0001, ≤ 2 singleton S1 lost; baseline already includes R3): PASS. Variant top2, t = 0.55 (tuned on fold 0).
  - Fold 0 +0.000214, fold 1 +0.000122 (1 singleton S1 lost). The top1 variant also passed (+0.000195 / +0.000134 at 0.85).
- **In the merge check:** +0.000170 alone (US +0.000011, India +0.000412); 284 val adds, precision 0.887.
- **Test:** 6,871 adds at prob ≥ 0.55 on records unclaimed by the base; 6,818 in v2 (India 6,785, US 33), 53 lost to higher-probability R3/family pairs.

**d) The combinations**

final_merge_v1 (family + R3):
- **Paired clean-val delta:** **+0.000408** (proxy 0.989421 → 0.989829). It equals the sum of the parts (0.000137 + 0.000272), and no val record is proposed by both sources.
  - US +0.000167, India +0.000775.
  - Fam folds +0.000403 / +0.000414; md5 folds +0.000515 / +0.000303.
  - Singletons −0.000429 (the one R3 singleton).
  - 746 val adds, precision 0.895.
- **Test file:** +11,577 pairs (R3 8,015; family 3,562), 10,327 S1 changed (India 8,240, US 2,087).
  - Matches per S1: US 3.3827 → 3.3859, India 3.3542 → 3.3658.

final_merge_v2 (family + R3 + R6):
- **Paired clean-val delta:** **+0.000576** (0.989421 → 0.989997), about the sum of the parts (0.000137 + 0.000272 + 0.000170).
  - US +0.000178, India +0.001181.
  - Fam folds +0.000571 / +0.000582; md5 folds +0.000729 / +0.000424.
  - Singletons −0.000858: 2 singleton S1 (one from R3, one from R6), both in the same fold.
  - 1,030 val adds, precision 0.893.
- **Test file:** +18,386 pairs (family 3,560, R3 8,008, R6 6,818), 16,155 S1 changed (India 14,035, US 2,120).
  - Matches per S1: US 3.3827 → 3.3860, India 3.3542 → 3.3742.

## 4. Tried and not included
- **State-dependent thresholds t(k, r)** (kernel `st-thr`): best variant +0.000144 held-out; singleton accuracy −0.0026 / −0.0050; gate failed.
- **Routes R1, R1b, R2 and R4:** recall of missed pairs 0.5–7.5%, but precision of the added candidates is only 0.02–0.6%, at 0.09–1.96 added candidates per S1. Too many adds to score and decode.
- **R5:** also rejected in chat 3 for low precision.
- **Synthetic French CE** (`syn-cont-fr`): US/India clean val 0.98862 vs 0.98942; as the France CE it removes clear matches. Rejected, as in his EXPERIMENTS.md.

## 5. Caveats
- The gains were measured on the proxy blend, not on the submitted blend (route average + base3/base4/base4b). The realised LB gain may be smaller; our estimate is roughly +0.0002 to +0.0004 LB for v1 and +0.0003 to +0.0005 for v2.
- One sign of this: family adds are 0.24% per US/India S1 on test vs 0.45% on val, because the submitted blend already claims more.
- R3's val adds are 97% precise, but R3 was trained on 330 positives.
- All changes are additions. If a component doesn't transfer, the downside is limited to those added pairs: v1 11,577 pairs on 10,327 S1 (0.6% of S1), v2 18,386 pairs on 16,155 S1 (0.9%). R6 adds are 99% India and were trained on 300 positives.

## 6. Where everything is
- **Kaggle kernels** (all `kavyachetwani/`):
  - `st-thr`, `fam-pre`, `fam-ce`, `fam-gate`, `rt-miss`, `rt-score`, `rt-build`, `rt-miss2-tr`, `rt-miss2-te`, `rt-score2`, `final-merge` (version 1 = v1, version 2 = v2).
  - Inputs from `jvmusic/`: `ber-dec-train`, `ber-dec-test3`, `ber-ce-base2`, `ber-ce-data`, `ber-ce-data2`, `ber-france-v2`, `ml-challenge`.
- **Local folders:** `~/Desktop/mlchallenge/runs/<kernel>/` (progress.log, metrics.json, reports) and `kernels/<kernel>/script.py`.
- **Merge reports:** v1 `runs/final-merge/{val_combo,diff_report}.json`; v2 `runs/final-merge-v2/{val_combo,diff_report}.json` (also in the dataset as `*_v2.json`).
- **Ledger:** `~/Desktop/mlchallenge/ledger.csv`, rows `st-thr`, `fam-pre`, `fam-ce`, `fam-gate`, `fam_v1`, `rt-miss`, `rt-score`, `rt-build`, `rt-miss2-tr`, `rt-miss2-te`, `rt-score2`, `final-merge` (version 1 = v1, version 2 = v2).
