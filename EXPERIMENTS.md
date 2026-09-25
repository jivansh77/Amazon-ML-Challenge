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

Pending: results of `ber-exp-tok`, `ber-exp-union` and `ber-exp-union-s2`.
