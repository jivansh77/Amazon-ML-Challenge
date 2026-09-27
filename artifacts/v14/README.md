`avg_ce4_v14` (the recommended final) = `avg_ce4_v13` + first match for empty US/India S1.

A US/India S1 that still has no match after decoding, the France rules and the route files gets its best unclaimed
candidate whose best-scoring S1 it is, when the blend is >= 0.5. This is `scripts/final_build.py --first_min`, which now
runs after the route files. The main thresholds are unchanged (US/India 0.8, France 0.87). France is unchanged.

Build: the `artifacts/v13` command plus `--first_min US=0.5,India=0.5`:

    A=artifacts/v13
    python scripts/final_build.py --data DS --runs $K/nr,$K/nr10 \
      --ce $K/nr/ce_test.parquet:1,$K/nr10/cebase_ce_test.parquet:1,$K/ber-ce-base4/work/ce_test.parquet:1,$K/ber-ce-base4b/work/ce_test.parquet:1,$K/qwen_ce_test.parquet:1 \
      --ce_fr $K/nr/ce_test_france.parquet,$K/nr10/ce_test_france_new.parquet \
      --rules A2P,A2F,A2N,HN_A,DP,E2,OOC,HNK --collapse_legal --first_min US=0.5,India=0.5 \
      --extra_pairs $A/native_adds_q90.parquet,$A/kv_r3r6_extras.parquet,$A/fr_inv_adds.parquet,$A/fr_acr_adds.parquet,$A/fr_fuzzy_street_adds.parquet,$A/legal_tiebreak_adds.parquet,$A/fr_inv_city_adds_f.parquet,$A/fr_acr_city_adds.parquet,$A/fr_inv_legal_adds.parquet \
      --out OUT

Output: 5,837,764 matches (v13 + 904: US 456, India 448), validator PASS, no record on more than one S1, same candidate
pairs as v13. `first_match_adds.parquet` holds the 904 pairs.
