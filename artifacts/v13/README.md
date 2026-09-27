Route/rule pairs added to the decoded matches for `avg_ce4_v13` (the recommended final; `avg_ce4_v12` = the first six files).
They are passed to `scripts/final_build.py --extra_pairs` in this order. When two files propose the same record, the earlier
file wins. A pair is added only when its record is unclaimed.

1. `native_adds_q90.parquet`: India native-script route (`scripts/native_route.py`, q >= 0.9, outside the candidate lists)
2. `kv_r3r6_extras.parquet`: Kavya's R3/R6 routes (final_merge_v2), records v4 left unclaimed
3. `fr_inv_adds.parquet`: France invented-name records at a unique S1 address (`scripts/france_name_replaced.py`)
4. `fr_acr_adds.parquet`: France acronym records at a unique S1 address (`scripts/france_name_replaced.py`)
5. `fr_fuzzy_street_adds.parquet` (v12): France acronym / invented-name records unclaimed in v11 on the same house number +
   city as exactly one France S1, where the street key differs only by a typo (rapidfuzz ratio 85-99, no other S1 on that
   number + city at >= 60). `scripts/france_name_replaced.py --fuzzy_street`.
6. `legal_tiebreak_adds.parquet`: empty-address same-name ties broken by the legal form (nsn 2-3, record has a legal form)
7. `fr_inv_city_adds_f.parquet` (v13): invented-name records at the house number + street key of exactly one France S1 *in
   their city* (`--unique_by city`; v10s/v11 needed it to be unique in all of France), unclaimed in v12. Web handles
   (@x, #x, ...com) that contain no S1 word and are not the S1's acronym are dropped (`--drop_foreign_handles`).
8. `fr_acr_city_adds.parquet` (v13): the same for acronym records.
9. `fr_inv_legal_adds.parquet` (v13): an invented word plus a legal form ("Nexaria Co"). The core is one word that is in no
   France S1 name, with no word shared with the S1. The record is at the house number + street + city (all non-numeric
   address parts) of exactly one France S1 and unclaimed in v13 before this file (`--inv_legal`).

Build (K = run outputs, see EXPERIMENTS.md "Final build"; qwen_ce_test.parquet is the Qwen2.5-7B LoRA test cut from the
yoddhas-llm-qwen7b-g5b job):

    A=artifacts/v13
    python scripts/final_build.py --data DS --runs $K/nr,$K/nr10 \
      --ce $K/nr/ce_test.parquet:1,$K/nr10/cebase_ce_test.parquet:1,$K/ber-ce-base4/work/ce_test.parquet:1,$K/ber-ce-base4b/work/ce_test.parquet:1,$K/qwen_ce_test.parquet:1 \
      --ce_fr $K/nr/ce_test_france.parquet,$K/nr10/ce_test_france_new.parquet \
      --rules A2P,A2F,A2N,HN_A,DP,E2,OOC,HNK --collapse_legal \
      --extra_pairs $A/native_adds_q90.parquet,$A/kv_r3r6_extras.parquet,$A/fr_inv_adds.parquet,$A/fr_acr_adds.parquet,$A/fr_fuzzy_street_adds.parquet,$A/legal_tiebreak_adds.parquet,$A/fr_inv_city_adds_f.parquet,$A/fr_acr_city_adds.parquet,$A/fr_inv_legal_adds.parquet \
      --out OUT

Output: 5,836,860 matches (v12: 5,834,113; v11: 5,832,932), validator PASS, no record on more than one S1.
