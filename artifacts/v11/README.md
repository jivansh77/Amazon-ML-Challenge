Route/rule pairs added to the decoded matches for `avg_ce4_v11` (passed to `scripts/final_build.py --extra_pairs` in this order;
the earlier file wins when two propose the same record; a pair is added only when its record is unclaimed):

1. `native_adds_q90.parquet`: India native-script route (`scripts/native_route.py`, q >= 0.9, outside the candidate lists)
2. `kv_r3r6_extras.parquet`: Kavya's R3/R6 routes (final_merge_v2), records v4 left unclaimed
3. `fr_inv_adds.parquet`: France invented-name records at a unique S1 address (`scripts/france_name_replaced.py`)
4. `fr_acr_adds.parquet`: France acronym records at a unique S1 address (`scripts/france_name_replaced.py`)
5. `legal_tiebreak_adds.parquet`: empty-address same-name ties broken by the legal form (nsn 2-3, record has a legal form)

Build (K = run outputs, see EXPERIMENTS.md "Final build"; qwen_ce_test.parquet is the Qwen2.5-7B LoRA test cut from the
yoddhas-llm-qwen7b-g5b job):

    python scripts/final_build.py --data DS --runs $K/nr,$K/nr10 \
      --ce $K/nr/ce_test.parquet:1,$K/nr10/cebase_ce_test.parquet:1,$K/ber-ce-base4/work/ce_test.parquet:1,$K/ber-ce-base4b/work/ce_test.parquet:1,$K/qwen_ce_test.parquet:1 \
      --ce_fr $K/nr/ce_test_france.parquet,$K/nr10/ce_test_france_new.parquet \
      --rules A2P,A2F,A2N,HN_A,DP,E2,OOC,HNK --collapse_legal \
      --extra_pairs artifacts/v11/native_adds_q90.parquet,artifacts/v11/kv_r3r6_extras.parquet,artifacts/v11/fr_inv_adds.parquet,artifacts/v11/fr_acr_adds.parquet,artifacts/v11/legal_tiebreak_adds.parquet \
      --out OUT
