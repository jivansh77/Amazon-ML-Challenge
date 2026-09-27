`avg_ce4_v15` (recommended final) = `avg_ce4_v14b` + `fr_acr_shared_adds.parquet` as the last `--extra_pairs` file.

The file holds 371 France acronym records at an address (house number + street key + all non-numeric address parts) shared
by 2+ France S1s, where exactly one of those S1s has the record's letters in its initials. It is made by:

    python scripts/france_name_replaced.py --data DS --claimed <v14b pred.parquet> --mode acr_shared --out fr_acr_shared_adds.parquet

Build: the `artifacts/v14` v14b command with `artifacts/v15/fr_acr_shared_adds.parquet` appended to `--extra_pairs`.
Output: 5,837,948 matches (v14b + 371 France), validator PASS, no record on more than one S1.
