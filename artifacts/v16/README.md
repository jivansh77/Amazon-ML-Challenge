`avg_ce4_v16` = `avg_ce4_v15b` + two route files appended to `--extra_pairs` (in this order):

- `fr_unit_adds.parquet` (297 pairs: France 287, India 10): single-token records at an address shared by 2+ S1s whose unit /
  sub-number (bis, ter, A, B, Unit 11, Apt 3...) exactly one tenant has.
- `fr_acr_city_shared_adds.parquet` (709 pairs, 671 new after the unit file): the v15 acronym-at-shared-address rule keyed on
  the recognised city.

Both are made from the v15b predictions:

    python scripts/france_shared_addr.py --data DS --claimed <v15b pred.parquet> --mode unit --out fr_unit_adds.parquet
    python scripts/france_shared_addr.py --data DS --claimed <v15b pred.parquet> --mode acr_city --out fr_acr_city_shared_adds.parquet

Build: the `artifacts/v15` v15b command with both files appended to `--extra_pairs`. The v16 files uploaded to Kaggle were made
by appending the unclaimed pairs to the v15b output directly (the same semantics: `--first_min` only fills US/India S1s that
are still empty, and none of these records were claimed in v15b). Output: 5,839,204 matches, validator PASS, no record on two S1s.
