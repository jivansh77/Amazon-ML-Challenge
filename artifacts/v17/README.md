`avg_ce4_v17` = `avg_ce4_v16` + `legal_tie_adds.parquet` appended to `--extra_pairs` (823 pairs: France 682, US 89, India 52).

    python scripts/legal_tie.py --data DS --claimed <v16 pred.parquet> --out legal_tie_adds.parquet

Address-less records whose core name is shared by 2+ S1s, placed by the legal form (see the script docstring and EXPERIMENTS.md
"Legal-form tie-break"). The uploaded v17 files append the unclaimed pairs to v16 (same `--extra_pairs` semantics).
Output: 5,840,027 matches, validator PASS, no record on two S1s.
