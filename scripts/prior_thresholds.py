"""Per-country decision thresholds corrected for the test set's higher decoy density.

Idea: the number of TRUE matches per S1 in each probability band is taken from validation
(France, which has no labels, uses the US/India average); the number of test pairs per S1 in
the same band is observed. Their ratio estimates the band's precision ON TEST. A pair only helps
macro F0.5 if its precision exceeds ~F*/(1+beta^2) ~ 0.78, so each country's threshold is the lowest
band edge above which every band clears that bar.

  python scripts/prior_thresholds.py VAL_SCORES TEST_SCORES TRAIN_S1 TEST_S1 OUT_TSV [min_prec]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import write_id_lists

from ber.pipeline import prior_thresholds, decode_by_country

val_p, test_p, tr_s1, te_s1, out = sys.argv[1:6]
MIN_PREC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.78
va, te = pl.read_parquet(val_p), pl.read_parquet(test_p)
tr1 = pl.read_parquet(tr_s1, columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t1 = pl.read_parquet(te_s1, columns=["entity_id", "country"]).rename({"entity_id": "s1"})
thr, est = prior_thresholds(va, te, tr1, t1, MIN_PREC)
print(est.select("country", "band", pl.col("prec").round(3)).pivot(on="country", index="band", values="prec").sort("band"))
print("per-country thresholds:", thr)
d = decode_by_country(te, thr, t1)
write_id_lists(out, t1["s1"].to_list(), d, "matched_entity_ids")
print("pairs", d.height)
