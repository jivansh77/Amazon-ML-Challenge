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

val_p, test_p, tr_s1, te_s1, out = sys.argv[1:6]
MIN_PREC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.78
EDGES = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.01]

def band(c):
    e = pl.lit(None, pl.Float64)
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        e = pl.when((pl.col(c) >= a) & (pl.col(c) < b)).then(pl.lit(a)).otherwise(e)
    return e.alias("band")

va = pl.read_parquet(val_p)
te = pl.read_parquet(test_p)
tr1 = pl.read_parquet(tr_s1, columns=["entity_id", "country"]).rename({"entity_id": "s1"})
t1 = pl.read_parquet(te_s1, columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nv = va.select("s1").unique().join(tr1, on="s1").group_by("country").len().rename({"len": "n"})
vt = (va.filter(pl.col("p") >= EDGES[0]).join(tr1, on="s1").with_columns(band("p"))
      .group_by("country", "band").agg(pl.col("y").sum().alias("tp")).join(nv, on="country")
      .with_columns((pl.col("tp") / pl.col("n")).alias("tp_per_s1")))
avg = vt.group_by("band").agg(pl.col("tp_per_s1").mean().alias("tp_avg"))
nt = t1.group_by("country").len().rename({"len": "n"})
tt = (te.filter(pl.col("p") >= EDGES[0]).join(t1, on="s1").with_columns(band("p"))
      .group_by("country", "band").agg(pl.len().alias("pairs")).join(nt, on="country")
      .with_columns((pl.col("pairs") / pl.col("n")).alias("test_per_s1")))
est = (tt.join(vt.select("country", "band", "tp_per_s1"), on=["country", "band"], how="left").join(avg, on="band")
       .with_columns((pl.coalesce("tp_per_s1", "tp_avg") / pl.col("test_per_s1")).clip(0, 1).alias("prec"))
       .sort("country", "band", descending=[False, True]))
thr = {}
for c in est["country"].unique().to_list():
    t = 0.99
    for b, p in est.filter(pl.col("country") == c).select("band", "prec").iter_rows():   # from the top band down
        if p < MIN_PREC:
            break
        t = b
    thr[c] = t
print("estimated test precision per band:")
print(est.select("country", "band", pl.col("prec").round(3)).pivot(on="country", index="band", values="prec").sort("band"))
print("per-country thresholds:", thr)
# decode: exclusivity, then the country's threshold
d = te.filter(pl.col("p") == pl.col("p").max().over("m")).join(t1, on="s1")
d = d.filter(pl.col("p") >= pl.col("country").replace_strict(thr, default=0.9, return_dtype=pl.Float64))
write_id_lists(out, t1["s1"].to_list(), d.select("s1", "m"), "matched_entity_ids")
print("pairs", d.height, "by country", d.group_by("country").len().sort("country").rows())
