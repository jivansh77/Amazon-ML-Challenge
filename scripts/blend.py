"""Final decoding: blend the stage-2 model with the cross-encoder, pick per-country thresholds, write outputs.

  python scripts/blend.py --data <dataset_dir> --work work --ce work_ce --out output [--w 0.6]

Inputs (from run.py / ce_train.py):
  work/val_scores.parquet   validation pairs (s1, m, y, p)       work/test_scores.parquet  test pairs (s1, m, p)
  work_ce/ce_val.parquet    cross-encoder on the val band (s1, m, ce)
  work_ce/ce_test.parquet   cross-encoder on the test band (s1, m, ce)
Pairs without a cross-encoder score (outside the uncertain band 0.02 <= p < 0.998) keep p.
The blend is a logit average  sigmoid(w * logit(p) + (1 - w) * logit(ce)); --w is chosen on validation
unless given. Thresholds per country come from ber.pipeline.prior_thresholds (validation matches per S1
per probability band divided by the test pairs per S1 in that band = the band's precision on test).
Writes <out>/matching_results.tsv and <out>/candidate_pairs.tsv (every pair the stage-2 model scored).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl

from ber.io import find_dataset_dir, read_ground_truth, read_source, write_id_lists
from ber.metric import macro_f05
from ber.pipeline import decode, decode_by_country, log, prior_thresholds

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--ce", required=True, help="dir(s) with ce_val.parquet / ce_test.parquet, comma-separated "
                                            "(later dirs add the pairs a re-scored band gained)")
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=float, default=None, help="weight of the stage-2 model (default: best on validation)")
ap.add_argument("--min_prec", type=float, default=0.78)
ap.add_argument("--thr", default=None, help="override thresholds, e.g. US=0.8,France=0.95 (others: prior estimate)")
ap.add_argument("--override_scores", default=None,
                help="COUNTRY:path[,COUNTRY:path]: test scores for that country's S1 come from another run "
                     "(COUNTRY 'unlabelled' = every test country without training labels, i.e. France)")
ap.add_argument("--ce_test_override", default=None,
                help="cross-encoder test scores that replace the --ce ones for their pairs (e.g. a country-adapted model)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dd = find_dataset_dir(a.data)


def blend(df, ce, w):
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    d = df.join(ce.select("s1", "m", "ce"), on=["s1", "m"], how="left")
    b = 1 / (1 + (-(w * lg("p") + (1 - w) * lg("ce"))).exp())
    return d.with_columns(pl.when(pl.col("ce").is_null()).then(pl.col("p")).otherwise(b).alias("p")).drop("ce")


def read_ce(name):
    paths = [f"{d}/{name}" for d in a.ce.split(",") if os.path.exists(f"{d}/{name}")]
    return pl.concat([pl.read_parquet(p, columns=["s1", "m", "ce"]) for p in paths]).unique(["s1", "m"], keep="last")


va = pl.read_parquet(f"{a.work}/val_scores.parquet")
cv = read_ce("ce_val.parquet")
ids = va.select("s1").unique()
truth = read_ground_truth(dd).filter(pl.col("s1").is_in(ids["s1"].implode()))
grid = [a.w] if a.w is not None else [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]
res = []
for w in grid:
    vb = blend(va, cv, w)
    f, t = max((macro_f05(decode(vb, t, True), truth, ids)["f05"], t) for t in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    res.append((f, w)); log(f"val F0.5 w={w}: {f:.5f} (thr {t})")
w = max(res)[1]
log("blend weight of the stage-2 model:", w)

ct = read_ce("ce_test.parquet")
if a.ce_test_override:
    ov = pl.read_parquet(a.ce_test_override, columns=["s1", "m", "ce"])
    ct = pl.concat([ct.join(ov, on=["s1", "m"], how="anti"), ov])
    log("cross-encoder scores replaced for", ov.height, "test pairs")
trc = read_source(os.path.join(dd, "train", "train_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
tec = read_source(os.path.join(dd, "test", "test_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
ts = pl.read_parquet(f"{a.work}/test_scores.parquet", columns=["s1", "m", "p"])
unlabelled = sorted(set(tec["country"].unique().to_list()) - set(trc["country"].unique().to_list()))
for item in (a.override_scores.split(",") if a.override_scores else []):
    c, path = item.split(":", 1)
    cs = unlabelled if c == "unlabelled" else [c]      # "unlabelled" = test countries without training labels
    ids = tec.filter(pl.col("country").is_in(cs))["s1"].implode()
    ts = pl.concat([ts.filter(~pl.col("s1").is_in(ids)), pl.read_parquet(path, columns=["s1", "m", "p"]).filter(pl.col("s1").is_in(ids))])
    log(f"{cs}: test scores taken from {path}")
te = blend(ts, ct, w)
thr, est = prior_thresholds(blend(va, cv, w), te, trc, tec, min_prec=a.min_prec)
log("per-country thresholds (prior estimate):", thr)
if a.thr:
    thr.update({k: float(v) for k, v in (x.split("=") for x in a.thr.split(","))})
    log("thresholds used:", thr)
log(est.select("country", "band", pl.col("prec").round(3)).pivot(on="country", index="band", values="prec").sort("band"))
pred = decode_by_country(te, thr, tec)
s1_ids = tec["s1"].to_list()
write_id_lists(f"{a.out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
write_id_lists(f"{a.out}/candidate_pairs.tsv", s1_ids, te.select("s1", "m"), "candidate_entity_ids")
log("written:", pred.height, "matches for", pred["s1"].n_unique(), "of", len(s1_ids), "S1;",
    te.height, "candidate pairs (", round(te.height / len(s1_ids), 2), "per S1 )")
log(pred.join(tec, on="s1").group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("S1")).sort("country"))
