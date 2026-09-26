"""Build the cross-encoder pair files from the pipeline's work dir (run after run.py train + test).

  python scripts/ce_data.py --data <dataset_dir> --work work --out ce_data [--parts train,val,test]

  train.parquet      hard pairs of --n_s1 training S1 outside the validation split: every true match,
                     every candidate within route rank 5 and 35% of those ranked 6-10 (columns s1, m, y, a, b)
  val_band.parquet   validation pairs in the uncertain band (0.02 <= p < 0.998) of work/val_scores.parquet
  test_band.parquet  test pairs in that band of work/test_scores.parquet
  pseudo.parquet     (--parts pseudo) self-training pairs for test countries WITHOUT training labels: confident
                     test predictions (blended with --ce_scores) as labels, --pseudo_n per class
The text of a record is "business_name | business_address".
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl

from ber.io import find_dataset_dir, read_ground_truth, read_split
from ber.pipeline import log

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--parts", default="train,val,test")
ap.add_argument("--n_s1", type=int, default=160_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude_s1", default=None, help="parquet(s) with an s1 column: S1 a previous cross-encoder trained on")
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.998)
ap.add_argument("--skip_scored", default=None, help="test: leave out pairs already in this ce_test.parquet")
ap.add_argument("--test_countries", default=None, help="test: only S1 of these countries (comma-separated, or 'unlabelled')")
ap.add_argument("--ce_scores", default=None, help="pseudo: cross-encoder test scores to blend in (logit, w=0.6)")
ap.add_argument("--pseudo_hi", type=float, default=0.995)
ap.add_argument("--pseudo_lo", type=float, default=0.02)
ap.add_argument("--pseudo_n", type=int, default=150_000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dd = find_dataset_dir(a.data)
parts = a.parts.split(",")


def text(df):
    return df.select("entity_id", (pl.col("business_name") + " | " + pl.col("business_address").fill_null("")).alias("t"))


def with_text(pairs, s1, s23):
    return (pairs.join(text(s1).rename({"entity_id": "s1", "t": "a"}), on="s1")
            .join(text(s23).rename({"entity_id": "m", "t": "b"}), on="m"))


band = (pl.col("p") >= a.lo) & (pl.col("p") < a.hi)
if "train" in parts or "val" in parts:
    s1, s23 = read_split(dd, "train")
    va = pl.read_parquet(f"{a.work}/val_scores.parquet")
    if "val" in parts:
        vb = with_text(va.filter(band), s1, s23).select("s1", "m", "y", "p", "a", "b")
        vb.write_parquet(f"{a.out}/val_band.parquet"); log("val band pairs", vb.height)
    if "train" in parts:
        s1n = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id"])["entity_id"]
        s23n = pl.read_parquet(f"{a.work}/train_s23n.parquet", columns=["entity_id"])["entity_id"]
        taken = va["s1"].unique()                                                  # never a validation S1
        for f in (a.exclude_s1.split(",") if a.exclude_s1 else []):
            taken = pl.concat([taken, pl.read_parquet(f, columns=["s1"])["s1"]])
        free = np.where(~s1n.is_in(taken.implode()).to_numpy())[0]
        pick = pl.Series(np.random.default_rng(a.seed).choice(free, min(a.n_s1, len(free)), replace=False)).cast(pl.Int32)

        def route(f, rk, cap):
            path = f"{a.work}/train_{f}.parquet"
            if not os.path.exists(path):
                return None
            return (pl.scan_parquet(path).filter(pl.col("qi").is_in(pick.implode()) & (pl.col(rk) <= cap))
                    .select(pl.col("qi").cast(pl.Int32), pl.col("ci").cast(pl.Int32), pl.col(rk).cast(pl.Float32).alias("r")).collect())
        pr = pl.concat([r for r in [route("cand", "tok_rank", 10), route("dense", "dense_rank", 10),
                                    route("dense_rev", "rdense_rank", 2)] if r is not None])
        pr = pr.group_by("qi", "ci").agg(pl.col("r").min())
        pr = pr.with_columns(s1n.gather(pr["qi"]).alias("s1"), s23n.gather(pr["ci"]).alias("m"))
        pr = pr.join(read_ground_truth(dd).with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left")
        pr = pr.with_columns(pl.col("y").fill_null(0))
        neg = pr.filter(pl.col("y") == 0)
        tr = pl.concat([pr.filter(pl.col("y") == 1), neg.filter(pl.col("r") <= 5),
                        neg.filter(pl.col("r") > 5).sample(fraction=0.35, seed=0)])
        tr = with_text(tr.sample(fraction=1.0, shuffle=True, seed=1).select("s1", "m", "y"), s1, s23)
        tr.write_parquet(f"{a.out}/train.parquet"); log("train pairs", tr.height, "positive rate", round(tr["y"].mean(), 3))
if "pseudo" in parts:
    s1, s23 = read_split(dd, "test")
    seen = set(read_split(dd, "train")[0]["country"].unique().to_list())
    new_c = sorted(set(s1["country"].unique().to_list()) - seen)
    te = pl.read_parquet(f"{a.work}/test_scores.parquet").join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
    te = te.filter(pl.col("country").is_in(new_c))
    if a.ce_scores:
        lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
        te = te.join(pl.read_parquet(a.ce_scores, columns=["s1", "m", "ce"]), on=["s1", "m"], how="left").with_columns(
            p=pl.when(pl.col("ce").is_null()).then(pl.col("p")).otherwise(1 / (1 + (-(0.6 * lg("p") + 0.4 * lg("ce"))).exp())))
    pos = te.filter(pl.col("p") >= a.pseudo_hi); neg = te.filter(pl.col("p") <= a.pseudo_lo)
    pos = pos.sample(min(a.pseudo_n, pos.height), seed=0).with_columns(pl.lit(1, pl.Int8).alias("y"))
    neg = neg.sample(min(a.pseudo_n, neg.height), seed=0).with_columns(pl.lit(0, pl.Int8).alias("y"))
    ps = with_text(pl.concat([pos, neg]).select("s1", "m", "y"), s1, s23)
    ps.write_parquet(f"{a.out}/pseudo.parquet"); log("pseudo pairs for", new_c, ps.height, "positive rate", round(ps["y"].mean(), 3))
if "test" in parts:
    s1, s23 = read_split(dd, "test")
    tb = pl.read_parquet(f"{a.work}/test_scores.parquet").filter(band)
    if a.test_countries:
        cs = a.test_countries.split(",")
        if cs == ["unlabelled"]:     # every test country without training labels
            cs = sorted(set(s1["country"].unique().to_list()) - set(read_split(dd, "train")[0]["country"].unique().to_list()))
        keep = s1.filter(pl.col("country").is_in(cs))["entity_id"].implode()
        tb = tb.filter(pl.col("s1").is_in(keep))
    if a.skip_scored:
        tb = tb.join(pl.read_parquet(a.skip_scored, columns=["s1", "m"]), on=["s1", "m"], how="anti")
    tb = with_text(tb, s1, s23).select("s1", "m", "p", "a", "b")
    tb.write_parquet(f"{a.out}/test_band.parquet"); log("test band pairs", tb.height)
