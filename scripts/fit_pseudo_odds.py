"""Learn decoy-word scores for a country without labels (France: no training data) from confident test predictions.

The decoy features score words that a candidate name ADDS to the S1 name ("Acme" -> "Acme Holdings") with
log-odds learned on labelled training pairs; words never seen in training (French qualifiers such as
"Ateliers", "Sainte", a city name, "France") score 0, so French look-alike branches stay uncertain.
This script treats confident test predictions (blended p >= --hi as match, <= --lo as non-match) as labels,
fits the same word log-odds on near-duplicate names, keeps the words the training odds do not know,
and shrinks them (--shrink; on US test the same procedure over-states the training scores ~2x).
The output is passed to `run.py --stages test --odds_extra`, which re-scores test with the new words.

    python scripts/fit_pseudo_odds.py --scores work/test_scores.parquet --ce work/ce_test.parquet \
        --s1n work/test_s1n.parquet --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet \
        --country France --out artifacts/odds_extra_france.parquet
"""
import argparse, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.features import fit_token_odds
from ber.normalize import normalize_frame

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True, help="test scores parquet (s1, m, p)")
ap.add_argument("--ce", default=None, help="cross-encoder scores parquet (s1, m, ce), blended in on logit scale")
ap.add_argument("--w", type=float, default=0.6, help="weight of the model in the logit blend with the cross-encoder")
ap.add_argument("--s1n", required=True, help="test Source 1 (normalised parquet with name_core, or raw)")
ap.add_argument("--s23n", required=True, help="test Source 2+3 (normalised parquet with name_core, or raw); comma-separated ok")
ap.add_argument("--train_odds", required=True, help="tokodds_extra.parquet learned on training labels")
ap.add_argument("--country", default="France")
ap.add_argument("--hi", type=float, default=0.99)
ap.add_argument("--lo", type=float, default=0.05)
ap.add_argument("--min_count", type=int, default=20)
ap.add_argument("--shrink", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def names(paths, ids):
    d = pl.concat([pl.scan_parquet(p).filter(pl.col("entity_id").is_in(ids)).collect() for p in paths.split(",")],
                  how="diagonal_relaxed")
    if "name_core" not in d.columns:
        d = normalize_frame(d, 4)
    return d.select("entity_id", "name_core")


te = pl.read_parquet(a.scores, columns=["s1", "m", "p"])
s1 = pl.scan_parquet(a.s1n).filter(pl.col("country") == a.country).select("entity_id").collect()
te = te.filter(pl.col("s1").is_in(s1["entity_id"].implode()))
if a.ce:
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    te = te.join(pl.read_parquet(a.ce, columns=["s1", "m", "ce"]), on=["s1", "m"], how="left").with_columns(
        p=pl.when(pl.col("ce").is_null()).then(pl.col("p"))
        .otherwise(1 / (1 + (-(a.w * lg("p") + (1 - a.w) * lg("ce"))).exp())))
na = names(a.s1n, te["s1"].unique().implode()).rename({"entity_id": "s1", "name_core": "na"})
nb = names(a.s23n, te["m"].unique().implode()).rename({"entity_id": "m", "name_core": "nb"})
d = te.join(na, on="s1").join(nb, on="m").filter((pl.col("p") >= a.hi) | (pl.col("p") <= a.lo))
print(a.country, "confident pairs:", d.height, "matches:", int((d["p"] >= a.hi).sum()), flush=True)
odds = fit_token_odds(d["na"].to_list(), d["nb"].to_list(), (d["p"] >= a.hi).to_numpy().astype(np.int8),
                      min_count=a.min_count)["extra"]
known = pl.read_parquet(a.train_odds)["token"].implode()
new = odds.filter(~pl.col("token").is_in(known)).with_columns((pl.col("score") * a.shrink).cast(pl.Float32))
new.sort("score").write_parquet(a.out)
print("words unknown to the training odds:", new.height, "| most decoy-like:", new.sort("score").head(20)["token"].to_list())
