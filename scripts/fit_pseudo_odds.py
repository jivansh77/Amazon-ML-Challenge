"""Learn decoy-word scores for a country without labels (France: no training data) from confident test predictions.

The decoy features score words that a candidate name ADDS to the S1 name ("Acme" -> "Acme Holdings") with
log-odds learned on labelled training pairs; words never seen in training (French qualifiers such as
"Ateliers", "Sainte", a city name, "France") score 0, so French look-alike branches stay uncertain.
This script treats confident test predictions (blended p >= --hi as match, <= --lo as non-match) as labels,
fits the same word log-odds on near-duplicate names and keeps the words the training odds do not know.
Their scores are put on the training scale with a monotone (isotonic) map fitted on the labelled countries,
where the same procedure can be compared with the training odds word by word (--calib shrink: halve them).
The output is passed to `run.py --stages test --odds_extra`, which re-scores test with the new words.

    python scripts/fit_pseudo_odds.py --scores work/test_scores.parquet --ce work/ce_test.parquet \
        --s1n work/test_s1n.parquet --s23n work/test_s23n.parquet --train_odds work/tokodds_extra.parquet \
        --data <dataset_dir> --out work/odds_extra.parquet
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
ap.add_argument("--data", default=None, help="dataset dir: fit every test country absent from the training data")
ap.add_argument("--country", default=None, help="country (or comma-separated list) to fit; default: see --data")
ap.add_argument("--hi", type=float, default=0.99)
ap.add_argument("--lo", type=float, default=0.05)
ap.add_argument("--min_count", type=int, default=20)
ap.add_argument("--calib", default="isotonic", choices=["isotonic", "shrink"],
                help="isotonic: map pseudo scores onto the training scale with a monotone fit learned on the LABELLED "
                     "countries' test pairs (same procedure, compared with the training odds); shrink: multiply by --shrink")
ap.add_argument("--shrink", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def names(paths, ids):
    d = pl.concat([pl.scan_parquet(p).filter(pl.col("entity_id").is_in(ids)).collect() for p in paths.split(",")],
                  how="diagonal_relaxed")
    if "name_core" not in d.columns:
        d = normalize_frame(d, 4)
    return d.select("entity_id", "name_core")


if a.country:
    countries = a.country.split(",")
else:     # the open set of countries: every test country that has no training labels
    from ber.io import find_dataset_dir, read_source
    dd = find_dataset_dir(a.data)
    seen = set(read_source(os.path.join(dd, "train", "train_source1.tsv"))["country"].unique().to_list())
    countries = sorted(set(pl.scan_parquet(a.s1n).select("country").unique().collect()["country"].to_list()) - seen)
print("countries without training labels:", countries, flush=True)
TE = pl.read_parquet(a.scores, columns=["s1", "m", "p"])
if a.ce:
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    TE = TE.join(pl.read_parquet(a.ce, columns=["s1", "m", "ce"]), on=["s1", "m"], how="left").with_columns(
        p=pl.when(pl.col("ce").is_null()).then(pl.col("p"))
        .otherwise(1 / (1 + (-(a.w * lg("p") + (1 - a.w) * lg("ce"))).exp())))
ALL_C = pl.scan_parquet(a.s1n).select("entity_id", "country").collect()


def pseudo_odds(cs):
    """Extra-word log-odds from the confident test pairs of countries cs (same recipe as the training odds)."""
    ids = ALL_C.filter(pl.col("country").is_in(cs))["entity_id"].implode()
    te = TE.filter(pl.col("s1").is_in(ids))
    na = names(a.s1n, te["s1"].unique().implode()).rename({"entity_id": "s1", "name_core": "na"})
    nb = names(a.s23n, te["m"].unique().implode()).rename({"entity_id": "m", "name_core": "nb"})
    d = te.join(na, on="s1").join(nb, on="m").filter((pl.col("p") >= a.hi) | (pl.col("p") <= a.lo))
    print(cs, "confident pairs:", d.height, "matches:", int((d["p"] >= a.hi).sum()), flush=True)
    return fit_token_odds(d["na"].to_list(), d["nb"].to_list(), (d["p"] >= a.hi).to_numpy().astype(np.int8),
                          min_count=a.min_count)["extra"]


odds = pseudo_odds(countries)
train = pl.read_parquet(a.train_odds)
new = odds.filter(~pl.col("token").is_in(train["token"].implode()))
if a.calib == "shrink":
    new = new.with_columns((pl.col("score") * a.shrink).cast(pl.Float32))
else:
    # the same recipe on the labelled countries, compared word by word with their training odds, gives a
    # monotone map from pseudo scores to the training scale the model was fitted on
    from sklearn.isotonic import IsotonicRegression
    labelled = sorted(set(ALL_C["country"].unique().to_list()) - set(countries))
    ref = pseudo_odds(labelled).join(train, on="token", suffix="_train")
    iso = IsotonicRegression(out_of_bounds="clip").fit(ref["score"].to_numpy(), ref["score_train"].to_numpy())
    print("calibration on", ref.height, "shared words of", labelled, "| pseudo -> train:",
          [(x, round(float(iso.predict([x])[0]), 2)) for x in (-8, -6, -4, -3, -2, -1, 0, 1, 2)], flush=True)
    new = new.with_columns(pl.Series("score", iso.predict(new["score"].to_numpy())).cast(pl.Float32))
new.sort("score").write_parquet(a.out)
print("words unknown to the training odds:", new.height, "| most decoy-like:", new.sort("score").head(20)["token"].to_list())
