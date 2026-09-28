"""Merge the phonetic and same-name route adds into one route pair file for final_build.py --extra_pairs
(src/artifacts/routes/v11/phonetic_samename_adds.parquet).

Phonetic route: route_adds_test.parquet of scripts/phonetic_route_score.py at prob >= 0.70 (as scripts/phonetic_route_build.py
applies it). Same-name route: route_adds_test_v2.parquet of scripts/samename_route_score.py at prob >= 0.55 (the threshold its
gate chose). Both only on records the base build leaves unclaimed and
S1 of countries with training labels; one S1 per record, the higher probability wins (also across the two routes). The result
keeps the records that --claimed (the build the file is added to; ours: v4) leaves unclaimed.

  python scripts/phonetic_samename_merge.py --data <D> --phonetic work/phonetic_score/route_adds_test.parquet \\
      --samename work/samename_score/route_adds_test_v2.parquet --base_matching BASE/matching_results.tsv \\
      --claimed V4/matching_results.tsv --out phonetic_samename_adds.parquet

On our runs this gives 7,155 of the 7,156 pairs of the shipped file. The shipped file was cut from the merged
phonetic / same-name / family-completion file (every pair it adds outside the base candidate lists), so it also holds one family-completion pair
that lies outside those lists (S1-350491974, S2-172056214).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dataset dir (train/ and test/)")
ap.add_argument("--phonetic", required=True, help="route_adds_test.parquet of scripts/phonetic_route_score.py")
ap.add_argument("--samename", required=True, help="route_adds_test_v2.parquet of scripts/samename_route_score.py")
ap.add_argument("--phonetic_t", type=float, default=0.70)
ap.add_argument("--samename_t", type=float, default=0.55)
ap.add_argument("--base_matching", required=True, help="matching_results.tsv of the base build the routes were scored against")
ap.add_argument("--claimed", required=True, help="matches of the build the file is added to (matching_results.tsv or an (s1, m) parquet)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dd = find_dataset_dir(args.data)


def pairs(path):
    """(s1, m) pairs of a matching_results.tsv or a parquet with s1, m columns."""
    if path.endswith(".parquet"):
        return pl.read_parquet(path, columns=["s1", "m"])
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (d.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
            .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))


def best(d):
    """One S1 per record: the pair with the highest probability."""
    return d.filter(pl.col("prob") == pl.col("prob").max().over("m")).unique("m", keep="first")


te = read_source(os.path.join(dd, "test", "test_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
trc = read_source(os.path.join(dd, "train", "train_source1.tsv")).select("country").unique()
lab = te.join(trc, on="country").select("s1")                          # S1 of countries with training labels
base = pairs(args.base_matching).select("m").unique()
phon = pl.read_parquet(args.phonetic, columns=["s1", "m", "prob", "claimed"])  # claimed: record matched in the base build
phon = best(phon.filter((pl.col("prob") >= args.phonetic_t) & (pl.col("claimed") == 0))).join(lab, on="s1")
same = pl.read_parquet(args.samename, columns=["s1", "m", "prob"])   # its claimed column refers to base + phonetic, so recompute
same = best(same.filter(pl.col("prob") >= args.samename_t).join(base, on="m", how="anti")).join(lab, on="s1")
both = best(pl.concat([phon.select("s1", "m", "prob"), same.select("s1", "m", "prob")]))
out = both.join(pairs(args.claimed).select("m").unique(), on="m", how="anti").select("s1", "m").sort("s1", "m")
print("phonetic adds", phon.height, "same-name adds", same.height, "merged", both.height, "unclaimed in --claimed", out.height)
out.write_parquet(args.out)
