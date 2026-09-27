"""Append route pair files to a finished build (the v16 / v17 steps): same semantics as final_build --extra_pairs
(unclaimed records only, one S1 per record, earlier file wins); the pairs are added to the candidate file too.

  python src/append_routes.py <base_dir with pred.parquet (+ cand.parquet)> <out_dir> <route1.parquet,route2.parquet,...>
"""
import sys, os
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.io import write_id_lists, read_source
if len(sys.argv) < 4:
    sys.exit(__doc__)
base, out, files = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
os.makedirs(out, exist_ok=True)
pred = pl.read_parquet(f"{base}/pred.parquet")
if os.path.exists(f"{base}/cand.parquet"):
    cand = pl.read_parquet(f"{base}/cand.parquet")
else:   # final_build.py output: read its candidate file
    c = pl.read_csv(f"{base}/candidate_pairs.tsv", separator="\t", quote_char=None, schema_overrides={"source1_entity_id": pl.Utf8, "candidate_entity_ids": pl.Utf8})
    cand = c.select(pl.col("source1_entity_id").alias("s1"), pl.col("candidate_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
t1 = read_source(os.environ.get("BER_TEST_S1", "dataset/test/test_source1.tsv"))
xp = pl.concat([pl.read_parquet(f, columns=["s1","m"]).with_columns(pl.lit(i).alias("_pri")) for i, f in enumerate(files)])
xp = xp.sort("_pri", maintain_order=True).unique("m", keep="first", maintain_order=True).drop("_pri")
xtra = xp.join(pred.select("m").unique(), on="m", how="anti").join(t1.select(pl.col("entity_id").alias("s1")), on="s1")
new = pl.concat([pred, xtra]).unique(maintain_order=True); nc = pl.concat([cand, xtra]).unique(maintain_order=True)
assert new.select("m").n_unique() == new.height
write_id_lists(f"{out}/matching_results.tsv", t1["entity_id"].to_list(), new, "matched_entity_ids")
write_id_lists(f"{out}/candidate_pairs.tsv", t1["entity_id"].to_list(), nc, "candidate_entity_ids")
new.write_parquet(f"{out}/pred.parquet"); nc.write_parquet(f"{out}/cand.parquet")
print("adds:", xtra.height, xtra.join(t1.select(pl.col("entity_id").alias("s1"), "country"), on="s1").group_by("country").len().sort("country").rows(), "matches:", new.height, "cands:", nc.height)
