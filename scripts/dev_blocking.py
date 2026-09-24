"""Blocking recall study on the dev sample."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.normalize import normalize_frame
from ber.blocking import generate_candidates
from ber.metric import macro_f05

D = sys.argv[1]
t = time.time()
if not os.path.exists(f"{D}/s1n.parquet"):
    normalize_frame(pl.read_parquet(f"{D}/s1.parquet")).write_parquet(f"{D}/s1n.parquet")
    normalize_frame(pl.read_parquet(f"{D}/s23.parquet")).write_parquet(f"{D}/s23n.parquet")
print("normalize", time.time() - t)
s1 = pl.read_parquet(f"{D}/s1n.parquet"); s23 = pl.read_parquet(f"{D}/s23n.parquet")
edges = pl.read_parquet(f"{D}/edges.parquet")
t = time.time()
cand = generate_candidates(s1, s23, k_tok=int(os.environ.get("KT", 30)), k_chr=int(os.environ.get("KC", 15)))
print("blocking", time.time() - t, cand.shape)
cand = cand.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[cand["qi"].to_numpy()]),
                         pl.Series("m", s23["entity_id"].to_numpy()[cand["ci"].to_numpy()]))
cand.write_parquet(f"{D}/cand.parquet")
lab = cand.join(edges.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left")
tot = edges.height
for r in ["tok", "chr"]:
    for k in [5, 10, 20, 30]:
        print(r, k, "recall", lab.filter((pl.col(f"{r}_rank") <= k) & (pl.col("y") == 1)).height / tot)
print("union recall", lab["y"].sum() / tot, "cands/S1", cand.height / s1.height)
ids = s1.select(pl.col("entity_id").alias("s1"), "country")
print("oracle ceiling", macro_f05(lab.filter(pl.col("y") == 1), edges, ids, by="country"))
