"""Build a small, self-contained development subset of train.

Samples a fraction of S1 entities, keeps all of their matched S2/S3 records, and keeps the
same fraction of unmatched S2/S3 records, so the match/distractor ratio is preserved.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_split, read_ground_truth

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.1)
a = ap.parse_args()
d = find_dataset_dir(a.data)
s1, s23 = read_split(d, "train")
edges = read_ground_truth(d)
s1s = s1.filter((pl.col("entity_id").hash(7) % 1000) < a.frac * 1000)
e = edges.filter(pl.col("s1").is_in(s1s["entity_id"].implode()))
matched_any = edges["m"].implode()
un = s23.filter(~pl.col("entity_id").is_in(matched_any) & ((pl.col("entity_id").hash(7) % 1000) < a.frac * 1000))
s23s = pl.concat([s23.filter(pl.col("entity_id").is_in(e["m"].implode())), un])
os.makedirs(a.out, exist_ok=True)
s1s.write_parquet(f"{a.out}/s1.parquet"); s23s.write_parquet(f"{a.out}/s23.parquet"); e.write_parquet(f"{a.out}/edges.parquet")
print(s1s.shape, s23s.shape, e.shape)
