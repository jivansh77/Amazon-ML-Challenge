"""Fit the Indic->Latin dictionary on the full training ground truth and save it as JSON."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_split, read_ground_truth
from ber.translit import fit_indic_dictionary

d = find_dataset_dir(sys.argv[1]); out = sys.argv[2]
t = time.time()
s1, s23 = read_split(d, "train")
e = read_ground_truth(d)
s23 = s23.filter(pl.col("business_name").str.contains(r"[ऀ-ൿ]") | pl.col("business_address").str.contains(r"[ऀ-ൿ]"))
p = (e.join(s23.select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="m")
      .join(s1.select(pl.col("entity_id").alias("s1"), "business_name", "business_address"), on="s1"))
print("pairs with indic", p.height, time.time() - t)
dn = fit_indic_dictionary(zip(p["business_name"], p["n2"]))
da = fit_indic_dictionary(zip(p["business_address"], p["a2"]))
for k, v in da.items():
    dn.setdefault(k, v)
json.dump(dn, open(out, "w"), ensure_ascii=False)
print(len(dn), "words", time.time() - t)
