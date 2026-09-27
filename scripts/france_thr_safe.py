"""France pairs a lower threshold would add, kept only where the pick is not a guess (v15b).

  python scripts/france_thr_safe.py --data DS --base BASE/pred.parquet --low LOW/pred.parquet --out ADDS.parquet

LOW is the same build as BASE with --thr France=0.85. A pair it adds is kept when its record is unclaimed in BASE and
either (a) the record has no address and its core name is the core name of exactly one France S1, the one it goes to, or
(b) the S1 is the only France S1 at its house number + street key + non-numeric address parts. The dropped rest are
same-name ties and tenant picks at shared addresses, which are coin flips (see EXPERIMENTS.md, v15b).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--base", required=True); ap.add_argument("--low", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
dd = find_dataset_dir(a.data)
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc", "pllc",
        "incorporated", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le",
        "les", "d", "l", "s", "a"]
STREET_TYPE = ["rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "place", "pl", "allee", "all", "ale", "impasse", "imp", "chemin",
               "ch", "che", "route", "rte", "cours", "crs", "square", "sq", "quai", "qu", "passage", "pass", "cite", "residence", "res", "voie",
               "sentier", "esplanade", "parvis", "promenade", "rond", "point", "hameau", "lieu", "dit", "no", "n", "numero", "de", "du", "des",
               "la", "le", "les", "d", "l", "et", "bis", "ter", "b", "t", "a", "c", "f", "st", "ste", "saint", "sainte"]
norm = lambda c: (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
                  .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
ck = norm("business_name").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique().list.sort().list.join(" ")
comp = pl.col("business_address").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("")
sk = (comp.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
      .str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STREET_TYPE) & ~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() >= 3)))
      .list.sort().list.join(" "))
citykey = (pl.col("business_address").fill_null("").str.split(",").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")).str.normalize("NFKD")
           .str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^a-z]+", " ").str.strip_chars()).list.eval(pl.element().filter(pl.element() != ""))
           .list.unique().list.sort().list.join("|"))
hn = norm("business_address").str.extract(r"(\d+)", 1).str.strip_chars_start("0")
s1 = read_source(os.path.join(dd, "test", "test_source1.tsv")).filter(pl.col("country") == "France").select(
    pl.col("entity_id").alias("s1"), ck.alias("ck1"), hn.alias("hn"), sk.alias("sk"), citykey.alias("city"))
s1 = s1.with_columns(pl.len().over("ck1").alias("nsn"),
                     pl.when(pl.col("hn").is_not_null() & (pl.col("sk") != "") & (pl.col("city") != "")).then(pl.len().over("hn", "sk", "city")).alias("n_addr"))
rec = pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)]).filter(pl.col("country") == "France").select(
    pl.col("entity_id").alias("m"), ck.alias("ck2"), (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("ea"))
base = pl.read_parquet(a.base, columns=["s1", "m"]); low = pl.read_parquet(a.low, columns=["s1", "m"])
add = low.join(base, on=["s1", "m"], how="anti").join(base.select("m").unique(), on="m", how="anti").join(s1, on="s1").join(rec, on="m")
ea_ok = pl.col("ea") & (pl.col("ck2") == pl.col("ck1")) & (pl.col("nsn") == 1)
addr_ok = ~pl.col("ea") & (pl.col("n_addr") == 1)
out = add.filter(ea_ok | addr_ok).select("s1", "m").unique("m")
out.write_parquet(a.out)
print("France 0.85 adds:", add.height, "kept:", out.height, "(no-address unique name:", add.filter(ea_ok).height, ", S1 alone at its address:", add.filter(addr_ok).height, ")")
