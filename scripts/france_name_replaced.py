"""France name-replaced copies at a unique S1 address (see EXPERIMENTS.md, "France invented-name records").

The generator renames some copies with an invented single-token name ("Onyxwex") or the S1's acronym ("CG" for
"Chasseurs Groupe SARL") and keeps the address. In train, such a record at the house number + street of exactly one S1 is
that S1's match 94% (invented; the rest are same-street collisions in another city) and 100% (acronym). US/India test
claims those classes at the base rate; France claims 60% / 83% because its blocking never proposes most of them.
Adds, for France only: record unclaimed, same house number + street key + city as exactly one France S1, and the record
name is an invented single token (no core word shared with the S1) or an acronym of the S1's core words.

  python scripts/france_name_replaced.py --data DS --claimed PRED.parquet --out_inv INV.parquet --out_acr ACR.parquet
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--claimed", required=True)
ap.add_argument("--out_inv", required=True); ap.add_argument("--out_acr", required=True)
a = ap.parse_args()
dd = find_dataset_dir(a.data)
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc", "pllc",
        "incorporated", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le",
        "les", "d", "l", "s", "a"]
STREET_TYPE = ["rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "place", "pl", "allee", "all", "ale", "impasse", "imp", "chemin",
               "ch", "che", "route", "rte", "cours", "crs", "square", "sq", "quai", "qu", "passage", "pass", "cite", "residence", "res", "voie",
               "sentier", "esplanade", "parvis", "promenade", "rond", "point", "hameau", "lieu", "dit", "no", "n", "numero", "de", "du", "des",
               "la", "le", "les", "d", "l", "et", "bis", "ter", "b", "t", "a", "c", "f", "st", "ste", "saint", "sainte"]
CITIES = sorted(["la baule escoublac", "la teste de buch", "lege cap ferret", "hellemmes lille", "saint herblain", "saint nazaire",
                 "dunkerque", "tourcoing", "bordeaux", "merignac", "le clion", "roubaix", "nantes", "calais", "pessac", "pornic", "lille",
                 "lomme"], key=len, reverse=True)
CITY_RE = "(" + "|".join(CITIES) + ")"
fold = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
norm = lambda c: fold(c).str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def street_key(col):
    """Same as final_build.street_key: sorted street tokens of the address component holding the first number."""
    comp = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("")
    s = comp.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    return s.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STREET_TYPE) & ~pl.element().str.contains(r"\d")
                                                          & (pl.element().str.len_chars() >= 3))).list.sort().list.join(" ")


def keyed(df):
    return df.with_columns(
        norm("business_name").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique(maintain_order=True).alias("core"),
        norm("business_name").str.split(" ").alias("toks"),
        norm("business_address").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"),
        street_key("business_address").alias("sk"),
        fold("business_address").str.replace_all(r"[^a-z]+", " ").str.extract_all(CITY_RE).list.last().alias("city"),
    ).filter(pl.col("hn").is_not_null() & (pl.col("sk") != ""))


s1 = keyed(read_source(os.path.join(dd, "test", "test_source1.tsv")).filter(pl.col("country") == "France"))
s23 = keyed(pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)]).filter(pl.col("country") == "France"))
uniq = s1.group_by("hn", "sk").agg(pl.len().alias("n"), pl.col("entity_id").first().alias("s1"), pl.col("core").first().alias("c1"),
                                   pl.col("toks").first().alias("t1"), pl.col("city").first().alias("city1")).filter(pl.col("n") == 1)
j = s23.select(pl.col("entity_id").alias("m"), "hn", "sk", "city", pl.col("toks").alias("t2"), pl.col("business_name").alias("raw")).join(uniq, on=["hn", "sk"])
j = j.filter(pl.col("city") == pl.col("city1")).join(pl.read_parquet(a.claimed, columns=["m"]).unique(), on="m", how="anti")
raw = pl.col("raw").fill_null("").str.strip_chars()
t2 = fold("raw").str.replace_all(r"[^a-z0-9]", "")
init = pl.col("c1").list.eval(pl.element().str.slice(0, 1)).list.join("")
web = fold("raw").str.contains(r"\.com|www|^@|^#")
acr = j.filter(~web & ~raw.str.contains(" ") & (t2.str.len_chars() >= 2) & (t2.str.len_chars() <= 5) & init.str.contains(t2, literal=True))
inv = j.filter(~fold("raw").str.contains(r"\.com|www") & ~raw.str.contains(" ") & (raw.str.len_chars() >= 5)
               & (pl.col("t1").list.set_intersection(pl.col("t2")).list.len() == 0)).join(acr.select("m"), on="m", how="anti")
inv.select("s1", "m").unique("m").write_parquet(a.out_inv); acr.select("s1", "m").unique("m").write_parquet(a.out_acr)
print("France invented-name adds:", inv.select("m").n_unique(), " acronym adds:", acr.select("m").n_unique())
