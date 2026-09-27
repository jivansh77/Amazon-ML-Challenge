"""France name-replaced copies at a unique S1 address (see EXPERIMENTS.md, "France invented-name records").

The generator renames some copies with an invented single-token name ("Onyxwex") or the S1's acronym ("CG" for
"Chasseurs Groupe SARL") and keeps the address. In train, such a record at the house number + street of exactly one S1 is
that S1's match 94% (invented; the rest are same-street collisions in another city) and 100% (acronym). US/India test
claims those classes at the base rate; France claims 60% / 83% because its blocking never proposes most of them.
Adds, for France only: record unclaimed, same house number + street key + city as exactly one France S1, and the record
name is an invented single token (no core word shared with the S1) or an acronym of the S1's core words.

  python scripts/france_name_replaced.py --data DS --claimed PRED.parquet --out_inv INV.parquet --out_acr ACR.parquet

--unique_by city: the S1 only has to be the one S1 at that house number + street key in its city (v13). The default
(country) also needs it to be the only France S1 at that house number + street key in any city (v10s/v11). Same number
and street in another city is common in France (Rue Voltaire, Boulevard Victor Hugo); in train such records are still the
city's S1 match 94% (US invented) to 100% (acronyms). --drop_foreign_handles drops invented-name records that are web
handles (@x, #x, xcom) containing no S1 word and not the S1's acronym (train: 30-51% matches).

--mode fuzzy_street (v12): acronym / invented-name records (no web handles) on the same house number + city as exactly one
France S1 whose street key differs only by a typo (rapidfuzz ratio 85-99, no other such S1 at >= 60). Train: 95% (US) /
86% (India) matches. Writes --out.
--mode inv_legal (v13): an invented word plus a legal form ("Nexaria Co"): one core word, in no France S1 name, no word
shared with the S1, at the house number + street key + city (all non-numeric address parts) of exactly one France S1.
Train: 96% (US) / 99% (India) matches. Writes --out.
--mode acr_shared (v15): an acronym record at an address (house number + street key + all non-numeric parts) shared by 2+ France
S1 where exactly one of them has those initials. Train: 99.7% (US) / 100% (India); test claims US 98%, India 99%, France 78%.
Writes --out.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--claimed", required=True)
ap.add_argument("--mode", default="exact", choices=["exact", "fuzzy_street", "inv_legal", "acr_shared"])
ap.add_argument("--out_inv"); ap.add_argument("--out_acr"); ap.add_argument("--out")
ap.add_argument("--unique_by", default="country", choices=["country", "city"])
ap.add_argument("--drop_foreign_handles", action="store_true")
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


def read_fr(ks):
    return pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in ks]).filter(pl.col("country") == "France")


claimed = pl.read_parquet(a.claimed, columns=["m"]).unique()
init = pl.col("c1").list.eval(pl.element().str.slice(0, 1)).list.join("")

if a.mode == "fuzzy_street":
    from rapidfuzz import fuzz, process
    s1 = keyed(read_fr([1])).filter(pl.col("city").is_not_null()).select(pl.col("entity_id").alias("s1"), "hn", "sk", "city", pl.col("toks").alias("t1"), pl.col("core").alias("c1"))
    s23 = read_fr([2, 3]).join(claimed.rename({"m": "entity_id"}), on="entity_id", how="anti")
    raw = pl.col("business_name").fill_null("").str.strip_chars()
    s23 = keyed(s23.filter(~raw.str.contains(" ") & ~raw.str.to_lowercase().str.contains(r"\.com|www|^@|^#"))).filter(pl.col("city").is_not_null())
    j = s23.select(pl.col("entity_id").alias("m"), "hn", "sk", "city", pl.col("toks").alias("t2"), pl.col("business_name").alias("n2")).join(s1, on=["hn", "city"], suffix="_1")
    j = j.with_columns(pl.Series("sr", process.cpdist(j["sk"].to_list(), j["sk_1"].to_list(), scorer=fuzz.ratio, workers=4)))
    t2 = pl.col("n2").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")
    j = j.filter(((t2.str.len_chars() >= 2) & (t2.str.len_chars() <= 5) & init.str.contains(t2, literal=True))
                 | ((pl.col("n2").str.len_chars() >= 5) & (pl.col("t1").list.set_intersection(pl.col("t2")).list.len() == 0)))
    j = j.with_columns((pl.col("sr") >= 60).sum().over("m").alias("n60"))
    out = j.filter((pl.col("sr") >= 85) & (pl.col("sr") < 100) & (pl.col("n60") == 1)).select("s1", "m").unique("m")
    out.write_parquet(a.out); print("France fuzzy-street name-replaced adds:", out.height)
    sys.exit()

if a.mode == "acr_shared":
    def citykey(col):
        comps = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")).str.normalize("NFKD")
                                                                   .str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^a-z]+", " ").str.strip_chars())
        return comps.list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join("|")
    s1 = keyed(read_fr([1])).with_columns(citykey("business_address").alias("city")).filter(pl.col("city") != "")
    s1 = s1.select(pl.col("entity_id").alias("s1"), "hn", "sk", "city", pl.col("core").alias("c1")).with_columns(
        pl.len().over("hn", "sk", "city").alias("nS1"), pl.col("c1").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("init"))
    raw = pl.col("business_name").fill_null("").str.strip_chars()
    s23 = read_fr([2, 3]).join(claimed.rename({"m": "entity_id"}), on="entity_id", how="anti")
    s23 = keyed(s23.filter(~raw.str.contains(" ") & ~raw.str.to_lowercase().str.contains(r"\.com|www|^@|^#")))
    s23 = s23.with_columns(citykey("business_address").alias("city"), fold("business_name").str.replace_all(r"[^a-z0-9]", "").alias("t2"))
    s23 = s23.filter((pl.col("city") != "") & (pl.col("t2").str.len_chars() >= 2) & (pl.col("t2").str.len_chars() <= 5)).select(pl.col("entity_id").alias("m"), "hn", "sk", "city", "t2")
    j = s23.join(s1, on=["hn", "sk", "city"]).with_columns(pl.col("init").str.contains(pl.col("t2"), literal=True).alias("hit"))
    j = j.with_columns(pl.col("hit").sum().over("m").alias("nhit")).filter(pl.col("hit") & (pl.col("nhit") == 1) & (pl.col("nS1") >= 2))
    out = j.select("s1", "m").unique("m"); out.write_parquet(a.out); print("France acronym adds at shared addresses:", out.height)
    sys.exit()

if a.mode == "inv_legal":
    def citykey(col):
        comps = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")).str.normalize("NFKD")
                                                                   .str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^a-z]+", " ").str.strip_chars())
        return comps.list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join("|")
    s1 = keyed(read_fr([1])).with_columns(citykey("business_address").alias("city")).filter(pl.col("city") != "")
    s1 = s1.select(pl.col("entity_id").alias("s1"), "hn", "sk", "city", pl.col("toks").alias("t1"), pl.col("core").alias("c1"))
    vocab = s1.select(pl.col("t1").explode().unique().alias("tok"))
    uniq = s1.with_columns(pl.len().over("hn", "sk", "city").alias("n")).filter(pl.col("n") == 1)
    raw = pl.col("business_name").fill_null("").str.strip_chars()
    s23 = keyed(read_fr([2, 3]).filter(raw.str.contains(" ") & ~raw.str.to_lowercase().str.contains(r"\.com|www|^@|^#")))
    s23 = s23.with_columns(citykey("business_address").alias("city")).filter((pl.col("city") != "") & (pl.col("core").list.len() == 1))
    s23 = s23.select(pl.col("entity_id").alias("m"), "hn", "sk", "city", pl.col("toks").alias("t2"), pl.col("core").list.first().alias("tok"))
    j = s23.join(vocab, on="tok", how="anti").join(uniq, on=["hn", "sk", "city"]).join(claimed, on="m", how="anti")
    is_acr = (pl.col("tok").str.len_chars() >= 2) & (pl.col("tok").str.len_chars() <= 5) & init.str.contains(pl.col("tok"), literal=True)
    j = j.filter(~is_acr & (pl.col("tok").str.len_chars() >= 5) & (pl.col("t1").list.set_intersection(pl.col("t2")).list.len() == 0))
    out = j.select("s1", "m").unique("m"); out.write_parquet(a.out); print("France invented-word + legal-form adds:", out.height)
    sys.exit()

s1 = keyed(read_fr([1]))
s23 = keyed(read_fr([2, 3]))
if a.unique_by == "country":
    uniq = s1.group_by("hn", "sk").agg(pl.len().alias("n"), pl.col("entity_id").first().alias("s1"), pl.col("core").first().alias("c1"),
                                       pl.col("toks").first().alias("t1"), pl.col("city").first().alias("city1")).filter(pl.col("n") == 1)
else:
    # one S1 at (number, street, city); skip keys where an S1 at that number + street has no recognised city (it could be the same city)
    nocity = s1.filter(pl.col("city").is_null()).select("hn", "sk").unique()
    uniq = (s1.filter(pl.col("city").is_not_null()).group_by("hn", "sk", "city")
            .agg(pl.len().alias("n"), pl.col("entity_id").first().alias("s1"), pl.col("core").first().alias("c1"), pl.col("toks").first().alias("t1"))
            .filter(pl.col("n") == 1).join(nocity, on=["hn", "sk"], how="anti").with_columns(pl.col("city").alias("city1")).drop("city"))
j = s23.select(pl.col("entity_id").alias("m"), "hn", "sk", "city", pl.col("toks").alias("t2"), pl.col("business_name").alias("raw")).join(uniq, on=["hn", "sk"])
j = j.filter(pl.col("city") == pl.col("city1")).join(claimed, on="m", how="anti")
raw = pl.col("raw").fill_null("").str.strip_chars()
t2 = fold("raw").str.replace_all(r"[^a-z0-9]", "")
web = fold("raw").str.contains(r"\.com|www|^@|^#")
acr = j.filter(~web & ~raw.str.contains(" ") & (t2.str.len_chars() >= 2) & (t2.str.len_chars() <= 5) & init.str.contains(t2, literal=True))
inv = j.filter(~fold("raw").str.contains(r"\.com|www") & ~raw.str.contains(" ") & (raw.str.len_chars() >= 5)
               & (pl.col("t1").list.set_intersection(pl.col("t2")).list.len() == 0)).join(acr.select("m"), on="m", how="anti")
if a.drop_foreign_handles:
    # web handles (@x, #x, ...com/fr/net/org) with no S1 word inside and not the S1's acronym: 30-51% matches in train
    flat = fold("raw").str.replace_all(r"[^a-z0-9]", "")
    handle = fold("raw").str.strip_chars().str.contains(r"^[@#]") | flat.str.contains(r"(com|fr|net|org)$")
    has_word = pl.struct(pl.col("c1"), flat.alias("f")).map_elements(lambda r: any(len(t) >= 3 and t in r["f"] for t in (r["c1"] or [])), return_dtype=pl.Boolean)
    stem = flat.str.replace(r"(com|fr|net|org)$", "")
    is_acr = (stem.str.len_chars() >= 2) & (stem.str.len_chars() <= 5) & init.str.contains(stem, literal=True)
    inv = inv.filter(~(handle & ~has_word & ~is_acr))
inv.select("s1", "m").unique("m").write_parquet(a.out_inv); acr.select("s1", "m").unique("m").write_parquet(a.out_acr)
print("France invented-name adds:", inv.select("m").n_unique(), " acronym adds:", acr.select("m").n_unique())
