"""Name-replaced records at addresses shared by 2+ S1s (v16; see EXPERIMENTS.md "Shared addresses: sub-numbers and the
city key").

--mode unit: a single-token record (invented name or acronym, no web handle) that carries a unit (Unit / Suite / Apt / # ...)
or a French sub-number (bis / ter / A / B ...), at the house number + street key + city of 2+ S1s, exactly one of which has
the identical unit + sub-number. Train (all owners present): that tenant owns the record 98.8% (US, 768) / 96.3% (India, 217);
without a unit on the record only 62% / 80% (not used). Test: US/India already claim these (0 / 10 left), France 287.
--mode acr_city: the v15 rule (an acronym record at an address shared by 2+ France S1s, exactly one of which has those
initials; train 99.7% US / 100% India) with the city recognised from the city list instead of the set of all non-numeric
address parts. The v15 key missed two thirds of the class because records often name the departement instead of the region
("Loire-Atlantique" for "Pays de la Loire") or drop it. Records whose sub-number or unit differs from that tenant's are skipped.

  python scripts/france_shared_addr.py --data DS --claimed PRED.parquet --mode unit --out fr_unit_adds.parquet
  python scripts/france_shared_addr.py --data DS --claimed PRED.parquet --mode acr_city --out fr_acr_city_shared_adds.parquet

PRED.parquet holds the (s1, m) pairs already claimed (v15b); only unclaimed records are added.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--claimed", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", required=True, choices=["unit", "acr_city"])
a = ap.parse_args()
dd = find_dataset_dir(a.data)
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc", "pllc",
        "incorporated", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le",
        "les", "d", "l", "s", "a"]
STREET_TYPE = ["rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "place", "pl", "allee", "all", "ale", "impasse", "imp", "chemin",
               "ch", "che", "route", "rte", "cours", "crs", "square", "sq", "quai", "qu", "passage", "pass", "cite", "residence", "res", "voie",
               "sentier", "esplanade", "parvis", "promenade", "rond", "point", "hameau", "lieu", "dit", "no", "n", "numero", "de", "du", "des",
               "la", "le", "les", "d", "l", "et", "bis", "ter", "b", "t", "a", "c", "f", "st", "ste", "saint", "sainte"]
FR_CITIES = sorted(["la baule escoublac", "la teste de buch", "lege cap ferret", "hellemmes lille", "saint herblain", "saint nazaire",
                    "dunkerque", "tourcoing", "bordeaux", "merignac", "le clion", "roubaix", "nantes", "calais", "pessac", "pornic",
                    "lille", "lomme"], key=len, reverse=True)
CITY_RE = "(" + "|".join(FR_CITIES) + ")"
UNIT = (r"(?:\b(?:unit|suite|ste|apt|apartment|appt|appartement|flat|fl|floor|room|rm|bldg|building|batiment|block|blk|door|no|plot|shop|"
        r"office|gala|wing)\b\.?\s*#?\s*[a-z0-9-]+|#\s*[a-z0-9-]+)")
fold = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def legal_tokens(col):
    """Name tokens with dotted legal forms (S.A.R.L., S.A.S.U., ...) collapsed to one token."""
    return (fold(col).str.replace_all(r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\b", "sarl").str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\s*\.?\s*u\b", "sasu")
            .str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\b", "sas").str.replace_all(r"\be\s*\.\s*u\s*\.\s*r\s*\.\s*l\b", "eurl")
            .str.replace_all(r"\bs\s*\.\s*a\b\.?", "sa").str.replace_all(r"\bs\s*\.\s*n\s*\.\s*c\b\.?", "snc").str.replace_all(r"\be\s*\.\s*i\b\.?", "ei")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" "))


def street_key(col):
    """Same as final_build.street_key: sorted street tokens of the address component holding the first number."""
    comp = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("")
    s = comp.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    return s.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STREET_TYPE) & ~pl.element().str.contains(r"\d")
                                                          & (pl.element().str.len_chars() >= 3))).list.sort().list.join(" ")


def citykey(col):
    """All non-numeric address parts (US / India city key; the v15 key)."""
    comps = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")).str.normalize("NFKD")
                                                               .str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^a-z]+", " ").str.strip_chars())
    return comps.list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join("|")


def addr_keys(df, france):
    ad = fold("business_address")
    ck = ad.str.replace_all(r"[^a-z]+", " ").str.replace_all(r"\bst\b", "saint").str.extract_all(CITY_RE).list.last() if france else citykey("business_address")
    return df.with_columns(
        ad.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"),
        street_key("business_address").alias("sk"), ck.alias("ck"),
        ad.str.extract_all(UNIT).list.eval(pl.element().str.replace_all(r"[^a-z0-9]", "")).list.sort().list.join("|").alias("unit"),
        ad.str.extract(r"\b\d+\s*(bis|ter|quater|[a-h])\b", 1).fill_null("").alias("sub")
    ).filter(pl.col("hn").is_not_null() & (pl.col("sk") != "") & pl.col("ck").is_not_null() & (pl.col("ck") != ""))


def single_token(df):
    raw = pl.col("business_name").fill_null("").str.strip_chars()
    return df.filter(~raw.str.contains(" ") & ~fold("business_name").str.contains(r"\.com|www|^@|^#") & (raw.str.len_chars() >= 2))


t1 = read_source(os.path.join(dd, "test", "test_source1.tsv"))
t23 = pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)])
claimed = pl.read_parquet(a.claimed, columns=["m"]).unique()
t23 = t23.join(claimed.rename({"m": "entity_id"}), on="entity_id", how="anti")
outs = []
for country in (["France", "US", "India"] if a.mode == "unit" else ["France"]):
    fr = country == "France"
    s1 = addr_keys(t1.filter(pl.col("country") == country), fr).with_columns(pl.len().over("hn", "sk", "ck").alias("nten")).filter(pl.col("nten") >= 2)
    rec = addr_keys(single_token(t23.filter(pl.col("country") == country)), fr)
    s1 = s1.select(pl.col("entity_id").alias("s1"), "hn", "sk", "ck", pl.col("unit").alias("u1"), pl.col("sub").alias("sb1"),
                   legal_tokens("business_name").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != "")))
                   .list.unique(maintain_order=True).list.eval(pl.element().str.slice(0, 1)).list.join("").alias("init"))
    rec = rec.select(pl.col("entity_id").alias("m"), "hn", "sk", "ck", pl.col("unit").alias("u2"), pl.col("sub").alias("sb2"),
                     fold("business_name").str.replace_all(r"[^a-z0-9]", "").alias("t2"))
    if a.mode == "unit":
        j = rec.filter((pl.col("u2") != "") | (pl.col("sb2") != "")).join(s1, on=["hn", "sk", "ck"])
        j = j.with_columns(((pl.col("u1") == pl.col("u2")) & (pl.col("sb1") == pl.col("sb2"))).alias("hit"))
    else:
        j = rec.filter((pl.col("t2").str.len_chars() >= 2) & (pl.col("t2").str.len_chars() <= 5)).join(s1, on=["hn", "sk", "ck"])
        j = j.with_columns(pl.col("init").str.contains(pl.col("t2"), literal=True).alias("hit"))
    j = j.with_columns(pl.col("hit").sum().over("m").alias("nhit")).filter(pl.col("hit") & (pl.col("nhit") == 1))
    if a.mode == "acr_city":   # the record's sub-number / unit must not contradict the tenant's
        j = j.filter((pl.col("sb1") == pl.col("sb2")) & ~((pl.col("u1") != "") & (pl.col("u2") != "") & (pl.col("u1") != pl.col("u2"))))
    print(country, a.mode, "adds:", j.height)
    outs.append(j.select("s1", "m"))
out = pl.concat(outs).unique("m", keep="first", maintain_order=True)
out.write_parquet(a.out); print("written", a.out, out.height)
