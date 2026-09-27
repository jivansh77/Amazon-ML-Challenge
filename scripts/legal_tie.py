"""Legal-form tie-break for address-less records whose core name is shared by 2+ S1s of the country (v17; see EXPERIMENTS.md
"Legal-form tie-break").

The record carries a legal form L (canonicalised: Incorporated -> Inc, Private -> Pvt, ...) and exactly one of the same-name
S1s has L. Train precision by the number of same-name S1s without any legal form (n0): n0 = 0 95.1% (US, 3,810) / 96.7%
(India, 365); n0 = 1 90.3% / 91.0%; n0 = 2 83.9% / 83.5%; n0 >= 3 56-61% (not used). France copies essentially never swap the
legal form (0.02% of claimed same-name pairs vs 2.6% US), so the rule is at least as precise there; France uses n0 <= 2,
US/India n0 <= 1.
Case B (France only, --case_b): the record's legal form is on none of the same-name S1s and exactly one of them has no legal form;
France copies never swap the form, so the owner is that S1 (the noise added the form) unless the record belongs to another name.

  python scripts/legal_tie.py --data DS --claimed PRED.parquet --out legal_tie_adds.parquet
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl
from ber.io import find_dataset_dir, read_source

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--claimed", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max_n0", default="France=2,US=1,India=1")
ap.add_argument("--case_b", default="France", help="countries where a record whose legal form no same-name S1 has goes to the one S1 without a legal form (legal form added by the noise); France only: US 80%% / India 39%% in train, where copies swap legal forms")
a = ap.parse_args()
dd = find_dataset_dir(a.data)
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc", "pllc",
        "incorporated", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le",
        "les", "d", "l", "s", "a"]
CANON = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt"}
LEGAL = ["inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "pa", "plc", "pllc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei"]
fold = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def legal_tokens(col):
    """Name tokens with dotted legal forms (S.A.R.L., S.A.S.U., ...) collapsed to one token."""
    return (fold(col).str.replace_all(r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\b", "sarl").str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\s*\.?\s*u\b", "sasu")
            .str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\b", "sas").str.replace_all(r"\be\s*\.\s*u\s*\.\s*r\s*\.\s*l\b", "eurl")
            .str.replace_all(r"\bs\s*\.\s*a\b\.?", "sa").str.replace_all(r"\bs\s*\.\s*n\s*\.\s*c\b\.?", "snc").str.replace_all(r"\be\s*\.\s*i\b\.?", "ei")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" "))


core = legal_tokens("business_name").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique().list.sort().list.join(" ")
legal = legal_tokens("business_name").list.eval(pl.element().replace(CANON).filter(pl.element().is_in(LEGAL))).list.unique().list.sort().list.join(",")
max_n0 = {k: int(v) for k, v in (x.split("=") for x in a.max_n0.split(","))}
s1 = read_source(os.path.join(dd, "test", "test_source1.tsv")).with_columns(core.alias("ck"), legal.alias("l1"))
ties = s1.group_by("country", "ck").agg(pl.len().alias("nsn")).filter((pl.col("nsn") >= 2) & (pl.col("ck") != ""))
rec = pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)])
rec = rec.filter(pl.col("business_address").str.strip_chars() == "").join(pl.read_parquet(a.claimed, columns=["m"]).unique().rename({"m": "entity_id"}),
                                                                         on="entity_id", how="anti")
rec = rec.with_columns(core.alias("ck"), legal.alias("l2")).filter(pl.col("l2") != "").join(ties, on=["country", "ck"])
j = rec.select(pl.col("entity_id").alias("m"), "country", "ck", "l2").join(s1.select(pl.col("entity_id").alias("s1"), "country", "ck", "l1"),
                                                                            on=["country", "ck"])
j = j.with_columns((pl.col("l1") == pl.col("l2")).alias("eq"), (pl.col("l1") == "").alias("none"))
j = j.with_columns(pl.col("eq").sum().over("m").alias("nL"), pl.col("none").sum().over("m").alias("n0"))
ja = j.filter(pl.col("eq") & (pl.col("nL") == 1) & (pl.col("n0") <= pl.col("country").replace_strict(max_n0, default=-1, return_dtype=pl.Int64)))
jb = j.filter(pl.col("country").is_in([c for c in a.case_b.split(",") if c]) & (pl.col("nL") == 0) & pl.col("none") & (pl.col("n0") == 1))
print("case A:", ja.group_by("country", "n0").len().sort("country", "n0").rows(), " case B:", jb.group_by("country").len().rows())
out = pl.concat([ja.select("s1", "m"), jb.select("s1", "m")]).unique("m")
out.write_parquet(a.out); print("legal-form tie-break adds:", out.height)
