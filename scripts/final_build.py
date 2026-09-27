"""Final test decoding: average of pipeline runs + cross-encoder ensemble, per-country thresholds, then the
label-free France generator rules (suffix adds, changed-house-number drops, category-swap drops).

  python scripts/final_build.py --data DS --runs RUN1,RUN2 --ce CE1:W1,CE2:W2 --ce_fr FRCE1,FRCE2 --out OUT

--runs   dirs with test_scores.parquet and fr_test_scores.parquet (France scored with the calibrated French words);
         a pair's p is the mean over the runs that scored it.
--ce     cross-encoder test files (s1, m, ce) with weights: logit average over the files that scored the pair.
--ce_fr  French cross-encoder files; they replace the --ce ensemble for their pairs (France).
Blend sigmoid(w logit(p) + (1 - w) logit(ce)); pairs without a cross-encoder score keep p.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, polars as pl
from rapidfuzz import fuzz, process
from ber.io import find_dataset_dir, read_source, write_id_lists
from ber.pipeline import decode_by_country, log

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ce", required=True); ap.add_argument("--ce_fr", required=True)
ap.add_argument("--w", type=float, default=0.6)
ap.add_argument("--thr", default="US=0.8,India=0.8,France=0.87")
ap.add_argument("--rules", default="A2,HN_A,D", help="France rules to apply (empty = none)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dd = find_dataset_dir(a.data)
L = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
tec = read_source(os.path.join(dd, "test", "test_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
fr_ids = tec.filter(pl.col("country") == "France").select("s1")


def mean_runs(name):
    runs = [pl.read_parquet(f"{r}/{name}", columns=["s1", "m", "p"]).rename({"p": f"p{i}"}) for i, r in enumerate(a.runs.split(","))]
    j = runs[0]
    for r in runs[1:]:
        j = j.join(r, on=["s1", "m"], how="full", coalesce=True)
    cols = [c for c in j.columns if c.startswith("p")]
    return j.select("s1", "m", pl.mean_horizontal(cols).cast(pl.Float64).alias("p"))


ts = mean_runs("test_scores.parquet")
ts = pl.concat([ts.join(fr_ids, on="s1", how="anti"), mean_runs("fr_test_scores.parquet").join(fr_ids, on="s1")])
log("test pairs", ts.height, "runs", a.runs)
# cross-encoder ensemble (logit average over available files), French CE overrides for its pairs
ens = None
for i, item in enumerate(a.ce.split(",")):
    path, w = item.rsplit(":", 1)
    c = pl.read_parquet(path, columns=["s1", "m", "ce"]).unique(["s1", "m"]).select("s1", "m", (float(w) * L("ce")).alias(f"l{i}"), pl.lit(float(w)).alias(f"w{i}"))
    ens = c if ens is None else ens.join(c, on=["s1", "m"], how="full", coalesce=True)
lc = [c for c in ens.columns if c.startswith("l")]; wc = [c for c in ens.columns if c.startswith("w")]
ens = ens.select("s1", "m", (pl.sum_horizontal(lc) / pl.sum_horizontal(wc)).cast(pl.Float64).alias("lce"))
fr = pl.concat([pl.read_parquet(p, columns=["s1", "m", "ce"]).with_columns(pl.col("ce").cast(pl.Float64)) for p in a.ce_fr.split(",")]).unique(["s1", "m"], keep="last").select("s1", "m", L("ce").cast(pl.Float64).alias("lce"))
ens = pl.concat([ens.join(fr, on=["s1", "m"], how="anti"), fr])
te = ts.join(ens, on=["s1", "m"], how="left").with_columns(
    p=pl.when(pl.col("lce").is_null()).then(pl.col("p")).otherwise(1 / (1 + (-(a.w * L("p") + (1 - a.w) * pl.col("lce"))).exp()))).drop("lce")
thr = {k: float(v) for k, v in (x.split("=") for x in a.thr.split(","))}
pred = decode_by_country(te, thr, tec)
log("decoded:", pred.height, "pairs; thresholds", thr)
pred.write_parquet(f"{a.out}/pred_before_rules.parquet")

# ---- France generator rules (label-free; see EXPERIMENTS.md "France generator rules") ----
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc",
        "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le", "les", "d", "l", "s", "a"]
SUFFIX = ["fils", "groupe", "developpement", "associes"]
CATEGORY = ['agricole', 'amicale', 'amis', 'anciens', 'atelier', 'ateliers', 'auto', 'automobile', 'cafe', 'club', 'collectif', 'college',
            'comite', 'conseil', 'culture', 'culturelle', 'danse', 'ecole', 'ehpad', 'elementaire', 'energie', 'fetes', 'foyer', 'gestion',
            'groupement', 'institut', 'jeunes', 'loisirs', 'lycee', 'maison', 'maternelle', 'medical', 'medico', 'musique', 'parents',
            'patrimoine', 'pharmacie', 'primaire', 'residence', 'sante', 'section', 'societe', 'soins', 'sportive', 'theatre', 'union', 'france']
rules = [r for r in a.rules.split(",") if r]
if rules:
    def norm(col):
        return (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
                .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
    def prep(df):
        df = df.with_columns(norm("business_name").alias("n"), norm("business_address").alias("a"))
        return df.with_columns(
            pl.col("n").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique().alias("core"),
            pl.col("a").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"))
    t1 = prep(read_source(os.path.join(dd, "test", "test_source1.tsv")))
    t23 = prep(pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)]))
    s1c = t1.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("n").alias("n2"), pl.col("a").alias("a2"),
                                                            pl.col("core").alias("core2"), pl.col("hn").alias("hn2"))
    cp = te.select("s1", "m", pl.col("p").alias("bp")).join(s1c, on="s1").join(
        t23.select(pl.col("entity_id").alias("m"), "n", "a", "core", "hn"), on="m")
    ra = process.cpdist(cp["a"].to_list(), cp["a2"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)
    rn = process.cpdist(cp["n"].to_list(), cp["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    cp = cp.with_columns(pl.Series("ra", ra, pl.Float64), pl.Series("rn", rn, pl.Float64)).with_columns(
        pl.col("core2").list.set_difference(pl.col("core")).list.len().alias("nd"),
        pl.col("core").list.set_difference(pl.col("core2")).alias("added"),
        pl.when(pl.col("hn").is_null() | pl.col("hn2").is_null()).then(pl.lit(None, pl.Boolean)).otherwise(pl.col("hn") == pl.col("hn2")).alias("same_hn"),
    ).with_columns(pl.col("added").list.len().alias("na"), pl.col("added").list.first().alias("aw")).drop("n", "a", "n2", "a2", "core", "core2", "added")
    if "A2" in rules:      # French noise suffix added, number kept, record unclaimed: its best S1 (by name + address similarity)
        best = cp.with_columns((pl.col("rn") + pl.col("ra")).alias("sc")).sort("sc", descending=True).group_by("m").first()
        add = (best.join(pred.select("m").unique(), on="m", how="anti")
               .filter((pl.col("same_hn") == True) & (pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(SUFFIX)).select("s1", "m"))
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("A2 suffix adds:", add.height)
    if "HN_A" in rules:    # claimed France pairs with a changed house number and blend < 0.998
        drop = pred.join(cp.filter(pl.col("hn").is_not_null() & pl.col("hn2").is_not_null() & (pl.col("hn") != pl.col("hn2")) & (pl.col("bp") < 0.998)),
                         on=["s1", "m"]).select("s1", "m")
        pred = pred.join(drop, on=["s1", "m"], how="anti"); log("HN_A changed-number drops:", drop.height)
    if "D" in rules:       # claimed France category swaps (one word dropped, a category word added, number kept)
        drop = pred.join(cp.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(CATEGORY) & (pl.col("same_hn") == True)),
                         on=["s1", "m"]).select("s1", "m")
        pred = pred.join(drop, on=["s1", "m"], how="anti"); log("D category-swap drops:", drop.height)
s1_ids = tec["s1"].to_list()
write_id_lists(f"{a.out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
write_id_lists(f"{a.out}/candidate_pairs.tsv", s1_ids, te.select("s1", "m"), "candidate_entity_ids")
log("written:", pred.height, "matches;", te.height, "candidate pairs (", round(te.height / len(s1_ids), 2), "per S1 )")
log(pred.join(tec, on="s1").group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("S1")).sort("country"))
