"""Final test decoding: average of pipeline runs + cross-encoder ensemble, per-country thresholds, then the
label-free France generator rules (suffix adds, changed-house-number drops, category-swap drops).

  python scripts/final_build.py --data DS --runs RUN1,RUN2 --ce CE1:W1,CE2:W2 --ce_fr FRCE1,FRCE2 --out OUT

--runs   dirs with test_scores.parquet and fr_test_scores.parquet (France scored with the calibrated French words);
         a pair's p is the mean over the runs that scored it.
--ce     cross-encoder test files (s1, m, ce) with weights: logit average over the files that scored the pair.
--ce_fr  French cross-encoder files; they replace the --ce ensemble for their pairs (France).
Blend sigmoid(w logit(p) + (1 - w) logit(ce)); pairs without a cross-encoder score keep p.

France rules (label-free, see EXPERIMENTS.md): A2 suffix adds, HN_A changed-number drops, D category-swap drops, E empty-address
adds; v2: A2F (France/Services/Cie suffix adds by position), DP (in-place category-swap drops over the whole French category
vocabulary, replaces D), E2 (E with the v2 suffixes), OOC (suffix-operation records outside the candidate lists), HNK
(changed-number records with the qualifier kept and no small upward shift).
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
ap.add_argument("--rules", default="A2,HN_A,D", help="France rules to apply (empty = none): A2, A2F, HN_A, D, DP, E, E2, OOC, HNK")
ap.add_argument("--e_margin", type=float, default=15.0)
ap.add_argument("--first_min", default="", help="S1 with no match after decoding: add its best exclusive candidate when p >= this "
                "(per country, e.g. US=0.6,India=0.6,France=0.6); for an empty S1 the F0.5 break-even precision is ~0.5, not ~0.73")
ap.add_argument("--collapse_legal", action="store_true", help="France rules: collapse dotted legal forms (S.A.R.L.) before comparing names")
ap.add_argument("--extra_pairs", default="", help="parquet files of (s1, m) pairs from an extra candidate route (e.g. the native-script route, "
                "scripts/native_route.py); a pair is added when its record is still unclaimed after decoding and rules, and it is added to "
                "the candidate file as well")
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
te = ts.with_columns(pl.col("p").alias("p_model")).join(ens, on=["s1", "m"], how="left").with_columns(
    p=pl.when(pl.col("lce").is_null()).then(pl.col("p")).otherwise(1 / (1 + (-(a.w * L("p") + (1 - a.w) * pl.col("lce"))).exp()))).drop("lce")
thr = {k: float(v) for k, v in (x.split("=") for x in a.thr.split(","))}
pred = decode_by_country(te.select("s1", "m", "p"), thr, tec)
log("decoded:", pred.height, "pairs; thresholds", thr)
if a.first_min:
    fm = {k: float(v) for k, v in (x.split("=") for x in a.first_min.split(","))}
    ex = te.select("s1", "m", "p").filter(pl.col("p") == pl.col("p").max().over("m")).join(pred.select("s1").unique(), on="s1", how="anti").join(tec, on="s1")
    add = ex.sort("p", descending=True).group_by("s1").first()
    add = add.filter(pl.col("p") >= pl.col("country").replace_strict(fm, default=1.1, return_dtype=pl.Float64)).select("s1", "m")
    pred = pl.concat([pred, add]); log("first-match adds for empty S1:", add.height, fm)
pred.write_parquet(f"{a.out}/pred_before_rules.parquet")

# ---- France generator rules (label-free; see EXPERIMENTS.md "France generator rules") ----
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc", "pa", "plc",
        "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des", "du", "la", "le", "les", "d", "l", "s", "a"]
SUFFIX = ["fils", "groupe", "developpement", "associes"]
CATEGORY = ['agricole', 'amicale', 'amis', 'anciens', 'atelier', 'ateliers', 'auto', 'automobile', 'cafe', 'club', 'collectif', 'college',
            'comite', 'conseil', 'culture', 'culturelle', 'danse', 'ecole', 'ehpad', 'elementaire', 'energie', 'fetes', 'foyer', 'gestion',
            'groupement', 'institut', 'jeunes', 'loisirs', 'lycee', 'maison', 'maternelle', 'medical', 'medico', 'musique', 'parents',
            'patrimoine', 'pharmacie', 'primaire', 'residence', 'sante', 'section', 'societe', 'soins', 'sportive', 'theatre', 'union', 'france']
# France generator, v2 (EXPERIMENTS.md "France suffix operation"): the match-noise operation drops one S1 word and appends
# (or prepends) a suffix from a French list - the same operation as US/India's Center/Services/Service/Partners - while the
# category-swap decoy replaces a word IN PLACE. "France", "Services" and "Cie" are on the French suffix list too.
NEW_SUF = ["france", "services", "cie"]
SUF_ALL = SUFFIX + NEW_SUF
LEGAL_FR = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei"]
OK_POS = ["appended", "front", "noleg_last"]
FR_CITIES = sorted(["la baule escoublac", "la teste de buch", "lege cap ferret", "hellemmes lille", "saint herblain", "saint nazaire",
                    "dunkerque", "tourcoing", "bordeaux", "merignac", "le clion", "roubaix", "nantes", "calais", "pessac", "pornic",
                    "lille", "lomme"], key=len, reverse=True)
STREET_TYPE = ["rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "place", "pl", "allee", "all", "ale", "impasse", "imp", "chemin",
               "ch", "che", "route", "rte", "cours", "crs", "square", "sq", "quai", "qu", "passage", "pass", "cite", "residence", "res", "voie",
               "sentier", "esplanade", "parvis", "promenade", "rond", "point", "hameau", "lieu", "dit", "no", "n", "numero", "de", "du", "des",
               "la", "le", "les", "d", "l", "et", "bis", "ter", "b", "t", "a", "c", "f", "st", "ste", "saint", "sainte"]


def legal_tokens(col):
    """Name tokens with dotted French legal forms (S.A.R.L., S.A.S.U., ...) collapsed to one token."""
    return (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
            .str.replace_all(r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\b", "sarl").str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\s*\.?\s*u\b", "sasu")
            .str.replace_all(r"\bs\s*\.\s*a\s*\.\s*s\b", "sas").str.replace_all(r"\be\s*\.\s*u\s*\.\s*r\s*\.\s*l\b", "eurl")
            .str.replace_all(r"\bs\s*\.\s*a\b\.?", "sa").str.replace_all(r"\bs\s*\.\s*n\s*\.\s*c\b\.?", "snc")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" "))


def with_position(df, raw_col):
    """Where the added word `aw` sits in the record name: front / appended (after the legal form) / middle (before it) /
    noleg_last / noleg_mid. The suffix operation appends or prepends; a category swap stays where the dropped word was."""
    df = df.with_columns(legal_tokens(raw_col).alias("_t"))
    df = df.with_columns(pl.col("_t").list.eval(pl.int_range(pl.len()).filter(pl.element().is_in(LEGAL_FR))).list.first().alias("_il"),
                         pl.col("_t").list.len().alias("_n"),
                         pl.struct(["_t", "aw"]).map_elements(lambda s: s["_t"].index(s["aw"]) if s["aw"] in (s["_t"] or []) else None,
                                                              return_dtype=pl.Int64).alias("_ia"))
    pos = (pl.when(pl.col("_ia").is_null()).then(pl.lit("na")).when(pl.col("_ia") == 0).then(pl.lit("front"))
           .when(pl.col("_il").is_not_null() & (pl.col("_ia") > pl.col("_il"))).then(pl.lit("appended"))
           .when(pl.col("_il").is_not_null()).then(pl.lit("middle"))
           .when(pl.col("_ia") == pl.col("_n") - 1).then(pl.lit("noleg_last")).otherwise(pl.lit("noleg_mid")))
    return df.with_columns(pos.alias("pos")).drop("_t", "_il", "_n", "_ia")


def is_abbrev(aw, dw):
    """aw is an abbreviation / typo / expansion of dw (cie~compagnie, frs~freres, cbu~club, c1ub~club)."""
    if aw is None or dw is None:
        return False
    it = iter(dw)
    if all(c in it for c in aw):
        return True
    it = iter(aw)
    if all(c in it for c in dw):
        return True
    return (len(aw) <= 4 and aw[:1] == dw[:1] and set(aw) <= set(dw)) or fuzz.ratio(aw, dw) >= 70


def street_key(col):
    """Sorted street-name tokens of the address component that holds the first number (no number, no street type)."""
    comp = pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("")
    s = comp.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    return s.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STREET_TYPE) & ~pl.element().str.contains(r"\d")
                                                          & (pl.element().str.len_chars() >= 3))).list.sort().list.join(" ")


def legal_set(col):
    return legal_tokens(col).list.eval(pl.element().filter(pl.element().is_in(LEGAL_FR))).list.unique().list.sort().list.join(",")


def city(col):
    n = (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
         .str.replace_all(r"[^a-z0-9]+", " ").str.replace_all(r"\bst\b", "saint").str.strip_chars())
    return n.map_elements(lambda s: next((c for c in FR_CITIES if f" {c} " in f" {s} "), None), return_dtype=pl.Utf8)


rules = [r for r in a.rules.split(",") if r]
if rules:
    def norm(col):
        return (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
                .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
    def prep(df):
        if a.collapse_legal:   # S.A.R.L. / E.U.R.L. / S.A.S. ... -> one token, so they do not count as added or dropped words
            df = df.with_columns(legal_tokens("business_name").list.join(" ").alias("_nc"))
            df = df.with_columns(norm("_nc").alias("n"), norm("business_address").alias("a")).drop("_nc")
        else:
            df = df.with_columns(norm("business_name").alias("n"), norm("business_address").alias("a"))
        return df.with_columns(
            pl.col("n").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique().alias("core"),
            pl.col("a").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"))
    t1 = prep(read_source(os.path.join(dd, "test", "test_source1.tsv")))
    t23 = prep(pl.concat([read_source(os.path.join(dd, "test", f"test_source{k}.tsv")) for k in (2, 3)]))
    s1c = t1.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("n").alias("n2"), pl.col("a").alias("a2"),
                                                            pl.col("core").alias("core2"), pl.col("hn").alias("hn2"))
    cp = te.select("s1", "m", pl.col("p").alias("bp")).join(s1c, on="s1").join(
        t23.select(pl.col("entity_id").alias("m"), "n", "a", "core", "hn", pl.col("business_name").alias("raw")), on="m")
    ra = process.cpdist(cp["a"].to_list(), cp["a2"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)
    rn = process.cpdist(cp["n"].to_list(), cp["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    cp = cp.with_columns(pl.Series("ra", ra, pl.Float64), pl.Series("rn", rn, pl.Float64)).with_columns(
        pl.col("core2").list.set_difference(pl.col("core")).list.len().alias("nd"),
        pl.col("core").list.set_difference(pl.col("core2")).alias("added"),
        pl.when(pl.col("hn").is_null() | pl.col("hn2").is_null()).then(pl.lit(None, pl.Boolean)).otherwise(pl.col("hn") == pl.col("hn2")).alias("same_hn"),
    ).with_columns(pl.col("added").list.len().alias("na"), pl.col("added").list.first().alias("aw"),
                   pl.col("core2").list.set_difference(pl.col("core")).list.first().alias("dw")).drop("n", "a", "n2", "a2", "core", "core2", "added")
    if "A2" in rules or "A2P" in rules:  # French noise suffix added, number kept, record unclaimed: its best S1 (by name + address)
        best = cp.with_columns((pl.col("rn") + pl.col("ra")).alias("sc")).sort("sc", descending=True).group_by("m").first()
        add = (best.join(pred.select("m").unique(), on="m", how="anti")
               .filter((pl.col("same_hn") == True) & (pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(SUFFIX)))
        if "A2P" in rules:  # not in the middle of the name: an in-place Groupe is a category swap (Groupe is also a category word)
            add = with_position(add, "raw").filter(pl.col("pos").is_in(OK_POS))
        add = add.select("s1", "m")
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("A2 suffix adds:", add.height)
    if "A2F" in rules:     # the same suffix operation with France / Services / Cie, which only the position separates from a
                           # category swap: the suffix is appended after the legal form (or at the end / front of the name)
        best = cp.with_columns((pl.col("rn") + pl.col("ra")).alias("sc")).sort("sc", descending=True).group_by("m").first()
        add = (best.join(pred.select("m").unique(), on="m", how="anti")
               .filter((pl.col("same_hn") == True) & (pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(NEW_SUF)))
        add = with_position(add, "raw").filter(pl.col("pos").is_in(OK_POS)).select("s1", "m")
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("A2F France/Services/Cie suffix adds:", add.height)
    if "A2N" in rules:     # the suffix operation on a record whose house number was dropped: the S1's street words all in the
                           # record's address, same city (US/India labels: 95-97% true matches for this class)
        best = cp.with_columns((pl.col("rn") + pl.col("ra")).alias("sc")).sort("sc", descending=True).group_by("m").first()
        c = (best.join(pred.select("m").unique(), on="m", how="anti")
             .filter(pl.col("hn").is_null() & pl.col("hn2").is_not_null() & (pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(SUF_ALL)))
        c = with_position(c, "raw").filter(pl.col("pos").is_in(OK_POS)).join(
            t1.select(pl.col("entity_id").alias("s1"), pl.col("business_address").alias("A1")), on="s1").join(
            t23.select(pl.col("entity_id").alias("m"), pl.col("business_address").alias("A2"), pl.col("a").alias("a2n")), on="m")
        c = c.with_columns(street_key("A1").alias("sk1"), city("A1").alias("c1"), city("A2").alias("c2"))
        c = c.filter((pl.col("sk1") != "") & (pl.col("sk1").str.split(" ").list.set_difference(pl.col("a2n").str.split(" ")).list.len() == 0)
                     & ~(pl.col("c1").is_not_null() & pl.col("c2").is_not_null() & (pl.col("c1") != pl.col("c2"))))
        add = c.select("s1", "m")
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("A2N suffix adds on records without a house number:", add.height)
    if "HN_A" in rules:    # claimed France pairs with a changed house number and blend < 0.998
        drop = pred.join(cp.filter(pl.col("hn").is_not_null() & pl.col("hn2").is_not_null() & (pl.col("hn") != pl.col("hn2")) & (pl.col("bp") < 0.998)),
                         on=["s1", "m"]).select("s1", "m")
        pred = pred.join(drop, on=["s1", "m"], how="anti"); log("HN_A changed-number drops:", drop.height)
    if "D" in rules:       # claimed France category swaps (one word dropped, a category word added, number kept)
        drop = pred.join(cp.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(CATEGORY) & (pl.col("same_hn") == True)),
                         on=["s1", "m"]).select("s1", "m")
        pred = pred.join(drop, on=["s1", "m"], how="anti"); log("D category-swap drops:", drop.height)
    if "DP" in rules:      # claimed in-place category swaps, any category word of the French vocabulary (incl. Centre, Compagnie,
                           # Service, Federation, which the model trusts because Center/Service are match words in the US); not
                           # the suffix words, not abbreviations or typos of the dropped word (cie~compagnie, frs~freres, c1ub)
        sw = cp.filter((pl.col("nd") == 1) & (pl.col("na") == 1))
        catv = [w for w in sw.group_by("dw").len().filter(pl.col("len") >= 40)["dw"].to_list() if w not in SUF_ALL]
        drop = pred.join(sw.filter((pl.col("same_hn") == True) & ~pl.col("aw").is_in(SUF_ALL) & pl.col("aw").is_in(catv)), on=["s1", "m"])
        drop = drop.filter(~pl.struct(["aw", "dw"]).map_elements(lambda s: is_abbrev(s["aw"], s["dw"]), return_dtype=pl.Boolean)).select("s1", "m")
        pred = pred.join(drop, on=["s1", "m"], how="anti"); log("DP in-place category-swap drops:", drop.height, "(vocabulary", len(catv), "words)")
    if "E" in rules or "E2" in rules:  # France records WITHOUT an address, unclaimed, whose name clearly points to one S1 (margin over the
                           # 2nd candidate): same core words and the stage-2 model alone above the France threshold, or a
                           # French noise suffix added (A2 needs a house number, so it never covers these)
        emp = t23.filter(pl.col("a") == "").select(pl.col("entity_id").alias("m"))
        e = te.select("s1", "m", "p_model").join(emp, on="m").join(s1c.select("s1", "n2", "core2"), on="s1").join(
            t23.select(pl.col("entity_id").alias("m"), "n", "core"), on="m")
        e = e.with_columns(pl.Series("sim", process.cpdist(e["n"].to_list(), e["n2"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1), pl.Float64))
        e = e.sort("sim", descending=True).group_by("m", maintain_order=True).agg(
            pl.all().first(), pl.col("sim").get(1, null_on_oob=True).fill_null(0).alias("sim2"))
        e = e.join(pred.select("m").unique(), on="m", how="anti").filter(pl.col("sim") - pl.col("sim2") >= a.e_margin)
        e = e.with_columns(pl.col("core2").list.set_difference(pl.col("core")).list.len().alias("nd"),
                           pl.col("core").list.set_difference(pl.col("core2")).alias("added")).with_columns(
                           pl.col("added").list.len().alias("na"), pl.col("added").list.first().alias("aw"))
        same = e.filter((pl.col("nd") == 0) & (pl.col("na") == 0) & (pl.col("p_model") >= thr["France"]))
        suf = e.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(SUFFIX))
        if "E2" in rules:  # + France / Services / Cie in the suffix position
            suf2 = e.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(NEW_SUF)).join(
                t23.select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("raw")), on="m")
            suf = pl.concat([suf.select("s1", "m"), with_position(suf2, "raw").filter(pl.col("pos").is_in(OK_POS)).select("s1", "m")])
        add = pl.concat([same.select("s1", "m"), suf.select("s1", "m")])
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("E empty-address adds:", same.height, "same core,", suf.height, "suffix")
    if "OOC" in rules:     # suffix-operation records OUTSIDE every candidate list (the stage-1 filter drops them: the model reads
                           # the French suffixes as decoy words): same number + street + city as exactly one France S1, the legal
                           # form not swapped and every address number equal; they are added to the candidate file as well
        dig = lambda c: pl.col(c).fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.join(" ")
        s1r = t1.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("core").alias("core2"), "hn",
                                                               street_key("business_address").alias("sk"), pl.col("business_name").alias("n1"),
                                                               pl.col("business_address").alias("a1"))
        rec = (t23.filter(pl.col("country") == "France").join(te.select("m").unique(), left_on="entity_id", right_on="m", how="anti")
               .select(pl.col("entity_id").alias("m"), "core", "hn", street_key("business_address").alias("sk"),
                       pl.col("business_name").alias("raw"), pl.col("business_address").alias("a2")))
        j = s1r.filter(pl.col("hn").is_not_null() & (pl.col("sk") != "")).join(rec.filter(pl.col("hn").is_not_null() & (pl.col("sk") != "")), on=["hn", "sk"])
        j = j.with_columns(pl.col("core2").list.set_difference(pl.col("core")).list.len().alias("nd"),
                           pl.col("core").list.set_difference(pl.col("core2")).alias("added")).with_columns(
                           pl.col("added").list.len().alias("na"), pl.col("added").list.first().alias("aw"))
        j = j.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(SUF_ALL))
        j = with_position(j, "raw").filter(pl.col("pos").is_in(OK_POS))
        j = j.with_columns(legal_set("n1").alias("l1"), legal_set("raw").alias("l2"), dig("a1").alias("d1"), dig("a2").alias("d2"),
                           city("a1").alias("c1"), city("a2").alias("c2"))
        j = j.filter(~((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))) & (pl.col("d1") == pl.col("d2"))
                     & ~(pl.col("c1").is_not_null() & pl.col("c2").is_not_null() & (pl.col("c1") != pl.col("c2"))))
        extra = j.filter(pl.len().over("m") == 1).select("s1", "m")
        pred = pl.concat([pred, extra]).unique(maintain_order=True); log("OOC out-of-candidate suffix adds:", extra.height)
    if "HNK" in rules:     # unclaimed France records with the S1's core name on the same street and city but another first house
                           # number, whose legal form / 'France' qualifier is NOT added or swapped and whose number is not shifted
                           # up by 1-100: in US/India labels that class is 97% true matches (a perturbed number); the decoy operation
                           # adds or swaps a qualifier and shifts the number up by a small step
        cnt_fr = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.count_matches(r"\bfrance\b")
        h = cp.filter((pl.col("nd") == 0) & (pl.col("na") == 0) & (pl.col("same_hn") == False)).join(pred.select("m").unique(), on="m", how="anti")
        exact = cp.filter((pl.col("nd") == 0) & (pl.col("na") == 0) & (pl.col("same_hn") == True)).select("m").unique()
        h = h.join(exact, on="m", how="anti").join(
            t1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1").join(
            t23.select(pl.col("entity_id").alias("m"), pl.col("business_address").alias("a2")), on="m")
        h = h.with_columns(street_key("a1").alias("sk1"), street_key("a2").alias("sk2"), legal_set("n1").alias("l1"), legal_set("raw").alias("l2"),
                           (cnt_fr("raw") > cnt_fr("n1")).alias("fr_added"), city("a1").alias("c1"), city("a2").alias("c2"),
                           (pl.col("a2").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False) - pl.col("a1").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)).alias("delta"))
        legchg = (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))          # legal form added or swapped
        h = h.filter((pl.col("sk1") == pl.col("sk2")) & (pl.col("sk1") != "") & ~legchg & ~pl.col("fr_added")
                     & ~(pl.col("c1").is_not_null() & pl.col("c2").is_not_null() & (pl.col("c1") != pl.col("c2")))
                     & ((pl.col("delta") < 0) | (pl.col("delta") > 100)))
        add = h.sort("bp", descending=True).group_by("m").first().select("s1", "m")
        pred = pl.concat([pred, add]).unique(maintain_order=True); log("HNK kept-qualifier changed-number adds:", add.height)
xtra = None
if a.extra_pairs:
    # one S1 per record: when several route files propose the same record, the earlier file wins (files are in priority order)
    xp = pl.concat([pl.read_parquet(f, columns=["s1", "m"]).with_columns(pl.lit(i).alias("_pri")) for i, f in enumerate(a.extra_pairs.split(","))])
    xp = xp.sort("_pri", maintain_order=True).unique("m", keep="first", maintain_order=True).drop("_pri")
    xtra = xp.join(pred.select("m").unique(), on="m", how="anti").join(tec.select("s1"), on="s1")
    pred = pl.concat([pred, xtra]).unique(maintain_order=True); log("extra-route adds (unclaimed records):", xtra.height, "of", xp.height)
s1_ids = tec["s1"].to_list()
cand = te.select("s1", "m")
if rules and "OOC" in rules:
    cand = pl.concat([cand, extra])
if xtra is not None:
    cand = pl.concat([cand, xtra]).unique(maintain_order=True)
write_id_lists(f"{a.out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
write_id_lists(f"{a.out}/candidate_pairs.tsv", s1_ids, cand, "candidate_entity_ids")
pred.write_parquet(f"{a.out}/pred.parquet")
log("written:", pred.height, "matches;", cand.height, "candidate pairs (", round(cand.height / len(s1_ids), 2), "per S1 )")
log(pred.join(tec, on="s1").group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("S1")).sort("country"))
