"""Phonetic route, step 1 of 3: candidates (US/India recall routes, STEPS 0-3).

  python scripts/phonetic_route_candidates.py --data <D> --work work --out work/phonetic_cand

--work is the pipeline work dir of steps 1-3 (train/test *_s1n, *_s23n, *_cand, *_dense, *_dense_rev parquet files and
val_scores.parquet). The script compares five label-free key routes (R1, R1b, R2, R3, R4 below) on the training labels;
only R3, the phonetic route, is used downstream (scripts/phonetic_route_score.py). The labels stay in the output files.

STEP 0  current candidate union = tok rank <= 20 | dense rank <= 20 | reverse dense rank <= 2 (the caps of step 3)
        | reverse-name route top-10 (char 3-gram TF-IDF on the name, S2/S3 records WITHOUT an address;
        recomputed here with the namerev stage's logic, for the records that matter: those in missed true pairs or
        in new-route pairs).
STEP 1  profile of the true (train) pairs NOT in that union.
STEP 2  four label-free, same-country key routes, each capped at top-2 NEW pairs per S1 by IDF-weighted cosine:
          R1  house number + sorted street-core address words (exact key)      R1b house number + one rare address word
          R2  compact name (core name, no spaces) sorted-neighbourhood window 5
          R3  Double Metaphone (package `metaphone`, BSD) of the first 2 core-name tokens + house number
          R4  transliteration-folded compact name (vowel/aspirate folding), exact key
STEP 3  per route: recall of the missed true pairs vs added pairs per S1 (train); test density.
Outputs in --out: metrics.json, progress.log, route_val.parquet (validation S1 of --work), route_test.parquet.
Country is used only to keep candidates inside the same country (as every other route does); never as a feature.
"""
import argparse, json, os, sys, time, traceback

import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.io import find_dataset_dir

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dataset dir (train/ and test/)")
ap.add_argument("--work", required=True, help="pipeline work dir (normalised records, candidates, dense routes, val_scores)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DD = find_dataset_dir(args.data)
W = args.out
os.makedirs(W, exist_ok=True)
T0 = time.time()
M = {"steps": {}}
LOGF = open(f"{W}/progress.log", "a")
CAPS = {"tok": 20, "dense": 20, "rdense": 2}
RNAME_K = 10
TOPN = 2


def log(*a):
    """Timestamped log line to stdout and progress.log."""
    s = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def save_metrics():
    """Write metrics.json (called after every step so a crash keeps partial results)."""
    M["runtime_min"] = round((time.time() - T0) / 60, 1)
    json.dump(M, open(f"{W}/metrics.json", "w"), indent=1, default=str)


def pk(qi, ci):
    """Pair key (qi << 32 | ci) as an Int64 expression."""
    return (qi.cast(pl.Int64) * (1 << 32) + ci.cast(pl.Int64)).alias("k")


# ------------------------------------------------------------------ loading
def load_split(split):
    """Normalised records (row order = qi / ci) and the current candidate union (without the name route)."""
    s1 = pl.read_parquet(f"{args.work}/{split}_s1n.parquet")
    s23 = pl.read_parquet(f"{args.work}/{split}_s23n.parquet")
    log(split, "s1n", s1.shape, s1.columns); log(split, "s23n", s23.shape, s23.columns)
    parts = []
    c = pl.scan_parquet(f"{args.work}/{split}_cand.parquet")
    log(split, "cand columns", c.collect_schema().names())
    parts.append(c.filter(pl.col("tok_rank") <= CAPS["tok"]).select("qi", "ci").collect())
    d = pl.scan_parquet(f"{args.work}/{split}_dense.parquet")
    log(split, "dense columns", d.collect_schema().names())
    parts.append(d.filter(pl.col("dense_rank") <= CAPS["dense"]).select("qi", "ci").collect())
    r = pl.scan_parquet(f"{args.work}/{split}_dense_rev.parquet")
    log(split, "dense_rev columns", r.collect_schema().names())
    parts.append(r.filter(pl.col("rdense_rank") <= CAPS["rdense"]).select("qi", "ci").collect())
    for n, p in zip(["tok", "dense", "rdense"], parts):
        log(split, f"route {n} pairs within cap: {p.height}")
    u = pl.concat([p.select(pk(pl.col("qi"), pl.col("ci"))) for p in parts]).unique()
    log(split, "union (no name route):", u.height, "pairs;", round(u.height / s1.height, 2), "per S1")
    return s1, s23, u


def prep(df):
    """Label-free keys used by the routes: house number, street-core words, compact / folded / phonetic names."""
    words = pl.col("addr_words").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return df.select(
        pl.col("country"),
        pl.col("name_core").fill_null("").alias("nc"),
        pl.col("addr_clean").fill_null("").alias("ac"),
        pl.col("addr_nums").fill_null("").str.split(" ").list.first().fill_null("").alias("hn"),
        words.list.unique().alias("aw"),
        pl.col("name_core").fill_null("").str.replace_all(r"[^a-z0-9]", "").alias("cmp"))


def fold_expr(c):
    """Transliteration folding of a compact name: aa/ee/oo, aspirates, w/v, z/j, y/i, doubled letters,
    non-initial vowels (Indic spelling variants such as shri/sri, laxmi/lakshmi, agarwal/aggarwal)."""
    e = pl.col(c)
    for a, b in [("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("ou", "u"), ("ksh", "x"), ("ks", "x"),
                 ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("ph", "f"), ("jh", "j"), ("ck", "k"), ("w", "v"), ("z", "j"), ("q", "k"), ("y", "i")]:
        e = e.str.replace_all(a, b, literal=True)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        e = e.str.replace_all(ch + "+", ch)
    return (e.str.slice(0, 1) + e.str.slice(1).str.replace_all(r"[aeiou]", "")).alias("fold")


def build_keys(s1, s23):
    """Per-record key columns for S1 (side 1) and S2/S3 (side 2)."""
    a = prep(s1).with_columns(pl.lit(1, pl.Int8).alias("side"), pl.int_range(pl.len()).cast(pl.Int32).alias("i"))
    b = prep(s23).with_columns(pl.lit(2, pl.Int8).alias("side"), pl.int_range(pl.len()).cast(pl.Int32).alias("i"))
    r = pl.concat([a, b])
    # address-word document frequency per country: frequent words (city, state, "road") are not street-core
    df = (r.select("country", pl.col("aw")).explode("aw").drop_nulls("aw").group_by("country", "aw").len("dfw"))
    tot = r.group_by("country").len("n")
    df = df.join(tot, on="country").with_columns((pl.col("dfw") / pl.col("n")).alias("dfr")).drop("n")
    ex = (r.select("side", "i", "country", "aw").explode("aw").drop_nulls("aw").join(df, on=["country", "aw"]))
    core = (ex.filter(pl.col("dfr") < 0.002).group_by("side", "i")
            .agg(pl.col("aw").sort().str.join(" ").alias("street"),
                 pl.col("aw").sort_by("dfw").head(2).alias("rare")))
    r = r.join(core, on=["side", "i"], how="left").with_columns(fold_expr("cmp"))
    # Double Metaphone (primary code) of the first two core-name tokens
    import metaphone
    toks = r.select(pl.col("nc").str.split(" ").list.head(2).alias("t")).explode("t")["t"].drop_nulls().unique()
    mp = {t: metaphone.doublemetaphone(t)[0] or t for t in toks.to_list()}
    log("metaphone codes for", len(mp), "tokens")
    mpf = pl.DataFrame({"t": list(mp.keys()), "code": list(mp.values())})
    ph = (r.select("side", "i", pl.col("nc").str.split(" ").list.head(2).alias("t")).explode("t")
          .with_columns(pl.int_range(pl.len()).over(["side", "i"]).alias("o"))
          .join(mpf, on="t", how="left").sort(["side", "i", "o"])
          .group_by("side", "i", maintain_order=True).agg(pl.col("code").fill_null("").str.join(" ").alias("phon")))
    return r.join(ph, on=["side", "i"], how="left")


def key_join(r, key, c1, c2, name):
    """Exact-key join of S1 x S2/S3 within a country; keys with more than c1 S1 or c2 S2/S3 records are skipped."""
    k = r.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).select("side", "i", "country", key)
    cnt = k.group_by("country", key).agg((pl.col("side") == 1).sum().alias("n1"), (pl.col("side") == 2).sum().alias("n2"))
    ok = cnt.filter((pl.col("n1") >= 1) & (pl.col("n2") >= 1) & (pl.col("n1") <= c1) & (pl.col("n2") <= c2))
    log(name, "keys", cnt.height, "joinable", ok.height, "skipped big", cnt.filter((pl.col("n1") > c1) | (pl.col("n2") > c2)).height)
    k = k.join(ok.select("country", key), on=["country", key])
    p = (k.filter(pl.col("side") == 1).select(pl.col("i").alias("qi"), "country", key)
         .join(k.filter(pl.col("side") == 2).select(pl.col("i").alias("ci"), "country", key), on=["country", key])
         .select("qi", "ci").unique())
    log(name, "raw pairs", p.height)
    return p


def route_pairs(r):
    """Raw pairs of every route (before the new-pair filter and the top-2 cap)."""
    out = {}
    hn = pl.col("hn") != ""
    r1 = r.with_columns(pl.when(hn & pl.col("street").is_not_null()).then(pl.col("hn") + "|" + pl.col("street")).alias("k1"))
    out["R1"] = key_join(r1, "k1", 20, 80, "R1")
    r1b = (r.filter(hn & pl.col("rare").is_not_null()).select("side", "i", "country", "hn", "rare").explode("rare")
           .with_columns((pl.col("hn") + "|" + pl.col("rare")).alias("k1b")))
    out["R1b"] = key_join(r1b, "k1b", 10, 40, "R1b")
    out["R3"] = key_join(r.with_columns(pl.when(hn & (pl.col("phon") != "")).then(pl.col("phon") + "|" + pl.col("hn")).alias("k3")),
                         "k3", 10, 40, "R3")
    out["R4"] = key_join(r.with_columns(pl.when(pl.col("fold").str.len_chars() >= 4).then(pl.col("fold")).alias("k4")),
                         "k4", 10, 40, "R4")
    # R2: sorted neighbourhood on the compact name, window 5 (pairs of different sides within 4 positions)
    s = (r.filter(pl.col("cmp").str.len_chars() >= 3).select("side", "i", "country", "cmp")
         .sort(["country", "cmp"]).with_row_index("pos"))
    ps = []
    for off in range(1, 5):
        sh = s.select(pl.col("pos") - off, pl.col("side").alias("side2"), pl.col("i").alias("i2"), pl.col("country").alias("c2"))
        j = s.join(sh, on="pos").filter((pl.col("country") == pl.col("c2")) & (pl.col("side") != pl.col("side2")))
        ps.append(j.select(pl.when(pl.col("side") == 1).then(pl.col("i")).otherwise(pl.col("i2")).alias("qi"),
                           pl.when(pl.col("side") == 1).then(pl.col("i2")).otherwise(pl.col("i")).alias("ci")))
    out["R2"] = pl.concat(ps).unique()
    log("R2 raw pairs", out["R2"].height)
    return out


# ------------------------------------------------------------------ name route (recomputed)
def name_route(s1, s23, need_ci):
    """Top-RNAME_K S1 by char 3-gram TF-IDF on the name for the given empty-address S2/S3 records
    (ber.pipeline.stage_namerev; vectoriser fitted on S1 names + every empty-address record name)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    emp = s23.with_row_index("ci").filter(pl.col("addr_clean").fill_null("") == "")
    need = emp.filter(pl.col("ci").is_in(need_ci.implode()))
    log("name route: empty-address records", emp.height, "queried", need.height)
    out = []
    for ctry in need["country"].unique().to_list():
        ci_s1 = s1.with_row_index("qi").filter(pl.col("country") == ctry)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        vec.fit(ci_s1["name_clean"].fill_null("").to_list() + emp.filter(pl.col("country") == ctry)["name_clean"].fill_null("").to_list())
        C = vec.transform(ci_s1["name_clean"].fill_null("").to_list()).astype(np.float32).T.tocsr()
        q = need.filter(pl.col("country") == ctry)
        qidx = ci_s1["qi"].to_numpy()
        for s in range(0, q.height, 20000):
            sub = q.slice(s, 20000)
            Q = vec.transform(sub["name_clean"].fill_null("").to_list()).astype(np.float32).tocsr()
            Mx = sp_matmul_topn(Q, C, top_n=RNAME_K, threshold=0.01, n_threads=4, sort=True).tocoo()
            out.append(pl.DataFrame({"qi": qidx[Mx.col].astype(np.int32), "ci": sub["ci"].to_numpy()[Mx.row].astype(np.int32)}))
        log("name route", ctry, "queries", q.height)
    if not out:
        return pl.DataFrame(schema={"k": pl.Int64})
    p = pl.concat(out)
    log("name route pairs", p.height)
    return p.select(pk(pl.col("qi"), pl.col("ci")))


# ------------------------------------------------------------------ similarity for ranking
def idf_cos(s1, s23, pairs):
    """IDF-weighted cosine of (core name + address) word sets for the pairs (sublinear TF-IDF, l2)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    d1 = (s1["name_core"].fill_null("") + " " + s1["addr_clean"].fill_null("")).to_list()
    d2 = (s23["name_core"].fill_null("") + " " + s23["addr_clean"].fill_null("")).to_list()
    vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, dtype=np.float32, binary=True)
    vec.fit(d1 + d2)
    A = vec.transform(d1).tocsr(); B = vec.transform(d2).tocsr()
    qi = pairs["qi"].to_numpy(); ci = pairs["ci"].to_numpy()
    out = np.zeros(len(qi), np.float32)
    for s in range(0, len(qi), 2_000_000):
        out[s:s + 2_000_000] = np.asarray(A[qi[s:s + 2_000_000]].multiply(B[ci[s:s + 2_000_000]]).sum(1)).ravel()
    return out


def run_split(split, truth=None, val_s1=None):
    """Build union, routes, name route; returns per-route new capped pairs (qi, ci, route, cos)."""
    s1, s23, u = load_split(split)
    r = build_keys(s1, s23)
    log(split, "keys built")
    raw = route_pairs(r)
    del r
    allp = pl.concat([p.with_columns(pl.lit(n).alias("route")) for n, p in raw.items()])
    allp = allp.with_columns(pk(pl.col("qi"), pl.col("ci")))
    allp = allp.join(u, on="k", how="anti")
    missed = None
    if truth is not None:
        tq = truth.join(s1.select(pl.col("entity_id").alias("s1")).with_row_index("qi"), on="s1") \
                  .join(s23.select(pl.col("entity_id").alias("m")).with_row_index("ci"), on="m")
        tq = tq.with_columns(pl.col("qi").cast(pl.Int32), pl.col("ci").cast(pl.Int32)).with_columns(pk(pl.col("qi"), pl.col("ci")))
        M["steps"][f"{split}_truth_pairs"] = tq.height
        missed = tq.join(u, on="k", how="anti")
        log(split, "true pairs", tq.height, "missed by union (no name route)", missed.height)
    need = allp.select(pl.col("ci")).unique()["ci"]
    if missed is not None:
        need = pl.concat([need, missed["ci"]]).unique()
    rn = name_route(s1, s23, need.cast(pl.UInt32))
    u_full = pl.concat([u, rn]).unique()
    M["steps"][f"{split}_union_pairs"] = u_full.height
    M["steps"][f"{split}_union_per_s1_approx"] = round(u_full.height / s1.height, 3)
    allp = allp.join(rn, on="k", how="anti")
    if missed is not None:
        n0 = missed.height
        missed = missed.join(rn, on="k", how="anti")
        log(split, "missed after name route", missed.height, "(name route recovered", n0 - missed.height, ")")
        M["steps"][f"{split}_missed"] = missed.height
        M["steps"][f"{split}_recall_union"] = round(1 - missed.height / M["steps"][f"{split}_truth_pairs"], 5)
    allp = allp.with_columns(pl.Series("cos", idf_cos(s1, s23, allp)))
    # top-2 NEW pairs per S1 within each route
    capped = (allp.sort(["qi", "route", "cos"], descending=[False, False, True])
              .with_columns(pl.int_range(pl.len()).over(["qi", "route"]).alias("rk")).filter(pl.col("rk") < TOPN))
    ctry = s1.select("country").with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
    nS1 = s1.group_by("country").len("nS1")
    res = {}
    for rt in sorted(capped["route"].unique().to_list()) + ["ALL"]:
        c = capped if rt == "ALL" else capped.filter(pl.col("route") == rt)
        c = c.unique("k")
        add = c.join(ctry, on="qi").group_by("country").len("added").join(nS1, on="country")
        row = {r_["country"]: {"added_per_s1": round(r_["added"] / r_["nS1"], 4)} for r_ in add.iter_rows(named=True)}
        if missed is not None:
            hit = missed.join(c.select("k"), on="k").join(ctry, on="qi").group_by("country").len("hit")
            tot = missed.join(ctry, on="qi").group_by("country").len("miss")
            tp = c.join(pl.concat([missed.select("k"), tq.select("k")]).unique(), on="k").join(ctry, on="qi").group_by("country").len("tp")
            for x in tot.join(hit, on="country", how="left").join(tp, on="country", how="left").join(add, on="country", how="left").iter_rows(named=True):
                row.setdefault(x["country"], {}).update({"missed": x["miss"], "recovered": x["hit"] or 0,
                                                          "recall_of_missed": round((x["hit"] or 0) / x["miss"], 4),
                                                          "precision_added": round((x["tp"] or 0) / max(x["added"] or 1, 1), 4)})
        res[rt] = row
        log(split, rt, json.dumps(row))
    M["steps"][f"{split}_routes"] = res
    save_metrics()
    capped = capped.with_columns(pl.col("qi").cast(pl.Int32), pl.col("ci").cast(pl.Int32))
    ids = capped.select("qi", "ci", "route", "cos", "k").join(
        s1.select(pl.col("entity_id").alias("s1"), "country").with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32)), on="qi").join(
        s23.select(pl.col("entity_id").alias("m")).with_row_index("ci").with_columns(pl.col("ci").cast(pl.Int32)), on="ci")
    return s1, s23, ids, missed


def profile_missed(s1, s23, missed):
    """STEP 1: shape of the missed true pairs (empty address, same number, name / address overlap)."""
    from rapidfuzz import fuzz, process
    a = s1.select("country", "name_core", "addr_clean", "addr_nums", "addr_words").with_row_index("qi")
    b = s23.select(pl.col("name_core").alias("nc2"), pl.col("addr_clean").alias("ac2"), pl.col("addr_nums").alias("an2"),
                   pl.col("addr_words").alias("aw2")).with_row_index("ci")
    d = missed.select("qi", "ci").with_columns(pl.col("qi").cast(pl.UInt32), pl.col("ci").cast(pl.UInt32)).join(a, on="qi").join(b, on="ci")
    jac = lambda x, y: (pl.col(x).fill_null("").str.split(" ").list.set_intersection(pl.col(y).fill_null("").str.split(" ")).list.len()
                        / pl.col(x).fill_null("").str.split(" ").list.set_union(pl.col(y).fill_null("").str.split(" ")).list.len().clip(1, None))
    d = d.with_columns(
        (pl.col("ac2").fill_null("") == "").alias("b_empty_addr"),
        (pl.col("addr_clean").fill_null("") == "").alias("a_empty_addr"),
        ((pl.col("addr_nums").fill_null("").str.split(" ").list.first() == pl.col("an2").fill_null("").str.split(" ").list.first())
         & (pl.col("an2").fill_null("") != "")).alias("same_hn"),
        jac("name_core", "nc2").alias("name_jac"), jac("addr_words", "aw2").alias("addr_jac"),
        (pl.col("name_core").fill_null("").str.replace_all(r"[^a-z0-9]", "") == pl.col("nc2").fill_null("").str.replace_all(r"[^a-z0-9]", "")).alias("cmp_eq"))
    d = d.with_columns(pl.Series("name_tset", process.cpdist(d["name_core"].fill_null("").to_list(), d["nc2"].fill_null("").to_list(),
                                                            scorer=fuzz.token_set_ratio, workers=-1)))
    d = d.with_columns(pl.when(pl.col("b_empty_addr")).then(pl.lit("b_empty_addr"))
                       .when(pl.col("same_hn") & (pl.col("name_jac") == 0) & (pl.col("addr_jac") >= 0.5)).then(pl.lit("rename_same_addr"))
                       .when(pl.col("cmp_eq")).then(pl.lit("compact_name_equal"))
                       .when((pl.col("name_tset") < 60) & (pl.col("addr_jac") < 0.5)).then(pl.lit("heavy_name_and_addr_change"))
                       .when(pl.col("name_tset") < 60).then(pl.lit("heavy_name_change"))
                       .when(pl.col("addr_jac") < 0.5).then(pl.lit("addr_change"))
                       .otherwise(pl.lit("other")).alias("cls"))
    out = {}
    for c in d["country"].unique().sort().to_list():
        x = d.filter(pl.col("country") == c)
        out[c] = {"n": x.height,
                  **{k: int(x[k].sum()) for k in ["b_empty_addr", "a_empty_addr", "same_hn", "cmp_eq"]},
                  "name_jac_0": int((x["name_jac"] == 0).sum()), "name_jac_ge_0.5": int((x["name_jac"] >= 0.5).sum()),
                  "addr_jac_ge_0.5": int((x["addr_jac"] >= 0.5).sum()), "name_tset_lt_60": int((x["name_tset"] < 60).sum()),
                  "classes": dict(x.group_by("cls").len().sort("len", descending=True).iter_rows())}
        log("missed profile", c, json.dumps(out[c]))
    M["steps"]["miss_profile"] = out
    ex = d.group_by("cls").head(8).select("country", "cls", "name_core", "addr_clean", "nc2", "ac2")
    M["steps"]["miss_examples"] = ex.rows()
    save_metrics()


def main():
    try:
        from importlib.metadata import metadata
        M["metaphone_license"] = metadata("metaphone").get("License")
    except Exception as e:  # noqa: BLE001
        M["metaphone_license"] = repr(e)
    log("metaphone license:", M["metaphone_license"])
    gt = pl.read_csv(os.path.join(DD, "train", "train_ground_truth.tsv"), separator="\t", quote_char=None, infer_schema=False)
    truth = (gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
             .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
    log("truth edges", truth.height)
    va = pl.read_parquet(f"{args.work}/val_scores.parquet", columns=["s1", "m", "y"])
    val_s1 = va.select("s1").unique()
    log("val S1", val_s1.height, "pairs", va.height)
    s1, s23, ids, missed = run_split("train", truth=truth, val_s1=val_s1)
    ids = ids.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    ids.join(val_s1, on="s1").write_parquet(f"{W}/route_val.parquet")
    log("route_val rows", ids.join(val_s1, on="s1").height)
    # missed pairs of the val S1 (for the report) + stage-1 filtered share on val
    mv = missed.join(s1.select(pl.col("entity_id").alias("s1")).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32)), on="qi")
    vt = truth.join(val_s1, on="s1")
    in_scores = vt.join(va.filter(pl.col("y") == 1).select("s1", "m"), on=["s1", "m"]).height
    M["steps"]["val_truth"] = vt.height; M["steps"]["val_truth_in_stage2_scores"] = in_scores
    M["steps"]["val_missed_never_candidate"] = mv.join(val_s1, on="s1").height
    save_metrics()
    try:
        profile_missed(s1, s23, missed)
    except Exception:
        log("profile failed", traceback.format_exc())
    del s1, s23, ids, missed
    s1, s23, idt, _ = run_split("test")
    idt.write_parquet(f"{W}/route_test.parquet")
    log("route_test rows", idt.height)
    M["done"] = True
    save_metrics()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        M["error"] = traceback.format_exc(); log("ERROR", M["error"]); save_metrics()
