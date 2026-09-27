"""rt-miss2 (US/India recall routes, follow-up): R5 + R6 for one split (SPLIT below; train and test run as two kernels).

Current union = tok rank <= 20 | dense rank <= 20 | reverse dense rank <= 2 | reverse-name route rank <= 10.
  R5  reverse-name route deeper: for EVERY S2/S3 record without an address, its S1 ranks 11-25 by char 3-gram TF-IDF
      on the name (the repo's stage_namerev, k = 25); ranks <= 10 complete the current union.
  R6  compact-name equality (core name without spaces) RESTRICTED to candidates that share the house number or an
      address word of city / district level (document frequency < 10% of the country's records) with the S1.
Each route keeps the top-2 NEW pairs per S1 by IDF-weighted cosine (core name + address).
Train: recall of the missed true pairs and added pairs per S1; writes route_val.parquet (ber-dec-train val S1).
Test: added pairs per S1; writes route_test.parquet. Country only keeps candidates inside the same country.
"""
import glob, json, os, subprocess, time, traceback

import numpy as np
import polars as pl

SPLIT = "test"
W = "/kaggle/working"
T0 = time.time()
M = {"split": SPLIT, "steps": {}}
LOGF = open(f"{W}/progress.log", "a")
CAPS = {"tok": 20, "dense": 20, "rdense": 2}
RNAME_K, RNAME_DEEP, TOPN = 10, 25, 2
CITY_DFR = 0.10


def log(*a):
    """Timestamped log line to stdout and progress.log."""
    s = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def save_metrics():
    """Write metrics.json (after every step)."""
    M["runtime_min"] = round((time.time() - T0) / 60, 1)
    json.dump(M, open(f"{W}/metrics.json", "w"), indent=1, default=str)


def find(name, sub=""):
    """First path under /kaggle/input ending with `name` and containing `sub`."""
    hits = sorted(h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if sub in h)
    if not hits:
        raise FileNotFoundError(f"{name} ({sub})")
    return hits[0]


def pk(qi, ci):
    """Pair key (qi << 32 | ci) as an Int64 expression."""
    return (qi.cast(pl.Int64) * (1 << 32) + ci.cast(pl.Int64)).alias("k")


def load_split(split):
    """Normalised records (row order = qi / ci) and the union of tok / dense / reverse dense within the caps."""
    blk = f"ber-block-{split}"
    s1 = pl.read_parquet(find(f"{split}_s1n.parquet", blk))
    s23 = pl.read_parquet(find(f"{split}_s23n.parquet", blk))
    parts = [pl.scan_parquet(find(f"{split}_cand.parquet", blk)).filter(pl.col("tok_rank") <= CAPS["tok"]),
             pl.scan_parquet(find(f"{split}_dense.parquet", "ber-dense2")).filter(pl.col("dense_rank") <= CAPS["dense"]),
             pl.scan_parquet(find(f"{split}_dense_rev.parquet", "ber-dense2")).filter(pl.col("rdense_rank") <= CAPS["rdense"])]
    u = pl.concat([p.select(pk(pl.col("qi"), pl.col("ci"))).collect() for p in parts]).unique()
    log(split, "s1", s1.height, "s23", s23.height, "union (no name route)", u.height)
    return s1, s23, u


def name_route(s1, s23):
    """Top-25 S1 by char 3-gram TF-IDF on the name for every S2/S3 record without an address (stage_namerev, k=25)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    emp = s23.with_row_index("ci").filter(pl.col("addr_clean").fill_null("") == "")
    log("name route: empty-address records", emp.height)
    out = []
    for ctry in emp["country"].unique().to_list():
        c1 = s1.with_row_index("qi").filter(pl.col("country") == ctry)
        q = emp.filter(pl.col("country") == ctry)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        vec.fit(c1["name_clean"].fill_null("").to_list() + q["name_clean"].fill_null("").to_list())
        C = vec.transform(c1["name_clean"].fill_null("").to_list()).astype(np.float32).T.tocsr()
        qidx = c1["qi"].to_numpy()
        for s in range(0, q.height, 20000):
            sub = q.slice(s, 20000)
            Q = vec.transform(sub["name_clean"].fill_null("").to_list()).astype(np.float32).tocsr()
            Mx = sp_matmul_topn(Q, C, top_n=RNAME_DEEP, threshold=0.01, n_threads=4, sort=True).tocoo()
            out.append(pl.DataFrame({"qi": qidx[Mx.col].astype(np.int32), "ci": sub["ci"].to_numpy()[Mx.row].astype(np.int32),
                                     "score": Mx.data.astype(np.float32)}))
        log("name route", ctry, "queries", q.height)
    p = pl.concat(out).with_columns(pl.col("score").rank("ordinal", descending=True).over("ci").alias("rank"))
    log("name route pairs", p.height)
    return p.with_columns(pk(pl.col("qi"), pl.col("ci")))


def r6_pairs(s1, s23):
    """Compact-name equality restricted to the same house number or a shared city / district-level address word."""
    def prep(df, side):
        return df.select(pl.lit(side, pl.Int8).alias("side"), pl.int_range(pl.len()).cast(pl.Int32).alias("i"), "country",
                         pl.col("name_core").fill_null("").str.replace_all(r"[^a-z0-9]", "").alias("cmp"),
                         pl.col("addr_nums").fill_null("").str.split(" ").list.first().fill_null("").alias("hn"),
                         pl.col("addr_words").fill_null("").str.split(" ").list.unique().alias("aw"))
    r = pl.concat([prep(s1, 1), prep(s23, 2)]).filter(pl.col("cmp").str.len_chars() >= 3)
    ex = r.select("side", "i", "country", "cmp", "aw").explode("aw").filter(pl.col("aw").is_not_null() & (pl.col("aw").str.len_chars() >= 3))
    tot = r.group_by("country").len("n")
    dfr = ex.group_by("country", "aw").len("dfw").join(tot, on="country").with_columns((pl.col("dfw") / pl.col("n")).alias("dfr"))
    ex = ex.join(dfr.filter(pl.col("dfr") < CITY_DFR).select("country", "aw"), on=["country", "aw"])
    keys = pl.concat([
        r.filter(pl.col("hn") != "").select("side", "i", "country", (pl.col("cmp") + "|#" + pl.col("hn")).alias("key")),
        ex.select("side", "i", "country", (pl.col("cmp") + "|" + pl.col("aw")).alias("key"))])
    cnt = keys.group_by("country", "key").agg((pl.col("side") == 1).sum().alias("n1"), (pl.col("side") == 2).sum().alias("n2"))
    ok = cnt.filter((pl.col("n1") >= 1) & (pl.col("n2") >= 1) & (pl.col("n1") <= 20) & (pl.col("n2") <= 80))
    log("R6 keys", cnt.height, "joinable", ok.height, "skipped big", cnt.filter((pl.col("n1") > 20) | (pl.col("n2") > 80)).height)
    keys = keys.join(ok.select("country", "key"), on=["country", "key"])
    p = (keys.filter(pl.col("side") == 1).select(pl.col("i").alias("qi"), "country", "key")
         .join(keys.filter(pl.col("side") == 2).select(pl.col("i").alias("ci"), "country", "key"), on=["country", "key"])
         .select("qi", "ci").unique())
    log("R6 raw pairs", p.height)
    return p


def idf_cos(s1, s23, pairs):
    """IDF-weighted cosine of (core name + address) word sets for the pairs."""
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


def main():
    subprocess.run("pip install -q sparse_dot_topn 'rapidfuzz>=3.6' 2>&1 | tail -1", shell=True)
    s1, s23, u = load_split(SPLIT)
    rn = name_route(s1, s23)
    u = pl.concat([u, rn.filter(pl.col("rank") <= RNAME_K).select("k")]).unique()
    M["steps"]["union_pairs"] = u.height
    r5 = rn.filter(pl.col("rank") > RNAME_K).select("qi", "ci")
    r6 = r6_pairs(s1, s23)
    allp = pl.concat([r5.with_columns(pl.lit("R5").alias("route")), r6.with_columns(pl.lit("R6").alias("route"))])
    allp = allp.with_columns(pl.col("qi").cast(pl.Int32), pl.col("ci").cast(pl.Int32)).with_columns(pk(pl.col("qi"), pl.col("ci")))
    allp = allp.join(u, on="k", how="anti")
    log("new pairs before cap", dict(allp.group_by("route").len().iter_rows()))
    allp = allp.with_columns(pl.Series("cos", idf_cos(s1, s23, allp)))
    capped = (allp.sort(["qi", "route", "cos"], descending=[False, False, True])
              .with_columns(pl.int_range(pl.len()).over(["qi", "route"]).alias("rk")).filter(pl.col("rk") < TOPN))
    ctry = s1.select("country").with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
    nS1 = s1.group_by("country").len("nS1")
    ids = capped.select("qi", "ci", "route", "cos", "k").join(
        s1.select(pl.col("entity_id").alias("s1"), "country").with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32)), on="qi").join(
        s23.select(pl.col("entity_id").alias("m")).with_row_index("ci").with_columns(pl.col("ci").cast(pl.Int32)), on="ci")
    missed = tq = None
    if SPLIT == "train":
        gt = pl.read_csv(find("train_ground_truth.tsv"), separator="\t", quote_char=None, infer_schema=False)
        truth = (gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
                 .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
        tq = (truth.join(s1.select(pl.col("entity_id").alias("s1")).with_row_index("qi"), on="s1")
              .join(s23.select(pl.col("entity_id").alias("m")).with_row_index("ci"), on="m").select(pk(pl.col("qi"), pl.col("ci")), "qi"))
        tq = tq.with_columns(pl.col("qi").cast(pl.Int32))
        missed = tq.join(u, on="k", how="anti")
        M["steps"]["truth"] = tq.height; M["steps"]["missed"] = missed.height
        log("true pairs", tq.height, "missed by union", missed.height)
    res = {}
    for rt in ["R5", "R6", "ALL"]:
        c = (capped if rt == "ALL" else capped.filter(pl.col("route") == rt)).unique("k")
        add = c.join(ctry, on="qi").group_by("country").len("added").join(nS1, on="country")
        row = {x["country"]: {"added_per_s1": round(x["added"] / x["nS1"], 4)} for x in add.iter_rows(named=True)}
        if missed is not None:
            hit = missed.join(c.select("k"), on="k").join(ctry, on="qi").group_by("country").len("hit")
            tot = missed.join(ctry, on="qi").group_by("country").len("miss")
            for x in tot.join(hit, on="country", how="left").join(add, on="country", how="left").iter_rows(named=True):
                row.setdefault(x["country"], {}).update({"missed": x["miss"], "recovered": x["hit"] or 0,
                                                          "recall_of_missed": round((x["hit"] or 0) / x["miss"], 4),
                                                          "precision_added": round((x["hit"] or 0) / max(x["added"] or 1, 1), 4)})
            allm = missed.height; allh = missed.join(c.select("k"), on="k").height
            row["pooled"] = {"recall_of_missed": round(allh / allm, 4), "added_per_s1": round(c.height / s1.height, 4)}
        res[rt] = row
        log(SPLIT, rt, json.dumps(row))
    M["steps"]["routes"] = res
    save_metrics()
    if SPLIT == "train":
        va = pl.read_parquet(find("val_scores.parquet", "ber-dec-train"), columns=["s1"]).unique()
        ids = ids.join(va, on="s1")
        ids.write_parquet(f"{W}/route_val.parquet")
    else:
        ids.write_parquet(f"{W}/route_test.parquet")
    log("route rows written", ids.height)
    M["done"] = True
    save_metrics()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        M["error"] = traceback.format_exc(); log("ERROR", M["error"]); save_metrics()
