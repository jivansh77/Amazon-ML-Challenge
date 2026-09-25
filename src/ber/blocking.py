"""Candidate generation (blocking).

Each route produces, for every S1 record, a ranked top-K list of S2/S3 records from the
same country label. Routes are unioned; the route scores/ranks are kept as features.

Routes
------
tok   : word-level TF-IDF cosine on (name core tokens + address tokens)
chr   : char 3-gram TF-IDF cosine on the name core (+ address words), catches typos and
        transliteration variants that break whole-word matches
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _split(s):
    return s.split()


def _topk_by_country(q_text, q_country, c_text, c_country, vec_factory, k, n_threads, chunk=20000):
    """Return polars frame (qi, ci, score, rank) of top-k matches restricted to equal country."""
    out = []
    q_country = np.asarray(q_country)
    c_country = np.asarray(c_country)
    for ctry in np.unique(q_country):
        qi = np.where(q_country == ctry)[0]
        ci = np.where(c_country == ctry)[0]
        if len(ci) == 0:
            continue
        vec = vec_factory()
        vec.fit([c_text[i] for i in ci] + [q_text[i] for i in qi])
        C = vec.transform([c_text[i] for i in ci]).astype(np.float32).T.tocsr()
        print(f"  block {ctry}: {len(qi)} queries x {len(ci)} candidates, vocab {len(vec.vocabulary_)}", flush=True)
        for s in range(0, len(qi), chunk):
            sub = qi[s:s + chunk]
            Q = vec.transform([q_text[i] for i in sub]).astype(np.float32).tocsr()
            M = sp_matmul_topn(Q, C, top_n=k, threshold=0.01, n_threads=n_threads, sort=True)
            M = M.tocoo()
            if M.nnz == 0:
                continue
            df = pl.DataFrame({"qi": sub[M.row].astype(np.int32), "ci": ci[M.col].astype(np.int32),
                               "score": M.data.astype(np.float32)})
            out.append(df)
    if not out:
        return pl.DataFrame(schema={"qi": pl.Int32, "ci": pl.Int32, "score": pl.Float32, "rank": pl.UInt32})
    df = pl.concat(out)
    return df.with_columns(pl.col("score").rank("ordinal", descending=True).over("qi").alias("rank"))


def _bigrams(col):
    """Address bigram tokens ("sector_16", "2_53"): rare even when each token is common."""
    return pl.col(col).str.split(" ").list.eval(
        pl.element() + "_" + pl.element().shift(-1)).list.drop_nulls().list.join(" ")


def tok_docs(df, side, extra=True):
    """Token documents for the TF-IDF route: name core (+ DBA name for S2/S3) + address,
    plus (extra=True) address bigrams and a compact name token ("swapmarbles")."""
    parts = [pl.col("name_core")]
    if side == "c":
        parts.append(pl.col("name_alt"))
    parts.append(pl.col("addr_clean"))
    if extra:
        parts += [_bigrams("addr_clean"), pl.lit("cmp_") + pl.col("name_core").str.replace_all(" ", "")]
    return df.select(pl.concat_str(parts, separator=" ").alias("d"))["d"].to_list()


def route_tok(s1, s23, k, n_threads, max_df=0.05, extra=False):
    q = tok_docs(s1, "q", extra)
    c = tok_docs(s23, "c", extra)
    fac = lambda: TfidfVectorizer(analyzer=_split, sublinear_tf=True, dtype=np.float32, min_df=1, max_df=max_df)
    return _topk_by_country(q, s1["country"].to_list(), c, s23["country"].to_list(), fac, k, n_threads)


def route_chr(s1, s23, k, n_threads):
    q = (s1["name_core"]).to_list()
    c = (s23["name_core"]).to_list()
    fac = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                  dtype=np.float32, min_df=2, max_df=0.02)
    return _topk_by_country(q, s1["country"].to_list(), c, s23["country"].to_list(), fac, k, n_threads)


def generate_candidates(s1, s23, k_tok=30, k_chr=15, k_key=30, n_threads=4, routes=("tok",),
                        tok_max_df=0.05, key_cap=600, tok_extra=False):
    """Union of routes -> frame (qi, ci, <route>_score, <route>_rank, ...) with nulls for misses."""
    import time
    parts = []
    if "tok" in routes:
        t = time.time()
        parts.append(route_tok(s1, s23, k_tok, n_threads, max_df=tok_max_df, extra=tok_extra)
                     .rename({"score": "tok_score", "rank": "tok_rank"}))
        print(f"  tok route {parts[-1].height} pairs in {time.time() - t:.0f}s", flush=True)
    if "chr" in routes:
        parts.append(route_chr(s1, s23, k_chr, n_threads).rename({"score": "chr_score", "rank": "chr_rank"}))
    if "key" in routes:
        t = time.time()
        parts.append(route_key(s1, s23, k_key, cap_frac=0.0, min_cap=key_cap))
        print(f"  key route {parts[-1].height} pairs in {time.time() - t:.0f}s", flush=True)
    cand = parts[0]
    for p in parts[1:]:
        cand = cand.join(p, on=["qi", "ci"], how="full", coalesce=True)
    return cand


def _record_keys(df, text_expr, addr_col):
    """Explode records into blocking keys: unigram tokens of (name + address) plus address
    bigrams ("327 cpl", "onkar nagar"), which stay rare even when each token is common."""
    base = df.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("idx"), "country",
                     text_expr.str.split(" ").alias("t"), pl.col(addr_col).str.split(" ").alias("a"))
    uni = base.select("idx", "country", "t").explode("t").rename({"t": "key"})
    bi = (base.select("idx", "country", "a").explode("a")
          .with_columns((pl.col("a") + "_" + pl.col("a").shift(-1).over("idx")).alias("key"))
          .select("idx", "country", "key"))
    return pl.concat([uni, bi]).filter(pl.col("key").is_not_null() & (pl.col("key").str.len_chars() > 1)).unique()


def route_key(s1, s23, k=30, max_keys=8, cap_frac=5e-4, min_cap=50, chunk=100_000):
    """Rare-key join: each record keeps its `max_keys` rarest keys whose document frequency
    within the country is <= cap; pairs sharing keys are scored by the summed idf."""
    k1 = _record_keys(s1, pl.col("name_core") + " " + pl.col("addr_clean"), "addr_clean")
    k2 = _record_keys(s23, pl.col("name_core") + " " + pl.col("name_alt") + " " + pl.col("addr_clean"), "addr_clean")
    n_c = pl.concat([s1.select("country"), s23.select("country")]).group_by("country").len().rename({"len": "n"})
    df = (pl.concat([k1, k2]).group_by(["country", "key"]).len().rename({"len": "df"})
          .join(n_c, on="country")
          .with_columns((pl.col("n") / pl.col("df")).log().cast(pl.Float32).alias("idf"),
                        pl.max_horizontal(pl.col("n") * cap_frac, pl.lit(min_cap)).alias("cap"))
          .filter(pl.col("df") <= pl.col("cap")).select("country", "key", "df", "idf"))

    def rarest(kk):
        return (kk.join(df, on=["country", "key"])
                .filter(pl.col("df").rank("ordinal").over("idx") <= max_keys)
                .select("idx", "country", "key", "idf"))

    k1, k2 = rarest(k1), rarest(k2)
    out = []
    for s in range(0, s1.height, chunk):
        q = k1.filter((pl.col("idx") >= s) & (pl.col("idx") < s + chunk))
        p = (q.join(k2.rename({"idx": "ci"}), on=["country", "key"], suffix="_c")
             .group_by(["idx", "ci"]).agg(pl.col("idf").sum().alias("key_score"), pl.len().alias("key_n"))
             .filter(pl.col("key_score").rank("ordinal", descending=True).over("idx") <= k))
        out.append(p)
    res = pl.concat(out).rename({"idx": "qi"}).with_columns(pl.col("qi").cast(pl.Int32), pl.col("ci").cast(pl.Int32))
    return res.with_columns(pl.col("key_score").rank("ordinal", descending=True).over("qi")
                            .cast(pl.Float32).alias("key_rank"), pl.col("key_n").cast(pl.Float32))
