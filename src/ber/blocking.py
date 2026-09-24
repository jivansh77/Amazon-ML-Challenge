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


def route_tok(s1, s23, k, n_threads, max_df=0.05):
    q = (s1["name_core"] + " " + s1["addr_clean"]).to_list()
    c = (s23["name_core"] + " " + s23["name_alt"] + " " + s23["addr_clean"]).to_list()
    fac = lambda: TfidfVectorizer(analyzer=_split, sublinear_tf=True, dtype=np.float32, min_df=1, max_df=max_df)
    return _topk_by_country(q, s1["country"].to_list(), c, s23["country"].to_list(), fac, k, n_threads)


def route_chr(s1, s23, k, n_threads):
    q = (s1["name_core"]).to_list()
    c = (s23["name_core"]).to_list()
    fac = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                  dtype=np.float32, min_df=2, max_df=0.02)
    return _topk_by_country(q, s1["country"].to_list(), c, s23["country"].to_list(), fac, k, n_threads)


def generate_candidates(s1, s23, k_tok=30, k_chr=15, n_threads=4, routes=("tok", "chr")):
    """Union of routes -> frame (qi, ci, <route>_score, <route>_rank) with nulls for misses."""
    parts = []
    if "tok" in routes:
        parts.append(route_tok(s1, s23, k_tok, n_threads).rename({"score": "tok_score", "rank": "tok_rank"}))
    if "chr" in routes:
        parts.append(route_chr(s1, s23, k_chr, n_threads).rename({"score": "chr_score", "rank": "chr_rank"}))
    cand = parts[0]
    for p in parts[1:]:
        cand = cand.join(p, on=["qi", "ci"], how="full", coalesce=True)
    return cand
