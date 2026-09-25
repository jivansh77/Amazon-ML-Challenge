"""End-to-end stages: normalise -> block -> features -> model -> decode.

Stages communicate through parquet files in a work directory so the expensive steps
(normalisation, blocking) can be cached and re-used across experiments.
"""
import json
import os
import time

import numpy as np
import polars as pl

from .blocking import generate_candidates
from .features import pair_features, context_features
from .io import read_split, read_ground_truth
from .normalize import normalize_frame, set_indic_dictionary


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage_normalize(dataset_dir, split, work, indic_dict_path, n_jobs=4):
    if indic_dict_path:
        set_indic_dictionary(json.load(open(indic_dict_path)))
    s1, s23 = read_split(dataset_dir, split)
    log(split, "loaded", s1.shape, s23.shape)
    normalize_frame(s1, n_jobs).write_parquet(f"{work}/{split}_s1n.parquet")
    normalize_frame(s23, n_jobs).write_parquet(f"{work}/{split}_s23n.parquet")
    log(split, "normalised")


def stage_block(split, work, k_tok=30, n_threads=4, routes=("tok",), tok_max_df=0.05, key_cap=600, tok_extra=False):
    s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet")
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet")
    cand = generate_candidates(s1, s23, k_tok=k_tok, k_key=k_tok, n_threads=n_threads, routes=routes,
                               tok_max_df=tok_max_df, key_cap=key_cap, tok_extra=tok_extra)
    cand.write_parquet(f"{work}/{split}_cand.parquet")
    log(split, "blocked", cand.shape, "per S1", cand.height / s1.height)


def stage_dense(split, work, k=20, dataset_dir=None, k_rev=0):
    """GPU dense retrieval per country; embeddings are not persisted (too large).
    Uses raw text, so it can read the source files directly (same row order as *_s1n/_s23n)."""
    from .dense import record_text, encode, topk_by_country
    cols = ["business_name", "business_address", "country"]
    if os.path.exists(f"{work}/{split}_s1n.parquet"):
        s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet", columns=cols)
        s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet", columns=cols)
    else:
        s1, s23 = read_split(dataset_dir, split)
        s1, s23 = s1.select(cols), s23.select(cols)
    parts, rparts = [], []
    for ctry in s1["country"].unique().to_list():
        qi = np.where(s1["country"].to_numpy() == ctry)[0]
        ci = np.where(s23["country"].to_numpy() == ctry)[0]
        ce = encode(record_text(s23[ci]))
        qe = encode(record_text(s1[qi]))
        log(split, ctry, "encoded", qe.shape, ce.shape)
        d = topk_by_country(qe, np.zeros(len(qi), np.int8), ce, np.zeros(len(ci), np.int8), k)
        parts.append(d.with_columns(pl.Series("qi", qi[d["qi"].to_numpy()]).cast(pl.Int32),
                                    pl.Series("ci", ci[d["ci"].to_numpy()]).cast(pl.Int32)))
        if k_rev:   # reverse direction: for every S2/S3 record, its top S1s
            r = topk_by_country(ce, np.zeros(len(ci), np.int8), qe, np.zeros(len(qi), np.int8), k_rev)
            rparts.append(pl.DataFrame({"qi": qi[r["ci"].to_numpy()].astype(np.int32),
                                        "ci": ci[r["qi"].to_numpy()].astype(np.int32),
                                        "rdense_score": r["dense_score"], "rdense_rank": r["dense_rank"]}))
        del qe, ce
    cand = pl.concat(parts)
    cand.write_parquet(f"{work}/{split}_dense.parquet")
    log(split, "dense candidates", cand.shape)
    if rparts:
        r = pl.concat(rparts)
        r.write_parquet(f"{work}/{split}_dense_rev.parquet")
        log(split, "reverse dense candidates", r.shape)


def add_second_best(df, col, grp, name):
    """Margin of `col` over the best *other* value in group `grp` (positive only for the top one)."""
    g = (df.group_by(grp).agg(pl.col(col).max().alias("_m1"),
                              pl.col(col).sort(descending=True).slice(1, 1).first().alias("_m2")))
    return (df.join(g, on=grp, how="left")
            .with_columns(pl.when(pl.col(col) >= pl.col("_m1"))
                          .then(pl.col(col) - pl.col("_m2").fill_null(0.0))
                          .otherwise(pl.col(col) - pl.col("_m1")).alias(name))
            .drop("_m1", "_m2"))


ROUTES = [("tok", "tok_score", "tok_rank"), ("key", "key_score", "key_rank"), ("dense", "dense_score", "dense_rank"),
          ("rdense", "rdense_score", "rdense_rank")]


def load_candidates(split, work, caps, keep_qi=None):
    """Union of all available routes, pruned by per-route rank caps, with global context
    features computed on blocking scores (cheap, and identical at train and test time)."""
    cand = pl.read_parquet(f"{work}/{split}_cand.parquet")
    for f in ["dense", "dense_rev"]:
        if os.path.exists(f"{work}/{split}_{f}.parquet"):
            cand = cand.join(pl.read_parquet(f"{work}/{split}_{f}.parquet"), on=["qi", "ci"], how="full", coalesce=True)
    keep = pl.lit(False)
    routes = [r for r in ROUTES if r[2] in cand.columns]      # only routes that produced candidates
    for r, sc, rk in routes:
        cand = cand.with_columns(pl.col(sc).cast(pl.Float32), pl.col(rk).cast(pl.Float32))
        keep = keep | (pl.col(rk) <= caps.get(r, 0))
    n0 = cand.height
    cand = cand.filter(keep)
    extra = [pl.col("key_n").cast(pl.Float32)] if "key_n" in cand.columns else []
    cand = cand.select("qi", "ci", *[c for _, sc, rk in routes for c in (sc, rk)], *extra)
    cand = cand.with_columns([pl.col(sc).fill_null(0.0) for _, sc, _ in routes] +
                             [pl.col(rk).fill_null(999.0) for _, _, rk in routes] +
                             ([pl.col("key_n").fill_null(0.0)] if extra else []))
    cand = context_features(cand, [sc for _, sc, _ in routes])
    for _, sc, _ in routes:
        cand = add_second_best(cand, sc, "ci", f"{sc}_margin_c")
        cand = add_second_best(cand, sc, "qi", f"{sc}_margin_q")
    log(split, "candidates", n0, "-> pruned", cand.height)
    if GLOBAL_SIMS:
        from .features import global_sims
        s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet", columns=["name_core", "addr_clean"])
        s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet", columns=["name_core", "addr_clean"])
        gn, ga = global_sims(s1, s23, cand)
        del s1, s23
        cand = cand.with_columns(pl.Series("g_name", gn), pl.Series("g_addr", ga))
        cand = cand.with_columns(
            pl.col("g_name").rank("ordinal", descending=True).over("ci").cast(pl.Float32).alias("g_name_rank_c"),
            (pl.col("g_name") >= 90).sum().over("ci").cast(pl.Float32).alias("g_name_twins_c"),
            (pl.col("g_name") >= 90).sum().over("qi").cast(pl.Float32).alias("g_name_twins_q"),
            pl.col("g_addr").rank("ordinal", descending=True).over("ci").cast(pl.Float32).alias("g_addr_rank_c"),
            ((pl.col("g_name") + pl.col("g_addr").clip(0, 100)) / 2).alias("g_mix"))
        cand = add_second_best(cand, "g_name", "ci", "g_name_margin_c")
        cand = add_second_best(cand, "g_mix", "ci", "g_mix_margin_c")
        cand = cand.with_columns(pl.col("g_mix").rank("ordinal", descending=True).over("ci").cast(pl.Float32)
                                 .alias("g_mix_rank_c"))
        log(split, "global similarity context added")
    if keep_qi is not None:     # context is computed on ALL pairs first, then rows are restricted
        cand = cand.filter(pl.col("qi").is_in(pl.Series(np.where(keep_qi)[0]).cast(pl.Int32).implode()))
        log(split, "restricted to", cand.height, "pairs")
    return cand


_SPACES = {}
GLOBAL_SIMS = False   # set by run.py --global_sims


def iter_pair_tables(split, work, cand, s1_filter=None, n_jobs=4, chunk_s1=250_000):
    """Yield feature tables for chunks of S1 rows (bounded memory)."""
    from .features import A_COLS, B_COLS, TokenSpace
    s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet", columns=["entity_id"] + A_COLS)
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet", columns=["entity_id", "src"] + B_COLS)
    s1_ids, s23_ids = s1["entity_id"], s23["entity_id"]
    src = s23["src"].to_numpy().astype(np.float32)
    if split not in _SPACES:     # idf/token spaces are reused across passes over the same split
        _SPACES[split] = (TokenSpace(s1["name_core"].to_list(), s23["name_core"].to_list()),
                          TokenSpace(s1["addr_clean"].to_list(), s23["addr_clean"].to_list()))
    spaces = _SPACES[split]
    rows = np.arange(s1.height) if s1_filter is None else np.where(s1_filter)[0]
    for s in range(0, len(rows), chunk_s1):
        sub = pl.Series(rows[s:s + chunk_s1]).cast(pl.Int32).implode()
        c = cand.filter(pl.col("qi").is_in(sub))
        f = pair_features(s1, s23, c, jobs=n_jobs, spaces=spaces)
        t = pl.concat([c, f], how="horizontal")
        del f
        t = t.with_columns(pl.Series("src", src[t["ci"].to_numpy()]),
                           ((pl.col("n_tset") + pl.col("a_tset")) / 2).alias("sim_mix"))
        t = t.with_columns((pl.col("sim_mix") - pl.col("sim_mix").max().over("qi")).alias("sim_mix_gap_q"),
                           pl.col("sim_mix").rank("ordinal", descending=True).over("qi").cast(pl.Float32)
                           .alias("sim_mix_rank_q"))
        t = add_second_best(t, "sim_mix", "qi", "sim_mix_margin_q")
        t = t.with_columns(s1_ids.gather(t["qi"]).alias("s1"), s23_ids.gather(t["ci"]).alias("m"))
        log(split, f"chunk {s // chunk_s1}: {t.shape}")
        yield t


def feature_columns(tab):
    drop = {"qi", "ci", "s1", "m", "country", "y", "fold", "p", "is_val"}
    return [c for c in tab.columns if c not in drop]


def decode(pairs, thr, excl=True):
    """pairs: frame (s1, m, p). Exclusivity: each m keeps only its best S1; then threshold."""
    d = pairs
    if excl:
        d = d.filter(pl.col("p") == pl.col("p").max().over("m"))
    return d.filter(pl.col("p") >= thr).select("s1", "m")


def decode_f05(pairs, excl=True, beta=0.5, floor=0.02):
    """Expected-F_beta set selection per S1 (approximation by ratio of expectations).

    For each S1 with candidate probabilities p (sorted desc) the expected score of
    predicting the top-k is ~ (1+b2) * sum_{i<=k} p_i / (k + b2 * sum_all p);
    predicting the empty set scores P(no true match) = prod(1 - p_i). Pick the best.
    """
    b2 = beta * beta
    d = pairs
    if excl:
        d = d.filter(pl.col("p") == pl.col("p").max().over("m"))
    d = d.filter(pl.col("p") >= floor).sort(["s1", "p"], descending=[False, True])
    d = d.with_columns(
        pl.col("p").cum_sum().over("s1").alias("_cum"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
        pl.col("p").sum().over("s1").alias("_S"),
        (1 - pl.col("p")).clip(1e-6, 1).log().sum().over("s1").exp().alias("_p0"))
    d = d.with_columns(((1 + b2) * pl.col("_cum") / (pl.col("_k") + b2 * pl.col("_S"))).alias("_e"))
    d = d.with_columns(pl.col("_e").max().over("s1").alias("_emax"))
    kbest = d.filter(pl.col("_e") == pl.col("_emax")).group_by("s1").agg(pl.col("_k").min().alias("_kb"))
    d = d.join(kbest, on="s1").filter((pl.col("_k") <= pl.col("_kb")) & (pl.col("_emax") > pl.col("_p0")))
    return d.select("s1", "m")


def score_context(p):
    """Context features from stage-1 probabilities p (frame qi, ci, p1) over ALL pairs:
    how this pair ranks within its S1 and among all S1s competing for the same candidate."""
    p = p.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("qi").cast(pl.Float32).alias("p1_rank_q"),
        (pl.col("p1") - pl.col("p1").max().over("qi")).alias("p1_gap_q"),
        pl.col("p1").sum().over("qi").alias("p1_sum_q"),
        (pl.col("p1") > 0.5).sum().over("qi").cast(pl.Float32).alias("p1_n50_q"),
        pl.col("p1").rank("ordinal", descending=True).over("ci").cast(pl.Float32).alias("p1_rank_c"),
        (pl.col("p1") - pl.col("p1").max().over("ci")).alias("p1_gap_c"),
        pl.col("p1").sum().over("ci").alias("p1_sum_c"),
        (pl.col("p1") > 0.3).sum().over("ci").cast(pl.Float32).alias("p1_n30_c"),
    )
    p = add_second_best(p, "p1", "qi", "p1_margin_q")
    p = add_second_best(p, "p1", "ci", "p1_margin_c")
    return p
