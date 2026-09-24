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


def stage_block(split, work, k_tok=30, n_threads=4, routes=("tok",), tok_max_df=0.05, key_cap=600):
    s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet")
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet")
    cand = generate_candidates(s1, s23, k_tok=k_tok, k_key=k_tok, n_threads=n_threads, routes=routes,
                               tok_max_df=tok_max_df, key_cap=key_cap)
    cand.write_parquet(f"{work}/{split}_cand.parquet")
    log(split, "blocked", cand.shape, "per S1", cand.height / s1.height)


def stage_dense(split, work, k=20):
    """GPU dense retrieval per country; embeddings are not persisted (too large)."""
    from .dense import record_text, encode, topk_by_country
    s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet", columns=["business_name", "business_address", "country"])
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet", columns=["business_name", "business_address", "country"])
    parts = []
    for ctry in s1["country"].unique().to_list():
        qi = np.where(s1["country"].to_numpy() == ctry)[0]
        ci = np.where(s23["country"].to_numpy() == ctry)[0]
        qe = encode(record_text(s1[qi]))
        ce = encode(record_text(s23[ci]))
        log(split, ctry, "encoded", qe.shape, ce.shape)
        d = topk_by_country(qe, np.zeros(len(qi)), ce, np.zeros(len(ci)), k)
        parts.append(d.with_columns(pl.Series("qi", qi[d["qi"].to_numpy()]).cast(pl.Int32),
                                    pl.Series("ci", ci[d["ci"].to_numpy()]).cast(pl.Int32)))
        del qe, ce
    cand = pl.concat(parts)
    cand.write_parquet(f"{work}/{split}_dense.parquet")
    log(split, "dense candidates", cand.shape)


def add_second_best(df, col, grp, name):
    """Margin of `col` over the best *other* value in group `grp` (positive only for the top one)."""
    g = (df.group_by(grp).agg(pl.col(col).max().alias("_m1"),
                              pl.col(col).sort(descending=True).slice(1, 1).first().alias("_m2")))
    return (df.join(g, on=grp, how="left")
            .with_columns(pl.when(pl.col(col) >= pl.col("_m1"))
                          .then(pl.col(col) - pl.col("_m2").fill_null(0.0))
                          .otherwise(pl.col(col) - pl.col("_m1")).alias(name))
            .drop("_m1", "_m2"))


def build_pair_table(split, work, s1_filter=None, n_jobs=4):
    """Features for all candidates; context features are computed over ALL pairs (as at test
    time) and only then optionally restricted to S1 rows where s1_filter is True."""
    s1 = pl.read_parquet(f"{work}/{split}_s1n.parquet")
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet")
    cand = pl.read_parquet(f"{work}/{split}_cand.parquet")
    for c in ["tok_score", "tok_rank", "key_score", "key_rank", "key_n"]:
        if c not in cand.columns:
            cand = cand.with_columns(pl.lit(None, pl.Float32).alias(c))
    cand = cand.select("qi", "ci", "tok_score", "tok_rank", "key_score", "key_rank", "key_n")
    ctx = ["tok_score", "key_score"]
    if os.path.exists(f"{work}/{split}_dense.parquet"):
        dn = pl.read_parquet(f"{work}/{split}_dense.parquet")
        cand = cand.join(dn, on=["qi", "ci"], how="full", coalesce=True)
        ctx.append("dense_score")
        log(split, "with dense route", cand.height)
    cand = cand.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in ["tok_score", "key_score", "key_n"]] +
                             [pl.col(c).cast(pl.Float32).fill_null(99.0) for c in ["tok_rank", "key_rank"]])
    cand = context_features(cand, ctx)
    cand = add_second_best(cand, "tok_score", "ci", "tok_margin_c")
    cand = add_second_best(cand, "tok_score", "qi", "tok_margin_q")
    log(split, "pairs to featurise", cand.height)
    f = pair_features(s1, s23, cand, jobs=n_jobs)
    log(split, "pair features done")
    tab = pl.concat([cand, f], how="horizontal")
    del f
    tab = tab.with_columns(pl.Series("src", s23["src"].to_numpy()[tab["ci"].to_numpy()]).cast(pl.Float32),
                           ((pl.col("n_tset") + pl.col("a_tset")) / 2).alias("sim_mix"))
    tab = context_features(tab.drop("n_cand_q", "n_cand_c"), ["sim_mix"])
    tab = add_second_best(tab, "sim_mix", "qi", "sim_mix_margin_q")
    tab = add_second_best(tab, "sim_mix", "ci", "sim_mix_margin_c")
    if s1_filter is not None:
        keep = pl.Series(np.where(s1_filter)[0]).cast(pl.Int32).implode()
        tab = tab.filter(pl.col("qi").is_in(keep))
    tab = tab.with_columns(
        pl.Series("s1", s1["entity_id"].to_numpy()[tab["qi"].to_numpy()]),
        pl.Series("m", s23["entity_id"].to_numpy()[tab["ci"].to_numpy()]))
    log(split, "pair table", tab.shape)
    return tab


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
