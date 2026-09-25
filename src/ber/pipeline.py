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


def load_candidates(split, work, caps, keep_qi=None, drop_qi=None, dup_mask=None, dup_near=10):
    """Union of all available routes, pruned by per-route rank caps, with global context
    features computed on blocking scores (cheap, and identical at train and test time)."""
    cand = pl.read_parquet(f"{work}/{split}_cand.parquet")
    for f in ["dense", "dense_rev"]:
        if os.path.exists(f"{work}/{split}_{f}.parquet"):
            cand = cand.join(pl.read_parquet(f"{work}/{split}_{f}.parquet"), on=["qi", "ci"], how="full", coalesce=True)
    if drop_qi is not None:
        # simulate the test density: dropped S1 disappear entirely, so their S2/S3 records become
        # orphans that compete for the remaining S1 (all context features are computed without them)
        cand = cand.filter(~pl.col("qi").is_in(pl.Series(np.where(drop_qi)[0]).cast(pl.Int32).implode()))
    keep = pl.lit(False)
    routes = [r for r in ROUTES if r[2] in cand.columns]      # only routes that produced candidates
    for r, sc, rk in routes:
        cand = cand.with_columns(pl.col(sc).cast(pl.Float32), pl.col(rk).cast(pl.Float32))
        keep = keep | (pl.col(rk) <= caps.get(r, 0))
    n0 = cand.height
    cand = cand.filter(keep)
    extra = [pl.col("key_n").cast(pl.Float32)] if "key_n" in cand.columns else []
    cand = cand.select("qi", "ci", *[c for _, sc, rk in routes for c in (sc, rk)], *extra)
    n23 = pl.scan_parquet(f"{work}/{split}_s23n.parquet").select(pl.len()).collect().item()
    if dup_mask is not None:
        # simulate the test's decoy density: every decoy record (belongs to no S1) that is near an S1
        # gets a virtual copy (ci + n23) BEFORE the competition/context features are computed
        near = pl.lit(False)
        for r, sc, rk in routes:
            near = near | (pl.col(rk) <= (2 if r == "rdense" else dup_near))
        dec = pl.Series(np.where(dup_mask)[0]).cast(pl.Int32).implode()
        d = cand.filter(pl.col("ci").is_in(dec) & near).with_columns((pl.col("ci") + n23).cast(pl.Int32).alias("ci"))
        log(split, "duplicated decoy pairs", d.height)
        cand = pl.concat([cand, d])
    cand = cand.with_columns(pl.when(pl.col("ci") >= n23).then(pl.col("ci") - n23).otherwise(pl.col("ci"))
                             .cast(pl.Int32).alias("cr"))
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
        t = t.with_columns(pl.Series("src", src[t["cr"].to_numpy()]),
                           ((pl.col("n_tset") + pl.col("a_tset")) / 2).alias("sim_mix"))
        t = t.with_columns((pl.col("sim_mix") - pl.col("sim_mix").max().over("qi")).alias("sim_mix_gap_q"),
                           pl.col("sim_mix").rank("ordinal", descending=True).over("qi").cast(pl.Float32)
                           .alias("sim_mix_rank_q"))
        t = add_second_best(t, "sim_mix", "qi", "sim_mix_margin_q")
        t = t.with_columns(s1_ids.gather(t["qi"]).alias("s1"), s23_ids.gather(t["cr"]).alias("m"))
        t = t.with_columns(pl.when(pl.col("ci") != pl.col("cr")).then(pl.col("m") + "#dup").otherwise(pl.col("m")).alias("m"))
        log(split, f"chunk {s // chunk_s1}: {t.shape}")
        yield t


def feature_columns(tab):
    drop = {"qi", "ci", "cr", "s1", "m", "country", "y", "fold", "p", "is_val"}
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


def add_triangle(split, work, ctx, jobs=4, chunk=5_000_000):
    """Consistency features: compare each candidate S2/S3 record with the S1's most confident OTHER
    candidate (its "anchor"). True matches are copies of the same business and resemble each other;
    a false merge is usually an outlier. ctx: (qi, ci, p1, p1_rank_q, ...) for all pairs of the S1s."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    top = ctx.filter(pl.col("p1_rank_q") <= 2).select("qi", "ci", "p1", "p1_rank_q")
    a1 = top.filter(pl.col("p1_rank_q") == 1).select("qi", pl.col("ci").alias("a1"), pl.col("p1").alias("a1_p"))
    a2 = top.filter(pl.col("p1_rank_q") == 2).select("qi", pl.col("ci").alias("a2"), pl.col("p1").alias("a2_p"))
    t = ctx.select("qi", "ci").join(a1, on="qi", how="left").join(a2, on="qi", how="left")
    t = t.with_columns(
        pl.when(pl.col("ci") == pl.col("a1")).then(pl.col("a2")).otherwise(pl.col("a1")).alias("anchor"),
        pl.when(pl.col("ci") == pl.col("a1")).then(pl.col("a2_p")).otherwise(pl.col("a1_p")).alias("tri_anchor_p"))
    s23 = pl.read_parquet(f"{work}/{split}_s23n.parquet", columns=["name_core", "addr_clean"])
    nm, ad = s23["name_core"], s23["addr_clean"]
    n23 = s23.height
    ci, an = t["ci"].to_numpy() % n23, t["anchor"].to_numpy()
    has = ~np.isnan(an.astype(np.float64)) if an.dtype.kind == "f" else t["anchor"].is_not_null().to_numpy()
    an_i = np.where(has, np.nan_to_num(an.astype(np.float64), nan=0), 0).astype(np.int64) % n23
    tn = np.full(len(ci), -1, np.float32); ta = np.full(len(ci), -1, np.float32)
    for s in range(0, len(ci), chunk):
        sl = slice(s, s + chunk)
        c, b = ci[sl], an_i[sl]
        tn[sl] = cpdist(nm.gather(c).to_list(), nm.gather(b).to_list(), scorer=fuzz.token_set_ratio, workers=jobs, dtype=np.float32)
        ta[sl] = cpdist(ad.gather(c).to_list(), ad.gather(b).to_list(), scorer=fuzz.token_set_ratio, workers=jobs, dtype=np.float32)
    tn[~has] = -1; ta[~has] = -1
    return ctx.with_columns(pl.Series("tri_name", tn), pl.Series("tri_addr", ta),
                            t["tri_anchor_p"].cast(pl.Float32).fill_null(-1).alias("tri_anchor_p"))


PRIOR_EDGES = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.01]


def prior_thresholds(va, te, tr_country, te_country, min_prec=0.78):
    """Per-country thresholds corrected for the test set's decoy density.

    va: validation (s1, m, y, p); te: test (s1, m, p); *_country: frames (s1, country).
    Expected TRUE matches per S1 in each probability band come from validation (countries without labels
    use the average of the labelled ones); dividing by the observed test pairs per S1 in the band estimates
    each band's precision on test. A pair helps macro F0.5 only above ~F*/(1+beta^2) ~ 0.78, so each
    country's threshold is the lowest band edge above which every band clears min_prec.
    Returns ({country: threshold}, table of estimated precisions)."""
    E = PRIOR_EDGES

    def band(c):
        e = pl.lit(None, pl.Float64)
        for lo, hi in zip(E[:-1], E[1:]):
            e = pl.when((pl.col(c) >= lo) & (pl.col(c) < hi)).then(pl.lit(lo)).otherwise(e)
        return e.alias("band")
    nv = va.select("s1").unique().join(tr_country, on="s1").group_by("country").len().rename({"len": "n"})
    vt = (va.filter(pl.col("p") >= E[0]).join(tr_country, on="s1").with_columns(band("p"))
          .group_by("country", "band").agg(pl.col("y").sum().alias("tp")).join(nv, on="country")
          .with_columns((pl.col("tp") / pl.col("n")).alias("tp_per_s1")))
    avg = vt.group_by("band").agg(pl.col("tp_per_s1").mean().alias("tp_avg"))
    nt = te_country.group_by("country").len().rename({"len": "n"})
    tt = (te.filter(pl.col("p") >= E[0]).join(te_country, on="s1").with_columns(band("p"))
          .group_by("country", "band").agg(pl.len().alias("pairs")).join(nt, on="country")
          .with_columns((pl.col("pairs") / pl.col("n")).alias("test_per_s1")))
    est = (tt.join(vt.select("country", "band", "tp_per_s1"), on=["country", "band"], how="left").join(avg, on="band")
           .with_columns((pl.coalesce("tp_per_s1", "tp_avg") / pl.col("test_per_s1")).clip(0, 1).alias("prec"))
           .sort("country", "band", descending=[False, True]))
    thr = {}
    for c in est["country"].unique().to_list():
        t = 0.99
        for b, pr in est.filter(pl.col("country") == c).select("band", "prec").iter_rows():
            if pr < min_prec:
                break
            t = b
        thr[c] = t
    return thr, est


def decode_by_country(pairs, thr, country, default=0.9):
    """Exclusivity (each record keeps its best S1), then the S1's country threshold."""
    d = pairs.filter(pl.col("p") == pl.col("p").max().over("m")).join(country, on="s1")
    d = d.filter(pl.col("p") >= pl.col("country").replace_strict(thr, default=default, return_dtype=pl.Float64))
    return d.select("s1", "m")
