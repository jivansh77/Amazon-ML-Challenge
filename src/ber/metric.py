"""Macro F0.5 exactly as defined by the challenge (per-S1, singletons included)."""
import polars as pl


def macro_f05(pred, truth, s1_ids, beta=0.5, by=None):
    """pred/truth: frames (s1, m) of pairs. s1_ids: frame with column s1 (+ optional `by`).

    Returns mean F_beta over s1_ids; if `by` is given, also a per-group breakdown.
    """
    b2 = beta * beta
    p = pred.select("s1", "m").unique().with_columns(pl.lit(1).alias("_p"))
    t = truth.select("s1", "m").unique().with_columns(pl.lit(1).alias("_t"))
    j = p.join(t, on=["s1", "m"], how="full", coalesce=True)
    g = j.group_by("s1").agg(pl.col("_p").sum().alias("np"), pl.col("_t").sum().alias("nt"),
                             (pl.col("_p").is_not_null() & pl.col("_t").is_not_null()).sum().alias("tp"))
    d = s1_ids.join(g, on="s1", how="left").fill_null(0)
    d = d.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise((1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * (pl.col("nt") - pl.col("tp"))
                                              + (pl.col("np") - pl.col("tp"))))
        .alias("f"))
    res = {"f05": d["f"].mean(),
           "f05_singletons": d.filter(pl.col("nt") == 0)["f"].mean(),
           "f05_matched": d.filter(pl.col("nt") > 0)["f"].mean()}
    if by:
        res["by"] = d.group_by(by).agg(pl.col("f").mean(), pl.len()).sort(by)
    return res
