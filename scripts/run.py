"""Run the pipeline.

  python scripts/run.py --data <dataset root> --work <work dir> --stages norm,block,train,test

Stages
  norm   normalise train + test sources           -> {split}_s1n / _s23n.parquet
  block  candidate generation for train + test    -> {split}_cand.parquet
  train  pair features, LightGBM on 90% of train S1, threshold tuned on the other 10%,
         then refit on all train S1                -> model.txt, cfg.json, val report
  test   score test candidates, decode, write output/matching_results.tsv + candidate_pairs.tsv
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lightgbm as lgb
import numpy as np
import polars as pl

from ber.io import find_dataset_dir, read_ground_truth, write_id_lists
from ber.metric import macro_f05
from ber.pipeline import (log, stage_normalize, stage_block, stage_dense, load_candidates, iter_pair_tables,
                          feature_columns, decode, decode_f05, score_context)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--stages", default="norm,block,train,test")
ap.add_argument("--indic", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "indic_dict.json"))
ap.add_argument("--k_tok", type=int, default=30)
ap.add_argument("--k_dense", type=int, default=20)
ap.add_argument("--splits", default="train,test", help="splits for the norm/block/dense stages")
ap.add_argument("--routes", default="tok,key")
ap.add_argument("--tok_max_df", type=float, default=0.01)
ap.add_argument("--key_cap", type=int, default=600)
ap.add_argument("--stage2", action="store_true", help="two-stage model with stage-1 score context")
ap.add_argument("--frac_a", type=float, default=0.25, help="stage-2: fraction of train S1 for the stage-1 model")
ap.add_argument("--frac_b", type=float, default=0.25, help="stage-2: fraction of train S1 for the stage-2 model")
ap.add_argument("--cap_tok", type=int, default=30, help="keep tok-route candidates with rank <= this")
ap.add_argument("--cap_key", type=int, default=30)
ap.add_argument("--cap_dense", type=int, default=30)
ap.add_argument("--reuse", default=None, help="dir with cached *_s1n/_s23n/_cand parquet files to copy in")
ap.add_argument("--train_frac", type=float, default=1.0, help="fraction of train S1 used to fit the model")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--rounds", type=int, default=1500)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
out = a.out or os.path.join(a.work, "output")
os.makedirs(out, exist_ok=True)
stages = a.stages.split(",")
if a.reuse:
    # comma-separated glob patterns of directories holding cached parquet stage outputs
    import glob
    for pat in a.reuse.split(","):
        for d in glob.glob(pat, recursive=True):
            for f in glob.glob(os.path.join(d, "*.parquet")):
                dst = os.path.join(a.work, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
                    log("reusing", f)
dd = find_dataset_dir(a.data)

SPLITS = a.splits.split(",")
if "norm" in stages:
    for sp in SPLITS:
        stage_normalize(dd, sp, a.work, a.indic, a.jobs)
if "block" in stages:
    for sp in SPLITS:
        stage_block(sp, a.work, k_tok=a.k_tok, n_threads=a.jobs, routes=tuple(a.routes.split(",")),
                    tok_max_df=a.tok_max_df, key_cap=a.key_cap)
        if sp == "train":   # full-scale recall report per route
            s1i = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id"])["entity_id"].to_numpy()
            s2i = pl.read_parquet(f"{a.work}/train_s23n.parquet", columns=["entity_id"])["entity_id"].to_numpy()
            c = pl.read_parquet(f"{a.work}/train_cand.parquet")
            c = c.with_columns(pl.Series("s1", s1i[c["qi"].to_numpy()]), pl.Series("m", s2i[c["ci"].to_numpy()]))
            ed = read_ground_truth(dd).with_columns(pl.lit(1).alias("y"))
            c = c.join(ed, on=["s1", "m"], how="left")
            for r in [x for x in ["tok", "key"] if f"{x}_rank" in c.columns]:
                log(r, [(k, round(c.filter((pl.col(f"{r}_rank") <= k) & (pl.col("y") == 1)).height / ed.height, 4))
                        for k in [5, 10, 20, 30]])
            log("union recall", round(c["y"].sum() / ed.height, 4), "cands/S1", round(c.height / len(s1i), 1))

if "dense" in stages:
    for sp in SPLITS:
        stage_dense(sp, a.work, k=a.k_dense, dataset_dir=dd)
    if "train" in SPLITS:   # full-scale recall of the dense route
        from ber.io import read_split
        _s1, _s23 = read_split(dd, "train")
        s1i, s2i = _s1["entity_id"].to_numpy(), _s23["entity_id"].to_numpy()
        del _s1, _s23
        ed = read_ground_truth(dd).with_columns(pl.lit(1).alias("y"))
        dn = pl.read_parquet(f"{a.work}/train_dense.parquet")
        dn = dn.with_columns(pl.Series("s1", s1i[dn["qi"].to_numpy()]), pl.Series("m", s2i[dn["ci"].to_numpy()]))
        dn = dn.join(ed, on=["s1", "m"], how="left")
        for k in [5, 10, 20, 30, 50]:
            log(f"dense recall@{k}", round(dn.filter((pl.col("dense_rank") <= k) & (pl.col("y") == 1)).height / ed.height, 4))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=a.jobs)

CAPS = {"tok": a.cap_tok, "key": a.cap_key, "dense": a.cap_dense}

def fit_lgb(X, y, isv, cols, tag):
    dtr = lgb.Dataset(X[~isv], y[~isv], feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(X[isv], y[isv], reference=dtr)
    bst = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    bst.save_model(f"{a.work}/{tag}.txt")
    imp = sorted(zip(cols, bst.feature_importance("gain")), key=lambda x: -x[1])
    log(tag, "best_iter", bst.best_iteration, "top features", [(c, int(g)) for c, g in imp[:25]])
    return bst


def tune_decoder(va, ids, truth):
    rec = va["y"].sum() / max(truth.height, 1)
    log("val blocking recall", round(rec, 4), "oracle f05",
        round(macro_f05(va.filter(pl.col("y") == 1), truth, ids)["f05"], 4))
    best = (0, None)
    for excl in [False, True]:
        for thr in np.arange(0.2, 0.91, 0.05):
            r = macro_f05(decode(va, thr, excl), truth, ids)
            if r["f05"] > best[0]:
                best = (r["f05"], (float(thr), excl))
            log(f"excl={excl} thr={thr:.2f} f05={r['f05']:.4f} sing={r['f05_singletons']:.4f} matched={r['f05_matched']:.4f}")
    for excl in [False, True]:
        r = macro_f05(decode_f05(va, excl), truth, ids)
        log(f"F05-decoder excl={excl} f05={r['f05']:.4f} sing={r['f05_singletons']:.4f} matched={r['f05_matched']:.4f}")
        if r["f05"] > best[0]:
            best = (r["f05"], ("f05", excl))
    thr, excl = best[1]
    pv = decode_f05(va, excl) if thr == "f05" else decode(va, thr, excl)
    log("BEST", best, macro_f05(pv, truth, ids, by="country"))
    return best


def is_val(col):
    return (pl.col(col).hash(13) % 10) == 0


if "train" in stages:
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "country"])
    h = s1.select((pl.col("entity_id").hash(11) % 1000).alias("h"))["h"].to_numpy()
    edges = read_ground_truth(dd)
    lab = edges.with_columns(pl.lit(1, pl.Int8).alias("y"))
    cand = load_candidates("train", a.work, CAPS)
    ids_all = s1.rename({"entity_id": "s1"})
    if not a.stage2:
        use = h < a.train_frac * 1000
        Xs, ys, vs, vas, cols = [], [], [], [], None
        for t in iter_pair_tables("train", a.work, cand, s1_filter=use, n_jobs=a.jobs):
            t = t.join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), is_val("s1").alias("is_val"))
            cols = cols or feature_columns(t)
            Xs.append(t.select(cols).to_numpy().astype(np.float32)); ys.append(t["y"].to_numpy())
            vs.append(t["is_val"].to_numpy()); vas.append(t.filter(pl.col("is_val")).select("s1", "m", "y"))
        del cand
        X, y, isv = np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs); del Xs, ys, vs
        log("features", len(cols), "pairs", len(y), "pos", int(y.sum()))
        bst = fit_lgb(X, y, isv, cols, "model")
        va = pl.concat(vas).with_columns(pl.Series("p", bst.predict(X[isv], num_iteration=bst.best_iteration)))
        del X
        ids = ids_all.filter(pl.Series(use)).filter(is_val("s1"))
        cfg = {"cols": cols, "best_iter": bst.best_iteration, "stage2": False}
    else:
        A = h < a.frac_a * 1000
        B = (h >= a.frac_a * 1000) & (h < (a.frac_a + a.frac_b) * 1000)
        # stage 1: fit on A
        Xs, ys, vs, cols = [], [], [], None
        for t in iter_pair_tables("train", a.work, cand, s1_filter=A, n_jobs=a.jobs):
            t = t.join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), is_val("s1").alias("is_val"))
            cols = cols or feature_columns(t)
            Xs.append(t.select(cols).to_numpy().astype(np.float32)); ys.append(t["y"].to_numpy()); vs.append(t["is_val"].to_numpy())
        X, y, isv = np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs); del Xs, ys, vs
        log("stage1 features", len(cols), "pairs", len(y), "pos", int(y.sum()))
        b1 = fit_lgb(X, y, isv, cols, "model1")
        del X
        # stage-1 scores for every non-A pair (out-of-sample); keep B feature rows for stage 2
        ps, Bt = [], []
        for t in iter_pair_tables("train", a.work, cand, s1_filter=~A, n_jobs=a.jobs):
            t = t.with_columns(pl.Series("p1", b1.predict(t.select(cols).to_numpy().astype(np.float32),
                                                         num_iteration=b1.best_iteration)).cast(pl.Float32))
            ps.append(t.select("qi", "ci", "p1"))
            Bt.append(t.filter(pl.col("qi").is_in(
                pl.Series(np.where(B)[0]).cast(pl.Int32).implode())))
        # A pairs: in-sample stage-1 scores (only used as competitors in the context features)
        for t in iter_pair_tables("train", a.work, cand, s1_filter=A, n_jobs=a.jobs):
            ps.append(t.select("qi", "ci").with_columns(pl.Series("p1", b1.predict(
                t.select(cols).to_numpy().astype(np.float32), num_iteration=b1.best_iteration)).cast(pl.Float32)))
        del cand
        ctx = score_context(pl.concat(ps)); del ps
        tb = pl.concat(Bt).drop("p1").join(ctx, on=["qi", "ci"], how="left"); del Bt, ctx
        tb = tb.join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), is_val("s1").alias("is_val"))
        cols2 = feature_columns(tb)
        X = tb.select(cols2).to_numpy().astype(np.float32)
        y, isv = tb["y"].to_numpy(), tb["is_val"].to_numpy()
        log("stage2 features", len(cols2), "pairs", len(y), "pos", int(y.sum()))
        bst = fit_lgb(X, y, isv, cols2, "model")
        va = tb.filter(pl.col("is_val")).select("s1", "m", "y").with_columns(
            pl.Series("p", bst.predict(X[isv], num_iteration=bst.best_iteration)))
        # stage-1-only score on the same validation rows, for comparison
        va1 = tb.filter(pl.col("is_val")).select("s1", "m", "y", pl.col("p1").alias("p"))
        del X, tb
        ids = ids_all.filter(pl.Series(B)).filter(is_val("s1"))
        truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
        log("---- stage-1 only on stage-2 validation rows ----")
        tune_decoder(va1, ids, truth)
        cfg = {"cols": cols2, "cols1": cols, "best_iter": bst.best_iteration, "best_iter1": b1.best_iteration,
               "stage2": True}
    va.write_parquet(f"{a.work}/val_scores.parquet")
    truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
    log("---- final model ----")
    best = tune_decoder(va, ids, truth)
    cfg.update({"thr": best[1][0], "excl": best[1][1], "val_f05": best[0], "caps": CAPS})
    json.dump(cfg, open(f"{a.work}/cfg.json", "w"))

if "test" in stages:
    cfg = json.load(open(f"{a.work}/cfg.json"))
    bst = lgb.Booster(model_file=f"{a.work}/model.txt")
    cand = load_candidates("test", a.work, cfg.get("caps", CAPS))
    parts = []
    if not cfg.get("stage2"):
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs):
            p = bst.predict(t.select(cfg["cols"]).to_numpy().astype(np.float32), num_iteration=cfg["best_iter"])
            parts.append(t.select("s1", "m").with_columns(pl.Series("p", p)))
        del cand
    else:
        b1 = lgb.Booster(model_file=f"{a.work}/model1.txt")
        ps = []
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs):
            ps.append(t.select("qi", "ci").with_columns(pl.Series("p1", b1.predict(
                t.select(cfg["cols1"]).to_numpy().astype(np.float32), num_iteration=cfg["best_iter1"])).cast(pl.Float32)))
        ctx = score_context(pl.concat(ps)); del ps
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs):
            t = t.join(ctx, on=["qi", "ci"], how="left")
            p = bst.predict(t.select(cfg["cols"]).to_numpy().astype(np.float32), num_iteration=cfg["best_iter"])
            parts.append(t.select("s1", "m").with_columns(pl.Series("p", p)))
        del cand, ctx
    tab = pl.concat(parts)
    tab.write_parquet(f"{a.work}/test_scores.parquet")
    pred = decode_f05(tab, cfg["excl"]) if cfg["thr"] == "f05" else decode(tab, cfg["thr"], cfg["excl"])
    s1_ids = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_id_lists(f"{out}/candidate_pairs.tsv", s1_ids, tab.select("s1", "m"), "candidate_entity_ids")
    write_id_lists(f"{out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    log("test written; S1 with >=1 match:", pred["s1"].n_unique(), "/", len(s1_ids), "pairs", pred.height)
    ctry = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    log(pred.join(ctry, on="s1").group_by("country").agg(pl.len(), pl.col("s1").n_unique()))
