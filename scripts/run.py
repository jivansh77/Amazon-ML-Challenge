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
from ber.pipeline import (log, stage_normalize, stage_block, stage_dense, build_pair_table, feature_columns, decode, decode_f05)

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
        stage_dense(sp, a.work, k=a.k_dense)
    if "train" in SPLITS:   # full-scale recall of the dense route
        s1i = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id"])["entity_id"].to_numpy()
        s2i = pl.read_parquet(f"{a.work}/train_s23n.parquet", columns=["entity_id"])["entity_id"].to_numpy()
        ed = read_ground_truth(dd).with_columns(pl.lit(1).alias("y"))
        dn = pl.read_parquet(f"{a.work}/train_dense.parquet")
        dn = dn.with_columns(pl.Series("s1", s1i[dn["qi"].to_numpy()]), pl.Series("m", s2i[dn["ci"].to_numpy()]))
        dn = dn.join(ed, on=["s1", "m"], how="left")
        for k in [5, 10, 20, 30, 50]:
            log(f"dense recall@{k}", round(dn.filter((pl.col("dense_rank") <= k) & (pl.col("y") == 1)).height / ed.height, 4))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=a.jobs)

if "train" in stages:
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "country"])
    use = (s1.select((pl.col("entity_id").hash(11) % 1000).alias("h"))["h"].to_numpy() < a.train_frac * 1000)
    tab = build_pair_table("train", a.work, s1_filter=use if a.train_frac < 1 else None, n_jobs=a.jobs)
    edges = read_ground_truth(dd)
    tab = tab.join(edges.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left") \
             .with_columns(pl.col("y").fill_null(0), ((pl.col("s1").hash(13) % 10) == 0).alias("is_val"))
    cols = feature_columns(tab)
    log("features", len(cols), "pairs", tab.height, "pos", tab["y"].sum())
    X = tab.select(cols).to_numpy().astype(np.float32)
    y = tab["y"].to_numpy()
    isv = tab["is_val"].to_numpy()
    va = tab.filter(pl.col("is_val")).select("s1", "m", "y")
    del tab
    dtr = lgb.Dataset(X[~isv], y[~isv], feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(X[isv], y[isv], reference=dtr)
    bst = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    bst.save_model(f"{a.work}/model.txt")
    va = va.with_columns(pl.Series("p", bst.predict(X[isv], num_iteration=bst.best_iteration)))
    del X, dtr, dva
    va.write_parquet(f"{a.work}/val_scores.parquet")
    # evaluation universe: all val S1 ids of the used subset (singletons included)
    ids = s1.rename({"entity_id": "s1"}).filter(pl.Series(use)).filter((pl.col("s1").hash(13) % 10) == 0)
    truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
    rec = va["y"].sum() / truth.height
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
    r = macro_f05(pv, truth, ids, by="country")
    log("BEST", best, r)
    imp = sorted(zip(cols, bst.feature_importance("gain")), key=lambda x: -x[1])
    log("top features", [(c, int(g)) for c, g in imp[:30]])
    json.dump({"thr": thr, "excl": excl, "cols": cols, "val_f05": best[0], "best_iter": bst.best_iteration},
              open(f"{a.work}/cfg.json", "w"))

if "test" in stages:
    cfg = json.load(open(f"{a.work}/cfg.json"))
    bst = lgb.Booster(model_file=f"{a.work}/model.txt")
    tab = build_pair_table("test", a.work, n_jobs=a.jobs)
    tab = tab.with_columns(pl.Series("p", bst.predict(tab.select(cfg["cols"]).to_numpy().astype(np.float32), num_iteration=cfg["best_iter"])))
    tab.select("s1", "m", "p").write_parquet(f"{a.work}/test_scores.parquet")
    pred = decode_f05(tab, cfg["excl"]) if cfg["thr"] == "f05" else decode(tab, cfg["thr"], cfg["excl"])
    s1_ids = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_id_lists(f"{out}/candidate_pairs.tsv", s1_ids, tab.select("s1", "m"), "candidate_entity_ids")
    write_id_lists(f"{out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    log("test written; S1 with >=1 match:", pred["s1"].n_unique(), "/", len(s1_ids), "pairs", pred.height)
    ctry = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    log(pred.join(ctry, on="s1").group_by("country").agg(pl.len(), pl.col("s1").n_unique()))
