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
ap.add_argument("--k_rev", type=int, default=0, help="dense stage: also keep top-k S1 per S2/S3 record")
ap.add_argument("--cap_rdense", type=int, default=0)
ap.add_argument("--splits", default="train,test", help="splits for the norm/block/dense stages")
ap.add_argument("--routes", default="tok,key")
ap.add_argument("--tok_max_df", type=float, default=0.01)
ap.add_argument("--key_cap", type=int, default=600)
ap.add_argument("--tok_extra", action="store_true", help="add address bigrams + compact name token to TF-IDF docs")
ap.add_argument("--stage2", action="store_true", help="two-stage model with stage-1 score context")
ap.add_argument("--frac_a", type=float, default=0.25, help="stage-2: fraction of train S1 for the stage-1 model")
ap.add_argument("--frac_b", type=float, default=0.25, help="stage-2: fraction of train S1 for the stage-2 model")
ap.add_argument("--cap_tok", type=int, default=30, help="keep tok-route candidates with rank <= this")
ap.add_argument("--cap_key", type=int, default=30)
ap.add_argument("--cap_dense", type=int, default=30)
ap.add_argument("--reuse", default=None, help="dir with cached *_s1n/_s23n/_cand parquet files to copy in")
ap.add_argument("--train_frac", type=float, default=1.0, help="fraction of train S1 used to fit the model")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--chunk_s1", type=int, default=150_000, help="S1 rows per feature chunk")
ap.add_argument("--rounds", type=int, default=1500)
ap.add_argument("--neg_rate", type=float, default=1.0,
                help="keep this fraction of EASY training negatives (weighted 1/rate); hard negatives always kept")
ap.add_argument("--global_sims", action="store_true", help="name/address sims + competition over all pairs")
ap.add_argument("--model", default="lgb", choices=["lgb", "xgb"], help="xgb = XGBoost on GPU (batched)")
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
            for f in (glob.glob(os.path.join(d, "*.parquet")) + glob.glob(os.path.join(d, "model*.txt")) +
                      glob.glob(os.path.join(d, "model*.json")) + glob.glob(os.path.join(d, "cfg.json"))):
                dst = os.path.join(a.work, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
                    log("reusing", f)
dd = find_dataset_dir(a.data)
import ber.pipeline as _bp
_bp.GLOBAL_SIMS = a.global_sims

SPLITS = a.splits.split(",")
if "norm" in stages:
    for sp in SPLITS:
        stage_normalize(dd, sp, a.work, a.indic, a.jobs)
if "block" in stages:
    for sp in SPLITS:
        stage_block(sp, a.work, k_tok=a.k_tok, n_threads=a.jobs, routes=tuple(a.routes.split(",")),
                    tok_max_df=a.tok_max_df, key_cap=a.key_cap, tok_extra=a.tok_extra)
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
        stage_dense(sp, a.work, k=a.k_dense, dataset_dir=dd, k_rev=a.k_rev)
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
        if os.path.exists(f"{a.work}/train_dense_rev.parquet"):
            rv = pl.read_parquet(f"{a.work}/train_dense_rev.parquet")
            rv = rv.with_columns(pl.Series("s1", s1i[rv["qi"].to_numpy()]), pl.Series("m", s2i[rv["ci"].to_numpy()]))
            rv = rv.join(ed, on=["s1", "m"], how="left")
            for k in [1, 2, 3, 5]:
                fw = dn.filter(pl.col("dense_rank") <= 20).select("s1", "m")
                un = pl.concat([fw, rv.filter(pl.col("rdense_rank") <= k).select("s1", "m")]).unique()
                log(f"reverse recall@{k}", round(rv.filter((pl.col("rdense_rank") <= k) & (pl.col("y") == 1)).height / ed.height, 4),
                    f"| forward@20 + reverse@{k}:", round(un.join(ed, on=["s1", "m"]).height / ed.height, 4),
                    "pairs/S1", round(un.height / len(s1i), 1))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=a.jobs)

CAPS = {"tok": a.cap_tok, "key": a.cap_key, "dense": a.cap_dense, "rdense": a.cap_rdense}

def fit_lgb(X, y, isv, cols, tag, w=None):
    dtr = lgb.Dataset(X[~isv], y[~isv], weight=None if w is None else w[~isv], feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(X[isv], y[isv], reference=dtr)
    bst = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    bst.save_model(f"{a.work}/{tag}.txt")
    imp = sorted(zip(cols, bst.feature_importance("gain")), key=lambda x: -x[1])
    log(tag, "best_iter", bst.best_iteration, "top features", [(c, int(g)) for c, g in imp[:25]])
    return bst


import shutil
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  device="cuda" if shutil.which("nvidia-smi") else "cpu",
                  eta=0.08, max_depth=0, grow_policy="lossguide", max_leaves=255, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256)


class _Batches:
    """xgboost.DataIter over lists of (X, y) chunks, so the full matrix never exists in host RAM."""
    def __new__(cls, Xs, ys, cols, ws=None):
        import xgboost as xgb

        class It(xgb.DataIter):
            def __init__(self):
                self.i = 0
                super().__init__()

            def next(self, input_data):
                if self.i == len(Xs):
                    return False
                input_data(data=Xs[self.i], label=ys[self.i], feature_names=cols,
                           weight=None if ws is None else ws[self.i])
                self.i += 1
                return True

            def reset(self):
                self.i = 0
        return It()


def fit_xgb(Xtr, ytr, Xva, yva, cols, tag, wtr=None):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(_Batches(Xtr, ytr, cols, wtr), max_bin=XGB_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(_Batches(Xva, yva, cols), ref=dtr)
    bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=a.rounds, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=200)
    bst.save_model(f"{a.work}/{tag}.json")
    imp = sorted(bst.get_score(importance_type="total_gain").items(), key=lambda x: -x[1])
    log(tag, "best_iter", bst.best_iteration, "top features", [(c, int(g)) for c, g in imp[:25]])
    return bst


def predict(model, X, best_iter):
    if a.model == "xgb" or not hasattr(model, "num_trees"):
        return model.inplace_predict(X, iteration_range=(0, best_iter + 1))
    return model.predict(X, num_iteration=best_iter)


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


def collect(split, cand, mask, lab, ctx=None, cols=None, seed=0):
    """Featurise the S1 rows in `mask`; return train/val chunk lists (with easy-negative sampling on
    the training rows only), the validation (s1, m, y[, p1]) frame and the feature columns."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, wtr, Xva, yva, vas = [], [], [], [], [], []
    for t in iter_pair_tables(split, a.work, cand, s1_filter=mask, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
        if ctx is not None:
            t = t.join(ctx, on=["qi", "ci"], how="left")
        t = t.join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), is_val("s1").alias("is_val"))
        cols = cols or feature_columns(t)
        tr = t.filter(~pl.col("is_val"))
        w = np.ones(tr.height, np.float32)
        if a.neg_rate < 1:
            hard = ((tr["y"] == 1) | (tr["sim_mix_rank_q"] <= 5) | (tr["tok_score_rank_q"] <= 5)
                    | ((tr["dense_score_rank_q"] <= 5) if "dense_score_rank_q" in tr.columns else False)).to_numpy()
            keep = hard | (rng.random(tr.height) < a.neg_rate)
            w = np.where(hard, 1.0, 1.0 / a.neg_rate).astype(np.float32)[keep]
            tr = tr.filter(pl.Series(keep))
        Xtr.append(tr.select(cols).to_numpy().astype(np.float32)); ytr.append(tr["y"].to_numpy().astype(np.float32))
        wtr.append(w)
        tv = t.filter(pl.col("is_val"))
        Xva.append(tv.select(cols).to_numpy().astype(np.float32)); yva.append(tv["y"].to_numpy().astype(np.float32))
        vas.append(tv.select("s1", "m", "y", *(["p1"] if "p1" in tv.columns else [])))
    log(split, "collected train", sum(map(len, ytr)), "val", sum(map(len, yva)), "pos", int(sum(y.sum() for y in ytr)))
    return Xtr, ytr, wtr, Xva, yva, pl.concat(vas), cols


def fit_model(Xtr, ytr, wtr, Xva, yva, cols, tag):
    if a.model == "xgb":
        bst = fit_xgb(Xtr, ytr, Xva, yva, cols, tag, wtr)
        return bst, bst.best_iteration
    X = np.concatenate(Xtr + Xva); y = np.concatenate(ytr + yva)
    w = np.concatenate(wtr + [np.ones(len(v), np.float32) for v in yva])
    isv = np.zeros(len(y), bool); isv[sum(map(len, ytr)):] = True
    bst = fit_lgb(X, y, isv, cols, tag, w)
    return bst, bst.best_iteration


def load_model(tag, kind):
    if kind == "xgb":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(f"{a.work}/{tag}.json"); return b
    return lgb.Booster(model_file=f"{a.work}/{tag}.txt")


def score_all(split, cand, model, cols, best_iter, mask=None):
    """Stage-1 probability for every candidate pair (optionally only S1 rows in mask)."""
    ps = []
    for t in iter_pair_tables(split, a.work, cand, s1_filter=mask, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
        ps.append(t.select("qi", "ci").with_columns(pl.Series("p1", predict(
            model, t.select(cols).to_numpy().astype(np.float32), best_iter)).cast(pl.Float32)))
    return pl.concat(ps)


if "train" in stages:
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "country"])
    h = s1.select((pl.col("entity_id").hash(11) % 1000).alias("h"))["h"].to_numpy()
    edges = read_ground_truth(dd)
    lab = edges.with_columns(pl.lit(1, pl.Int8).alias("y"))
    ids_all = s1.rename({"entity_id": "s1"})
    if not a.stage2:
        use = h < a.train_frac * 1000
        cand = load_candidates("train", a.work, CAPS, keep_qi=use)
        Xtr, ytr, wtr, Xva, yva, vfr, cols = collect("train", cand, use, lab)
        del cand
        bst, best_iter = fit_model(Xtr, ytr, wtr, Xva, yva, cols, "model")
        vas = [vfr]
        pv = np.concatenate([predict(bst, x, best_iter) for x in Xva])
        va = pl.concat(vas).with_columns(pl.Series("p", pv))
        del Xva, yva
        Xtr = ytr = wtr = None     # free training matrices before the test stage
        import gc; gc.collect()
        ids = ids_all.filter(pl.Series(use)).filter(is_val("s1"))
        cfg = {"cols": cols, "best_iter": best_iter, "stage2": False, "model": a.model}
    else:
        A = h < a.frac_a * 1000
        B = (h >= a.frac_a * 1000) & (h < (a.frac_a + a.frac_b) * 1000)
        cand = load_candidates("train", a.work, CAPS)
        # stage 1 on A
        Xtr, ytr, wtr, Xva, yva, _, cols = collect("train", cand, A, lab)
        b1, it1 = fit_model(Xtr, ytr, wtr, Xva, yva, cols, "model1")
        Xtr = ytr = wtr = Xva = yva = None; import gc; gc.collect()
        # stage-1 scores for EVERY train pair (A rows are in-sample; they only act as competitors)
        ctx = score_context(score_all("train", cand, b1, cols, it1))
        ctx = ctx.filter(pl.col("qi").is_in(pl.Series(np.where(B)[0]).cast(pl.Int32).implode()))
        gc.collect()
        # stage 2 on B
        Xtr, ytr, wtr, Xva, yva, tbv, cols2 = collect("train", cand, B, lab, ctx=ctx, seed=1)
        del cand, ctx
        bst, it2 = fit_model(Xtr, ytr, wtr, Xva, yva, cols2, "model")
        va = tbv.select("s1", "m", "y").with_columns(
            pl.Series("p", np.concatenate([predict(bst, x, it2) for x in Xva])))
        va1 = tbv.select("s1", "m", "y", pl.col("p1").alias("p"))
        Xtr = ytr = wtr = Xva = yva = None; gc.collect()
        ids = ids_all.filter(pl.Series(B)).filter(is_val("s1"))
        truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
        log("---- stage-1 only on stage-2 validation rows ----")
        tune_decoder(va1, ids, truth)
        cfg = {"cols": cols2, "cols1": cols, "best_iter": it2, "best_iter1": it1, "stage2": True, "model": a.model}
    va.write_parquet(f"{a.work}/val_scores.parquet")
    truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
    log("---- final model ----")
    best = tune_decoder(va, ids, truth)
    cfg.update({"thr": best[1][0], "excl": best[1][1], "val_f05": best[0], "caps": CAPS})
    json.dump(cfg, open(f"{a.work}/cfg.json", "w"))

if "test" in stages:
    cfg = json.load(open(f"{a.work}/cfg.json"))
    a.model = cfg.get("model", "lgb")
    if a.model == "xgb":
        import xgboost as xgb
        bst = xgb.Booster(); bst.load_model(f"{a.work}/model.json")
    else:
        bst = lgb.Booster(model_file=f"{a.work}/model.txt")
    cand = load_candidates("test", a.work, cfg.get("caps", CAPS))
    parts = []
    import gc
    if not cfg.get("stage2"):
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
            p = predict(bst, t.select(cfg["cols"]).to_numpy().astype(np.float32), cfg["best_iter"])
            parts.append(t.select("qi", "ci").with_columns(pl.Series("p", p).cast(pl.Float32)))   # compact
            del t, p; gc.collect()
        del cand
    else:
        b1 = load_model("model1", a.model)
        ctx = score_context(score_all("test", cand, b1, cfg["cols1"], cfg["best_iter1"]))
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
            t = t.join(ctx, on=["qi", "ci"], how="left")
            p = predict(bst, t.select(cfg["cols"]).to_numpy().astype(np.float32), cfg["best_iter"])
            parts.append(t.select("qi", "ci").with_columns(pl.Series("p", p).cast(pl.Float32)))
            del t, p; gc.collect()
        del cand, ctx
    tab = pl.concat(parts); del parts
    s1_all = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"]
    s23_all = pl.read_parquet(f"{a.work}/test_s23n.parquet", columns=["entity_id"])["entity_id"]
    tab = tab.with_columns(s1_all.gather(tab["qi"]).alias("s1"), s23_all.gather(tab["ci"]).alias("m")).drop("qi", "ci")
    tab.write_parquet(f"{a.work}/test_scores.parquet")
    pred = decode_f05(tab, cfg["excl"]) if cfg["thr"] == "f05" else decode(tab, cfg["thr"], cfg["excl"])
    s1_ids = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_id_lists(f"{out}/candidate_pairs.tsv", s1_ids, tab.select("s1", "m"), "candidate_entity_ids")
    write_id_lists(f"{out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    log("test written; S1 with >=1 match:", pred["s1"].n_unique(), "/", len(s1_ids), "pairs", pred.height)
    ctry = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    log(pred.join(ctry, on="s1").group_by("country").agg(pl.len(), pl.col("s1").n_unique()))
