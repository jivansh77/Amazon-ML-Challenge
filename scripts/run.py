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
                          feature_columns, decode, decode_f05, score_context, add_triangle)

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
ap.add_argument("--drop_s1", type=float, default=0.0,
                help="train/validate with this fraction of S1 removed (orphaned S2/S3 records, test-like density)")
ap.add_argument("--s2_topk", type=int, default=0, help="stage 2 only scores the stage-1 top-k per S1 (0 = all)")
ap.add_argument("--s2_minp", type=float, default=0.0, help="stage 2 only scores pairs with stage-1 p >= this")
ap.add_argument("--prior_thr", action="store_true",
                help="test stage: per-country thresholds corrected for the test decoy density (needs val_scores)")
ap.add_argument("--decoy_feats", action="store_true", help="decoy-signature features (extra/missing name words, signed house-number shift)")
ap.add_argument("--dup_decoys", action="store_true", help="duplicate near decoy candidates in training (test-like decoy density)")
ap.add_argument("--dup_near", type=int, default=10, help="only duplicate decoys within this route rank of the S1")
ap.add_argument("--triangle", action="store_true", help="stage-2 consistency features vs the S1's anchor match")
ap.add_argument("--train_countries", default=None,
                help="train both stages and the decoy odds only on these countries (comma-separated); validation keeps "
                     "every country, so the others act as labelled 'unseen countries' (a stand-in for France)")
ap.add_argument("--self_train", action="store_true",
                help="with --train_countries: pseudo-label the other countries' non-validation S1 with the stage-2 model "
                     "(p >= --st_hi -> match, <= --st_lo -> non-match), refit stage 2 on labels + pseudo-labels")
ap.add_argument("--pseudo_scores", default=None,
                help="selftrain stage: parquet (s1, m, p) of the final (blended) test probabilities used as pseudo-labels")
ap.add_argument("--st_hi", type=float, default=0.99)
ap.add_argument("--st_lo", type=float, default=0.05)
ap.add_argument("--st_weight", type=float, default=1.0)
ap.add_argument("--swap_ab", action="store_true",
                help="stage 1 on the usual stage-2 slice and stage 2 on the usual stage-1 slice (a second, diverse model); "
                     "also scores the usual model's validation S1 into val_scores_other.parquet so the two can be averaged")
ap.add_argument("--global_sims", action="store_true", help="name/address sims + competition over all pairs")
ap.add_argument("--model", default="lgb", choices=["lgb", "xgb"], help="xgb = XGBoost on GPU (batched)")
ap.add_argument("--odds_extra", default=None,
                help="parquet(s) (token, score), comma-separated: extra-word scores for words the training odds "
                     "do not know (e.g. French words learned from confident test predictions, fit_pseudo_odds.py)")
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
                if os.path.basename(f) == "test_scores.parquet":     # an output of this run, never an input
                    continue
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
          try:
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
          except Exception as ex:
            log("recall report skipped:", repr(ex))

if "dense" in stages:
    for sp in SPLITS:
        if not os.environ.get("BER_DENSE_REPORT_ONLY"):     # (testing hook)
            stage_dense(sp, a.work, k=a.k_dense, dataset_dir=dd, k_rev=a.k_rev)
    if "train" in SPLITS:   # full-scale recall of the dense routes (never allowed to fail the run)
        try:
            from ber.io import read_split
            _s1, _s23 = read_split(dd, "train")
            m1 = pl.DataFrame({"s1": _s1["entity_id"], "qi": np.arange(_s1.height, dtype=np.int64)})
            m2 = pl.DataFrame({"m": _s23["entity_id"], "ci": np.arange(_s23.height, dtype=np.int64)})
            n1 = _s1.height
            del _s1, _s23
            ek = read_ground_truth(dd).join(m1, on="s1").join(m2, on="m") \
                .select((pl.col("qi") * 16777216 + pl.col("ci")).alias("k"))
            del m1, m2
            N = ek.height

            def keys(path, rk, k):
                return pl.read_parquet(path, columns=["qi", "ci", rk]).filter(pl.col(rk) <= k).select(
                    (pl.col("qi").cast(pl.Int64) * 16777216 + pl.col("ci").cast(pl.Int64)).alias("k"))
            for k in [5, 10, 20, 30]:
                fk = keys(f"{a.work}/train_dense.parquet", "dense_rank", k)
                log(f"dense recall@{k}", round(ek.join(fk, on="k").height / N, 4))
            if os.path.exists(f"{a.work}/train_dense_rev.parquet"):
                fw = keys(f"{a.work}/train_dense.parquet", "dense_rank", 20)
                for k in [1, 2, 3, 5]:
                    rk = keys(f"{a.work}/train_dense_rev.parquet", "rdense_rank", k)
                    un = pl.concat([fw, rk]).unique()
                    log(f"reverse recall@{k}", round(ek.join(rk, on="k").height / N, 4),
                        f"| forward@20 + reverse@{k}:", round(ek.join(un, on="k").height / N, 4),
                        "pairs/S1", round(un.height / n1, 1))
                    del rk, un
        except Exception as ex:   # reporting only
            log("recall report skipped:", repr(ex))

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
    params = dict(XGB_PARAMS)
    if not os.path.exists("/dev/nvidia0"):      # no GPU in this session: same model on CPU
        params["device"] = "cpu"
    bst = xgb.train(params, dtr, num_boost_round=a.rounds, evals=[(dva, "val")],
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
        for thr in list(np.arange(0.2, 0.96, 0.05)) + [0.97, 0.98, 0.99]:
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


def fit_decoy_odds(C, edges):
    """Learn extra/missing word log-odds on a held-out slice C of training S1 (not used to train either
    stage), save them next to the model, and switch the decoy features on."""
    import ber.features as bf
    from ber.pipeline import ROUTES
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "name_core"])
    s23 = pl.read_parquet(f"{a.work}/train_s23n.parquet", columns=["entity_id", "name_core"])
    keep = pl.Series(np.where(C)[0]).cast(pl.Int32).implode()
    parts = []
    for f, rk, cap in [("cand", "tok_rank", CAPS["tok"]), ("dense", "dense_rank", CAPS["dense"]), ("dense_rev", "rdense_rank", CAPS["rdense"])]:
        path = f"{a.work}/train_{f}.parquet"
        if os.path.exists(path) and cap:
            parts.append(pl.scan_parquet(path).filter(pl.col("qi").is_in(keep) & (pl.col(rk) <= cap)).select("qi", "ci").collect())
    pr = pl.concat(parts).unique()
    pr = pr.with_columns(s1["entity_id"].gather(pr["qi"]).alias("s1"), s23["entity_id"].gather(pr["ci"]).alias("m"),
                         s1["name_core"].gather(pr["qi"]).alias("na"), s23["name_core"].gather(pr["ci"]).alias("nb"))
    pr = pr.join(edges.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    odds = bf.fit_token_odds(pr["na"].to_list(), pr["nb"].to_list(), pr["y"].to_numpy(), jobs=a.jobs)
    odds = {k: bf.add_french_equivalents(t) for k, t in odds.items()}
    for k, t in odds.items():
        t.write_parquet(f"{a.work}/tokodds_{k}.parquet")
    bf.TOK_ODDS = odds
    log("decoy word odds fitted on", pr.height, "pairs:", {k: v.height for k, v in odds.items()},
        "most decoy-like extra words:", odds["extra"].sort("score").head(12)["token"].to_list())


def load_decoy_odds():
    import ber.features as bf
    bf.TOK_ODDS = {k: pl.read_parquet(f"{a.work}/tokodds_{k}.parquet") for k in ("extra", "missing")}
    for path in (a.odds_extra.split(",") if a.odds_extra else []):
        t = bf.TOK_ODDS["extra"]
        add = pl.read_parquet(path).select("token", pl.col("score").cast(pl.Float32))
        add = add.filter(~pl.col("token").is_in(t["token"].implode()))       # the training odds take precedence
        bf.TOK_ODDS["extra"] = pl.concat([t.select("token", pl.col("score").cast(pl.Float32)), add])
        log("extra word odds from", path, "+", add.height, "words:", add.sort("score").head(12)["token"].to_list())


def s2_filter(ctx):
    """Stage 1 acts as a learned filter: only its top candidates per S1 go to stage 2 (and into
    candidate_pairs.tsv)."""
    m = pl.lit(True)
    if a.s2_topk:
        m = m & (pl.col("p1_rank_q") <= a.s2_topk)
    if a.s2_minp:
        m = m & (pl.col("p1") >= a.s2_minp)
    return ctx.filter(m)


def collect(split, cand, mask, lab, ctx=None, cols=None, seed=0):
    """Featurise the S1 rows in `mask`; return train/val chunk lists (with easy-negative sampling on
    the training rows only), the validation (s1, m, y[, p1]) frame and the feature columns."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, wtr, Xva, yva, vas = [], [], [], [], [], []
    for t in iter_pair_tables(split, a.work, cand, s1_filter=mask, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
        if ctx is not None:
            sub = ctx.filter(pl.col("qi").is_between(t["qi"].min(), t["qi"].max()))
            t = t.join(sub, on=["qi", "ci"], how="inner")      # rows filtered out by stage 1 are dropped
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
        b = xgb.Booster(); b.load_model(f"{a.work}/{tag}.json")
        if not os.path.exists("/dev/nvidia0"):     # trained on GPU; predict on CPU when there is none
            b.set_param({"device": "cpu"})
        return b
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
    drop = (s1.select((pl.col("entity_id").hash(17) % 1000).alias("d"))["d"].to_numpy() < a.drop_s1 * 1000)
    if a.drop_s1 > 0:
        log("dropping", int(drop.sum()), "S1 to simulate test density")
    ids_all = s1.rename({"entity_id": "s1"}).filter(pl.Series(~drop))
    cmask = np.ones(len(h), bool)
    if a.train_countries:
        cmask = s1["country"].is_in(a.train_countries.split(",")).to_numpy()
        log("training countries:", a.train_countries, "S1:", int(cmask.sum()))
    if a.decoy_feats:
        fit_decoy_odds((h >= 900) & ~drop & cmask, edges)        # held-out 10% slice, never used by A / B / use
    dupm = None
    if a.dup_decoys:
        s23i = pl.read_parquet(f"{a.work}/train_s23n.parquet", columns=["entity_id"])["entity_id"]
        dupm = (~s23i.is_in(edges["m"].implode())).to_numpy()
        log("decoy records (belong to no S1):", int(dupm.sum()))
    h = np.where(drop, 10_000, h)      # dropped S1 never enter A / B / use
    if not a.stage2:
        use = h < a.train_frac * 1000
        cand = load_candidates("train", a.work, CAPS, keep_qi=use, drop_qi=drop if a.drop_s1 > 0 else None, dup_mask=dupm, dup_near=a.dup_near)
        Xtr, ytr, wtr, Xva, yva, vfr, cols = collect("train", cand, use, lab)
        del cand
        bst, best_iter = fit_model(Xtr, ytr, wtr, Xva, yva, cols, "model")
        vas = [vfr]
        pv = np.concatenate([predict(bst, x, best_iter) for x in Xva])
        va = pl.concat(vas).with_columns(pl.Series("p", pv))
        del Xva, yva
        Xtr = ytr = wtr = None     # free training matrices before the test stage
        import gc; gc.collect()
        ids = ids_all.join(s1.rename({"entity_id": "s1"}).filter(pl.Series(use)).select("s1"), on="s1").filter(is_val("s1"))
        cfg = {"cols": cols, "best_iter": best_iter, "stage2": False, "model": a.model}
    else:
        A = h < a.frac_a * 1000
        B = (h >= a.frac_a * 1000) & (h < (a.frac_a + a.frac_b) * 1000)
        if a.train_countries:     # training rows only from the training countries; validation S1 of every country
            isv = s1.select(is_val("entity_id"))["entity_id"].to_numpy()
            A = A & cmask
            B = B & (cmask | isv)
        XV = None
        if a.swap_ab:
            XV = B & s1.select(is_val("entity_id"))["entity_id"].to_numpy()     # the usual model's validation S1
            A, B = B, A
        cand = load_candidates("train", a.work, CAPS, drop_qi=drop if a.drop_s1 > 0 else None, dup_mask=dupm, dup_near=a.dup_near)
        # stage 1 on A
        Xtr, ytr, wtr, Xva, yva, _, cols = collect("train", cand, A, lab)
        b1, it1 = fit_model(Xtr, ytr, wtr, Xva, yva, cols, "model1")
        Xtr = ytr = wtr = Xva = yva = None; import gc; gc.collect()
        # stage-1 scores for EVERY train pair (A rows are in-sample; they only act as competitors)
        ctx = score_context(score_all("train", cand, b1, cols, it1))
        ctx_x = None
        if XV is not None:     # stage-1 slice S1 held out of stage-1 training (early stopping only)
            ctx_x = s2_filter(ctx.filter(pl.col("qi").is_in(pl.Series(np.where(XV)[0]).cast(pl.Int32).implode())))
        ctx_p = None
        if a.self_train and a.train_countries:     # unlabelled pool: other countries' S1 outside validation
            isv = s1.select(is_val("entity_id"))["entity_id"].to_numpy()
            P = (h < (a.frac_a + a.frac_b) * 1000) & ~cmask & ~isv
            ctx_p = s2_filter(ctx.filter(pl.col("qi").is_in(pl.Series(np.where(P)[0]).cast(pl.Int32).implode())))
            log("self-training pool S1:", int(P.sum()))
        ctx = ctx.filter(pl.col("qi").is_in(pl.Series(np.where(B)[0]).cast(pl.Int32).implode()))
        if a.triangle:
            ctx = add_triangle("train", a.work, ctx, jobs=a.jobs)
        ctx = s2_filter(ctx)
        gc.collect()
        # stage 2 on B
        Xtr, ytr, wtr, Xva, yva, tbv, cols2 = collect("train", cand, B, lab, ctx=ctx, seed=1)
        del ctx
        Xes, yes = Xva, yva
        if a.train_countries:      # early stopping on the training countries' validation rows only
            tr_s1 = s1.filter(pl.Series(cmask))["entity_id"].implode()
            m_tr = tbv["s1"].is_in(tr_s1).to_numpy()
            Xes, yes = [np.concatenate(Xva)[m_tr]], [np.concatenate(yva)[m_tr]]
        bst, it2 = fit_model(Xtr, ytr, wtr, Xes, yes, cols2, "model")
        if ctx_p is not None:
            tbv.select("s1", "m", "y").with_columns(pl.Series("p", np.concatenate([predict(bst, x, it2) for x in Xva]))) \
               .write_parquet(f"{a.work}/val_scores_base.parquet")
            ids_p, Xp, pp = [], [], []
            for t in iter_pair_tables("train", a.work, cand, s1_filter=P, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
                sub = ctx_p.filter(pl.col("qi").is_between(t["qi"].min(), t["qi"].max()))
                t = t.join(sub, on=["qi", "ci"], how="inner")
                X = t.select(cols2).to_numpy().astype(np.float32); pr = predict(bst, X, it2)
                keep = (pr >= a.st_hi) | (pr <= a.st_lo)
                ids_p.append(t.select("s1", "m").filter(pl.Series(keep))); Xp.append(X[keep]); pp.append(pr[keep])
                del t, X
            ids_p = pl.concat(ids_p); pp = np.concatenate(pp); yp = (pp >= a.st_hi).astype(np.float32)
            chk = ids_p.with_columns(pl.Series("yp", yp)).join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
            log("pseudo-labels:", len(yp), "positive", int(yp.sum()), "| accuracy vs the hidden labels:",
                round(float((chk["yp"] == chk["y"]).mean()), 4), "| positives precision:",
                round(float(chk.filter(pl.col("yp") == 1)["y"].mean()), 4))
            Xp = np.concatenate(Xp)
            bst, it2 = fit_model(Xtr + [Xp], ytr + [yp], wtr + [np.full(len(yp), a.st_weight, np.float32)], Xes, yes, cols2, "model")
            del Xp, ids_p, chk
            log("stage 2 refit with pseudo-labels, best_iter", it2)
        if ctx_x is not None:
            _, _, _, Xx, _, tbx, _ = collect("train", cand, XV, lab, ctx=ctx_x, cols=cols2, seed=2)
            tbx.select("s1", "m", "y").with_columns(pl.Series("p", np.concatenate([predict(bst, x, it2) for x in Xx]))) \
               .write_parquet(f"{a.work}/val_scores_other.parquet")
            log("scored the usual model's validation S1 ->", "val_scores_other.parquet", tbx.height, "pairs")
            del ctx_x, Xx, tbx
        del cand
        va = tbv.select("s1", "m", "y").with_columns(
            pl.Series("p", np.concatenate([predict(bst, x, it2) for x in Xva])))
        va1 = tbv.select("s1", "m", "y", pl.col("p1").alias("p"))
        Xtr = ytr = wtr = Xva = yva = None; gc.collect()
        ids = ids_all.join(s1.rename({"entity_id": "s1"}).filter(pl.Series(B)).select("s1"), on="s1").filter(is_val("s1"))
        truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
        log("---- stage-1 only on stage-2 validation rows ----")
        tune_decoder(va1, ids, truth)
        cfg = {"cols": cols2, "cols1": cols, "best_iter": it2, "best_iter1": it1, "stage2": True, "model": a.model}
    va.write_parquet(f"{a.work}/val_scores.parquet")
    truth = edges.filter(pl.col("s1").is_in(ids["s1"].implode()))
    log("---- final model ----")
    best = tune_decoder(va, ids, truth)
    cfg.update({"thr": best[1][0], "excl": best[1][1], "val_f05": best[0], "caps": CAPS,
                "s2_topk": a.s2_topk, "s2_minp": a.s2_minp, "triangle": a.triangle, "decoy_feats": a.decoy_feats,
                "dup_decoys": a.dup_decoys,
                "global_sims": a.global_sims})
    json.dump(cfg, open(f"{a.work}/cfg.json", "w"))

if "dump" in stages:
    # feature matrices for offline model comparison (e.g. CatBoost vs XGBoost on Colab)
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "country"])
    h = s1.select((pl.col("entity_id").hash(11) % 1000).alias("h"))["h"].to_numpy()
    drop = (s1.select((pl.col("entity_id").hash(17) % 1000).alias("d"))["d"].to_numpy() < a.drop_s1 * 1000)
    use = (h < a.train_frac * 1000) & ~drop
    edges = read_ground_truth(dd)
    lab = edges.with_columns(pl.lit(1, pl.Int8).alias("y"))
    cand = load_candidates("train", a.work, CAPS, keep_qi=use, drop_qi=drop if a.drop_s1 > 0 else None)
    Xtr, ytr, wtr, Xva, yva, vfr, cols = collect("train", cand, use, lab)
    del cand
    D = f"{a.work}/dump"; os.makedirs(D, exist_ok=True)
    pl.DataFrame(np.concatenate(Xtr), schema=cols).with_columns(pl.Series("y", np.concatenate(ytr)),
        pl.Series("w", np.concatenate(wtr))).write_parquet(f"{D}/train.parquet")
    pl.DataFrame(np.concatenate(Xva), schema=cols).with_columns(pl.Series("y", np.concatenate(yva)),
        vfr["s1"], vfr["m"]).write_parquet(f"{D}/val.parquet")
    ids = s1.rename({"entity_id": "s1"}).filter(pl.Series(use)).filter(is_val("s1"))
    ids.write_parquet(f"{D}/val_ids.parquet")
    edges.filter(pl.col("s1").is_in(ids["s1"].implode())).write_parquet(f"{D}/val_truth.parquet")
    log("dumped", D, os.listdir(D))

if "selftrain" in stages:
    # domain adaptation for test countries WITHOUT training labels (France): refit stage 2 on its usual training
    # rows plus the confident pseudo-labels of those countries' test pairs, then re-score those countries only
    import gc
    cfg = json.load(open(f"{a.work}/cfg.json"))
    a.model = cfg.get("model", "lgb")
    _bp.GLOBAL_SIMS = cfg.get("global_sims", a.global_sims) or a.global_sims
    a.s2_topk, a.s2_minp = cfg.get("s2_topk", 0), cfg.get("s2_minp", 0.0)
    caps = cfg.get("caps", CAPS)
    if cfg.get("decoy_feats"):
        load_decoy_odds()
    b1 = load_model("model1", a.model)
    s1 = pl.read_parquet(f"{a.work}/train_s1n.parquet", columns=["entity_id", "country"])
    h = s1.select((pl.col("entity_id").hash(11) % 1000).alias("h"))["h"].to_numpy()
    B = (h >= a.frac_a * 1000) & (h < (a.frac_a + a.frac_b) * 1000)
    lab = read_ground_truth(dd).with_columns(pl.lit(1, pl.Int8).alias("y"))
    cand = load_candidates("train", a.work, caps)          # 1) the stage-2 training rows, as in training
    ctx = score_context(score_all("train", cand, b1, cfg["cols1"], cfg["best_iter1"]))
    ctx = s2_filter(ctx.filter(pl.col("qi").is_in(pl.Series(np.where(B)[0]).cast(pl.Int32).implode())))
    Xtr, ytr, wtr, Xva, yva, tbv, cols2 = collect("train", cand, B, lab, ctx=ctx, cols=cfg["cols"], seed=1)
    del cand, ctx; gc.collect()
    tec = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"])
    new_c = sorted(set(tec["country"].unique().to_list()) - set(s1["country"].unique().to_list()))
    U = tec["country"].is_in(new_c).to_numpy()
    log("self-training on", new_c, "test S1:", int(U.sum()))
    cand = load_candidates("test", a.work, caps)           # 2) their test pairs (candidates never cross countries)
    ctx = s2_filter(score_context(score_all("test", cand, b1, cfg["cols1"], cfg["best_iter1"], mask=U)))
    ids_u, Xu = [], []
    for t in iter_pair_tables("test", a.work, cand, s1_filter=U, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
        t = t.join(ctx.filter(pl.col("qi").is_between(t["qi"].min(), t["qi"].max())), on=["qi", "ci"], how="inner")
        ids_u.append(t.select("s1", "m")); Xu.append(t.select(cols2).to_numpy().astype(np.float32))
        del t
    del cand, ctx; gc.collect()
    ids_u = pl.concat(ids_u); Xu = np.concatenate(Xu)
    import glob as _g
    src = a.pseudo_scores if os.path.exists(a.pseudo_scores) else _g.glob(a.pseudo_scores, recursive=True)[0]
    ps = ids_u.join(pl.read_parquet(src, columns=["s1", "m", "p"]), on=["s1", "m"], how="left")["p"] \
        .fill_null(0.5).to_numpy()
    conf = (ps >= a.st_hi) | (ps <= a.st_lo)
    yp = (ps[conf] >= a.st_hi).astype(np.float32)
    log("pseudo-labels:", int(conf.sum()), "of", len(ps), "pairs, positive", int(yp.sum()))
    bst, it2 = fit_model(Xtr + [Xu[conf]], ytr + [yp], wtr + [np.full(len(yp), a.st_weight, np.float32)],
                         Xva, yva, cols2, "model_st")
    va = tbv.select("s1", "m", "y").with_columns(pl.Series("p", np.concatenate([predict(bst, x, it2) for x in Xva])))
    va.write_parquet(f"{a.work}/val_scores_st.parquet")
    ids = s1.rename({"entity_id": "s1"}).filter(pl.Series(B)).filter(is_val("s1"))
    log("labelled-country validation after self-training:")
    tune_decoder(va, ids, lab.select("s1", "m").filter(pl.col("s1").is_in(ids["s1"].implode())))
    ids_u.with_columns(pl.Series("p", predict(bst, Xu, it2)).cast(pl.Float32)).write_parquet(f"{a.work}/test_scores_st.parquet")
    log("wrote test_scores_st.parquet for", new_c, len(ps), "pairs")

if "test" in stages:
    cfg = json.load(open(f"{a.work}/cfg.json"))
    a.model = cfg.get("model", "lgb")
    _bp.GLOBAL_SIMS = cfg.get("global_sims", a.global_sims) or a.global_sims   # must match training
    if cfg.get("decoy_feats"):
        load_decoy_odds()
    bst = load_model("model", a.model)
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
        a.s2_topk, a.s2_minp = cfg.get("s2_topk", 0), cfg.get("s2_minp", 0.0)
        if a.triangle:
            ctx = add_triangle("train", a.work, ctx, jobs=a.jobs)
        if cfg.get("triangle"):
            ctx = add_triangle("test", a.work, ctx, jobs=a.jobs)
        ctx = s2_filter(ctx).sort("qi")
        qs = ctx["qi"].to_numpy()
        for t in iter_pair_tables("test", a.work, cand, n_jobs=a.jobs, chunk_s1=a.chunk_s1):
            lo, hi = np.searchsorted(qs, [t["qi"].min(), t["qi"].max() + 1])     # this chunk's slice only
            t = t.join(ctx.slice(lo, hi - lo), on=["qi", "ci"], how="inner")
            p = predict(bst, t.select(cfg["cols"]).to_numpy().astype(np.float32), cfg["best_iter"])
            parts.append(t.select("qi", "ci").with_columns(pl.Series("p", p).cast(pl.Float32)))
            del t, p; gc.collect()
        del cand, ctx
    tab = pl.concat(parts); del parts
    s1_all = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"]
    s23_all = pl.read_parquet(f"{a.work}/test_s23n.parquet", columns=["entity_id"])["entity_id"]
    tab = tab.with_columns(s1_all.gather(tab["qi"]).alias("s1"), s23_all.gather(tab["ci"]).alias("m")).drop("qi", "ci")
    if os.path.islink(f"{a.work}/test_scores.parquet"):     # never write through a reused link
        os.unlink(f"{a.work}/test_scores.parquet")
    tab.write_parquet(f"{a.work}/test_scores.parquet")
    if a.prior_thr and os.path.exists(f"{a.work}/val_scores.parquet"):
        from ber.pipeline import prior_thresholds, decode_by_country
        from ber.io import read_source      # raw source: test-only runs don't mount the train caches
        trc = read_source(os.path.join(dd, "train", "train_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
        tec = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
        thr_c, est = prior_thresholds(pl.read_parquet(f"{a.work}/val_scores.parquet"), tab, trc, tec)
        log("prior-corrected per-country thresholds:", thr_c)
        log(est.select("country", "band", pl.col("prec").round(3)).pivot(on="country", index="band", values="prec").sort("band"))
        pred = decode_by_country(tab, thr_c, tec)
    else:
        pred = decode_f05(tab, cfg["excl"]) if cfg["thr"] == "f05" else decode(tab, cfg["thr"], cfg["excl"])
    s1_ids = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_id_lists(f"{out}/candidate_pairs.tsv", s1_ids, tab.select("s1", "m"), "candidate_entity_ids")
    write_id_lists(f"{out}/matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    log("test written; S1 with >=1 match:", pred["s1"].n_unique(), "/", len(s1_ids), "pairs", pred.height)
    ctry = pl.read_parquet(f"{a.work}/test_s1n.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    log(pred.join(ctry, on="s1").group_by("country").agg(pl.len(), pl.col("s1").n_unique()))
