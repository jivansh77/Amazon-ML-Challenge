"""rt-score (US/India recall routes, STEPS 4-6).

4  Score the kept-route candidates (from kavyachetwani/rt-miss) for clean val and test: string features + the
   ber-ce-base2 cross-encoder (fp16). Fit a small XGBoost on clean val (2 folds by S1, out-of-fold) predicting a
   match for the route-only candidates.
5  GATE on clean val (reproduce the proxy blend first: dec p + base2 CE, logit w 0.6, thr 0.75 = 0.98942):
   add route candidates with prob >= t (unclaimed records only, exclusivity among the adds), t tuned on fold A,
   checked on fold B; pass = >= +0.0002 on both folds, singletons not worse.
6  If it passes: route_adds_test.parquet (s1, m, prob) and a delta-edited copy of the submitted file
   (jvmusic/ber-france-v2 avg_ce4_v2h + route adds; exclusivity; rows of test countries without training labels
   byte-identical; validator --check-ids). Nothing is submitted.
Country is used only to leave the unlabelled test country (France) untouched; never as a model feature.
"""
import glob, hashlib, json, os, subprocess, sys, time, traceback

import numpy as np
import polars as pl

W = "/kaggle/working"
T0 = time.time()
M = {}
LOGF = open(f"{W}/progress.log", "a")
ROUTES = os.environ.get("RT_ROUTES", "R3").split(",")
W_BLEND, THR_BASE, BAND = 0.6, 0.75, (0.02, 0.998)
EPS = 1e-6


def log(*a):
    """Timestamped log line to stdout and progress.log."""
    s = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def save_metrics():
    """Write metrics.json (after every step)."""
    M["runtime_min"] = round((time.time() - T0) / 60, 1)
    json.dump(M, open(f"{W}/metrics.json", "w"), indent=1, default=str)


def find(name, sub=""):
    """First path under /kaggle/input ending with `name` and containing `sub`."""
    hits = sorted(h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if sub in h)
    if not hits:
        raise FileNotFoundError(f"{name} ({sub})")
    return hits[0]


def read_source(path):
    """Challenge TSV (quoting disabled, as in ber.io)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.col("business_address").fill_null(""), pl.col("business_name").fill_null(""))


def read_lists(path, col):
    """source1_entity_id \\t comma list -> exploded (s1, m)."""
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (d.select(pl.col("source1_entity_id").alias("s1"), pl.col(col).fill_null("").str.split(",").alias("m"))
            .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))


def macro_f05(pred, truth, ids, beta=0.5):
    """Challenge macro F0.5 over `ids` (copy of ber.metric.macro_f05): overall, singletons, matched."""
    b2 = beta * beta
    p = pred.select("s1", "m").unique().with_columns(pl.lit(1).alias("_p"))
    t = truth.select("s1", "m").unique().with_columns(pl.lit(1).alias("_t"))
    j = p.join(t, on=["s1", "m"], how="full", coalesce=True)
    g = j.group_by("s1").agg(pl.col("_p").sum().alias("np"), pl.col("_t").sum().alias("nt"),
                             (pl.col("_p").is_not_null() & pl.col("_t").is_not_null()).sum().alias("tp"))
    d = ids.select("s1").join(g, on="s1", how="left").fill_null(0)
    d = d.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                       .otherwise((1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * (pl.col("nt") - pl.col("tp"))
                                                             + (pl.col("np") - pl.col("tp")))).alias("f"))
    return {"f05": d["f"].mean(), "sing": d.filter(pl.col("nt") == 0)["f"].mean(), "matched": d.filter(pl.col("nt") > 0)["f"].mean(),
            "n": d.height}


def decode(pairs, thr):
    """Exclusivity (each record keeps its best S1), then threshold (as ber.pipeline.decode)."""
    return pairs.filter(pl.col("p") == pl.col("p").max().over("m")).filter(pl.col("p") >= thr).select("s1", "m")


def L(c):
    """Logit of a probability column."""
    x = pl.col(c).clip(EPS, 1 - EPS)
    return (x / (1 - x)).log()


def fold_of(s):
    """Stable 2-fold assignment by S1 id (md5)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


# ------------------------------------------------------------------ baseline
def baseline():
    """Clean val (val S1 minus every S1 the CEs trained on), blended p, decoded prediction, truth."""
    va = pl.read_parquet(find("val_scores.parquet", "ber-dec-train"), columns=["s1", "m", "y", "p"])
    ce = pl.read_parquet(find("ce_val.parquet", "ber-ce-base2"), columns=["s1", "m", "ce"]).unique(["s1", "m"])
    va = va.join(ce, on=["s1", "m"], how="left")
    excl = pl.concat([pl.read_parquet(find("train.parquet", d), columns=["s1"]) for d in ("ber-ce-data/", "ber-ce-data2/")]).unique()
    ids = va.select("s1").unique().join(excl, on="s1", how="anti")
    va = va.join(ids, on="s1")
    va = va.with_columns(pl.when(pl.col("ce").is_null()).then(pl.col("p"))
                         .otherwise(1 / (1 + (-(W_BLEND * L("p") + (1 - W_BLEND) * L("ce"))).exp())).alias("p"))
    gt = read_lists(find("train_ground_truth.tsv"), "matched_entity_ids")
    truth = gt.join(ids, on="s1")
    pred = decode(va, THR_BASE)
    r = macro_f05(pred, truth, ids)
    log("clean val S1", ids.height, "pairs", va.height, "baseline", r)
    M["baseline"] = r; M["clean_s1"] = ids.height
    return va, ids, truth, pred


# ------------------------------------------------------------------ features
def texts(split, ids_s1, ids_m):
    """CE text 'business_name | business_address' for the needed ids of a split."""
    dd = os.path.dirname(find(f"{split}_source1.tsv"))
    s1 = read_source(f"{dd}/{split}_source1.tsv").join(ids_s1, left_on="entity_id", right_on="s1")
    s23 = pl.concat([read_source(f"{dd}/{split}_source{k}.tsv") for k in (2, 3)]).join(ids_m, left_on="entity_id", right_on="m")
    f = lambda d: d.select("entity_id", (pl.col("business_name") + " | " + pl.col("business_address")).alias("t"))
    return f(s1), f(s23)


def ce_score(a, b, bs=256):
    """ber-ce-base2 probabilities for text pairs, fp16, length-sorted batches."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    md = os.path.dirname(find("model.safetensors", "ber-ce-base2"))
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md, num_labels=1).cuda().eval().half()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.zeros(len(a), np.float32)
    t = time.time()
    with torch.no_grad():
        for s in range(0, len(a), bs):
            ix = order[s:s + bs]
            enc = tok([a[i] for i in ix], [b[i] for i in ix], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
            out[ix] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if s % (bs * 400) == 0:
                log(f"  CE {s}/{len(a)} {s / max(time.time() - t, 1):.0f} pairs/s")
    return out


def features(split, rt):
    """Route candidates (s1, m) of the kept routes with string features and the CE score."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    rt = rt.filter(pl.col("route").is_in(ROUTES))
    g = rt.group_by("s1", "m", "qi", "ci").agg(pl.col("cos").max(), pl.col("route").unique().alias("routes"))
    for r in ROUTES:
        g = g.with_columns(pl.col("routes").list.contains(r).cast(pl.Int8).alias(f"r_{r}"))
    g = g.with_columns(pl.col("routes").list.len().alias("n_routes")).drop("routes")
    blk = f"ber-block-{split}"
    s1n = pl.read_parquet(find(f"{split}_s1n.parquet", blk), columns=["name_core", "name_legal", "addr_clean", "addr_nums", "addr_words"]).with_row_index("qi")
    s23n = pl.read_parquet(find(f"{split}_s23n.parquet", blk), columns=["name_core", "name_legal", "addr_clean", "addr_nums", "addr_words"]).with_row_index("ci")
    g = (g.with_columns(pl.col("qi").cast(pl.UInt32), pl.col("ci").cast(pl.UInt32)).join(s1n, on="qi")
         .join(s23n.rename({c: c + "_b" for c in s23n.columns if c != "ci"}), on="ci"))
    del s1n, s23n
    A = lambda c: g[c].fill_null("").to_list()
    for nm, sc in [("n_ratio", fuzz.ratio), ("n_tset", fuzz.token_set_ratio), ("n_tsort", fuzz.token_sort_ratio), ("n_partial", fuzz.partial_ratio)]:
        g = g.with_columns(pl.Series(nm, process.cpdist(A("name_core"), A("name_core_b"), scorer=sc, workers=-1)))
    g = g.with_columns(pl.Series("n_jw", process.cpdist(A("name_core"), A("name_core_b"), scorer=JaroWinkler.normalized_similarity, workers=-1)))
    for nm, sc in [("a_tset", fuzz.token_set_ratio), ("a_ratio", fuzz.ratio)]:
        g = g.with_columns(pl.Series(nm, process.cpdist(A("addr_clean"), A("addr_clean_b"), scorer=sc, workers=-1)))
    sp = lambda c: pl.col(c).fill_null("").str.split(" ")
    g = g.with_columns(
        (sp("addr_words").list.set_intersection(sp("addr_words_b")).list.len() / sp("addr_words").list.set_union(sp("addr_words_b")).list.len().clip(1, None)).alias("a_jac"),
        (sp("name_core").list.set_intersection(sp("name_core_b")).list.len() / sp("name_core").list.set_union(sp("name_core_b")).list.len().clip(1, None)).alias("n_jac"),
        ((sp("addr_nums").list.first() == sp("addr_nums_b").list.first()) & (pl.col("addr_nums_b").fill_null("") != "")).cast(pl.Int8).alias("hn_eq"),
        (pl.col("addr_clean_b").fill_null("") == "").cast(pl.Int8).alias("b_empty"),
        (pl.col("addr_clean").fill_null("") == "").cast(pl.Int8).alias("a_empty"),
        (pl.col("name_legal").fill_null("") == pl.col("name_legal_b").fill_null("")).cast(pl.Int8).alias("legal_eq"),
        (pl.col("name_core").fill_null("").str.replace_all(" ", "") == pl.col("name_core_b").fill_null("").str.replace_all(" ", "")).cast(pl.Int8).alias("cmp_eq"),
        sp("name_core").list.len().alias("n_len_a"), sp("name_core_b").list.len().alias("n_len_b"))
    g = g.drop([c for c in g.columns if c.endswith("_b") and c not in ("n_len_b",)] + ["name_core", "name_legal", "addr_clean", "addr_nums", "addr_words"])
    ta, tb = texts(split, g.select("s1").unique(), g.select("m").unique())
    g = g.join(ta.rename({"entity_id": "s1", "t": "ta"}), on="s1").join(tb.rename({"entity_id": "m", "t": "tb"}), on="m")
    log(split, "route candidates to score", g.height)
    g = g.with_columns(pl.Series("ce", ce_score(g["ta"].to_list(), g["tb"].to_list()))).drop("ta", "tb")
    return g


FEATS = None


def fit_predict(tr, te):
    """XGBoost (depth 4, 400 rounds, eta 0.05) on tr -> probabilities for te."""
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="logloss", n_jobs=4)
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
    return m.predict_proba(te.select(FEATS).to_numpy())[:, 1], m


def with_adds(base_pred, cand, t):
    """Baseline prediction + candidates with prob >= t on unclaimed records (each record: best S1 only)."""
    add = cand.filter((pl.col("prob") >= t) & (pl.col("claimed") == 0))
    add = add.filter(pl.col("prob") == pl.col("prob").max().over("m")).unique("m", keep="first")
    return pl.concat([base_pred.select("s1", "m"), add.select("s1", "m")]), add.height


def gate(va_cand, base_pred, truth, ids):
    """Tune t on one fold, check on the other (both directions reported); pass rule on (A -> B)."""
    grid = [round(x, 2) for x in np.arange(0.30, 0.99, 0.05)] + [0.97, 0.99]
    res = {}
    fids = {f: ids.filter(pl.col("fold") == f) for f in (0, 1)}
    base = {f: macro_f05(base_pred.join(fids[f], on="s1"), truth.join(fids[f], on="s1"), fids[f]) for f in (0, 1)}
    table = {}
    for f in (0, 1):
        c = va_cand.filter(pl.col("fold") == f)
        for t in grid:
            p, n = with_adds(base_pred.join(fids[f], on="s1"), c, t)
            r = macro_f05(p, truth.join(fids[f], on="s1"), fids[f])
            table[(f, t)] = {"d": r["f05"] - base[f]["f05"], "d_sing": r["sing"] - base[f]["sing"], "adds": n}
    for tune, chk in ((0, 1), (1, 0)):
        t = max(grid, key=lambda x: table[(tune, x)]["d"])
        res[f"tune{tune}_check{chk}"] = {"t": t, "tune": table[(tune, t)], "check": table[(chk, t)]}
    M["gate_table"] = {f"fold{f}_t{t}": v for (f, t), v in table.items()}
    M["gate"] = res; M["gate_base"] = base
    a = res["tune0_check1"]
    ok = a["tune"]["d"] >= 2e-4 and a["check"]["d"] >= 2e-4 and a["tune"]["d_sing"] >= 0 and a["check"]["d_sing"] >= 0
    M["gate_pass"] = bool(ok)
    log("GATE", json.dumps(res), "PASS" if ok else "FAIL")
    return ok, a["t"]


def main():
    global FEATS
    subprocess.run("pip install -q 'rapidfuzz>=3.6' 2>&1 | tail -1", shell=True)
    M["routes"] = ROUTES
    va, ids, truth, base_pred = baseline()
    save_metrics()
    # ---- clean val route candidates
    rv = pl.read_parquet(find("route_val.parquet", "rt-miss")).join(ids, on="s1")
    fv = features("train", rv).join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    fv = fv.join(va.select("s1", "m"), on=["s1", "m"], how="anti")          # route-only: never already scored
    nacc = base_pred.group_by("s1").len("n_acc")
    fv = fv.join(nacc, on="s1", how="left").with_columns(pl.col("n_acc").fill_null(0),
                                                         pl.col("m").is_in(base_pred["m"].implode()).cast(pl.Int8).alias("claimed"))
    FEATS = [c for c in fv.columns if c not in ("s1", "m", "qi", "ci", "y", "fold", "prob")]
    M["feats"] = FEATS
    ids = ids.with_columns(pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
    fv = fv.join(ids, on="s1")
    log("val route candidates", fv.height, "positives", int(fv["y"].sum()), "by route",
        {r: [int(fv.filter(pl.col(f"r_{r}") == 1).height), int(fv.filter(pl.col(f"r_{r}") == 1)["y"].sum())] for r in ROUTES})
    M["val_cands"] = fv.height; M["val_pos"] = int(fv["y"].sum())
    from sklearn.metrics import roc_auc_score
    oof = np.zeros(fv.height, np.float32)
    for f in (0, 1):
        tr = fv.filter(pl.col("fold") != f); ix = np.where(fv["fold"].to_numpy() == f)[0]
        oof[ix], _ = fit_predict(tr, fv[ix])
    fv = fv.with_columns(pl.Series("prob", oof))
    M["oof_auc"] = roc_auc_score(fv["y"], fv["prob"]); M["ce_auc"] = roc_auc_score(fv["y"], fv["ce"])
    log("OOF AUC", M["oof_auc"], "CE alone", M["ce_auc"])
    fv.select("s1", "m", "y", "prob", "ce", "claimed", *[f"r_{r}" for r in ROUTES]).write_parquet(f"{W}/val_route_oof.parquet")
    save_metrics()
    ok, t = gate(fv, base_pred, truth, ids)
    save_metrics()
    # ---- test (always scored so chat 1 can use the probabilities; file edits only if the gate passes)
    tec = read_source(find("test_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
    trc = read_source(find("train_source1.tsv")).select("country").unique()
    lab = tec.join(trc, on="country")                                        # S1 of countries with training labels
    rt = pl.read_parquet(find("route_test.parquet", "rt-miss")).join(lab.select("s1"), on="s1")
    sub_m = read_lists(find("matching_results_avg_ce4_v2h.tsv"), "matched_entity_ids")
    sub_c = read_lists(find("candidate_pairs_avg_ce4_v2.tsv"), "candidate_entity_ids")
    rt = rt.join(sub_c, on=["s1", "m"], how="anti")
    ft = features("test", rt)
    ft = ft.join(sub_m.group_by("s1").len("n_acc"), on="s1", how="left").with_columns(
        pl.col("n_acc").fill_null(0), pl.col("m").is_in(sub_m["m"].implode()).cast(pl.Int8).alias("claimed"))
    _, model = fit_predict(fv, fv.head(1))
    ft = ft.with_columns(pl.Series("prob", model.predict_proba(ft.select(FEATS).to_numpy())[:, 1]))
    ft.select("s1", "m", "prob", "ce", "claimed", *[f"r_{r}" for r in ROUTES]).write_parquet(f"{W}/route_adds_test.parquet")
    M["test_cands"] = ft.height
    M["test_prob_bands"] = {str(b): int((ft["prob"] >= b).sum()) for b in (0.5, 0.7, 0.8, 0.9, 0.95)}
    log("test route candidates", ft.height, M["test_prob_bands"])
    save_metrics()
    if ok:
        build_files(ft, t, tec, lab)
    M["done"] = True
    save_metrics()


def build_files(ft, t, tec, lab):
    """Submitted file + route adds (prob >= t, unclaimed, labelled-country S1 only), rows of other S1 untouched."""
    pm, pc = find("matching_results_avg_ce4_v2h.tsv"), find("candidate_pairs_avg_ce4_v2.tsv")
    add = ft.filter((pl.col("prob") >= t) & (pl.col("claimed") == 0))
    add = add.filter(pl.col("prob") == pl.col("prob").max().over("m")).unique("m", keep="first").join(lab.select("s1"), on="s1")
    addg = dict(add.group_by("s1").agg(pl.col("m")).iter_rows())
    M["test_adds"] = add.height; M["test_add_s1"] = len(addg); M["t"] = t
    log("test adds", add.height, "S1", len(addg), "t", t)

    def edit(src, dst):
        """Append the added ids to the S1 rows (others byte-identical)."""
        n = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i == 0:
                    g.write(line); continue
                s1, lst = line.rstrip("\n").split("\t")
                if s1 in addg:
                    have = set(lst.split(",")) if lst else set()
                    new = [m for m in addg[s1] if m not in have]
                    if new:
                        lst = ",".join(([lst] if lst else []) + new); n += 1
                    line = f"{s1}\t{lst}\n"
                g.write(line)
        return n
    os.makedirs(f"{W}/output", exist_ok=True)
    om, oc = f"{W}/output/matching_results_rt.tsv", f"{W}/output/candidate_pairs_rt.tsv"
    M["rows_changed_matching"] = edit(pm, om); M["rows_changed_candidates"] = edit(pc, oc)
    # rows of S1 without training labels (France) must be byte-identical
    unl = set(tec.join(lab.select("s1"), on="s1", how="anti")["s1"].to_list())
    for a, b, nm in ((pm, om, "matching"), (pc, oc, "candidates")):
        la = [l for l in open(a, encoding="utf-8") if l.split("\t", 1)[0] in unl]
        lb = [l for l in open(b, encoding="utf-8") if l.split("\t", 1)[0] in unl]
        M[f"unlabelled_identical_{nm}"] = la == lb and len(la) == len(unl)
    log("unlabelled rows identical", M["unlabelled_identical_matching"], M["unlabelled_identical_candidates"])
    v = find("validate_submission.py")
    td = os.path.dirname(find("test_source1.tsv"))
    r = subprocess.run([sys.executable, v, "--matching", om, "--candidate", oc, "--test-dir", td, "--check-ids"], capture_output=True, text=True)
    M["validator"] = (r.stdout + r.stderr)[-1500:]
    log("validator:", M["validator"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        M["error"] = traceback.format_exc(); log("ERROR", M["error"]); save_metrics()
