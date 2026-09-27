"""Native-script (Indic) out-of-candidate route for India, trained on train labels (see EXPERIMENTS.md).

~0.9% of India S1 have a true match whose name is written in an Indic script and that no blocking route put in the
candidate lists (generic transliterated names such as "Global Exports" tie with many S1, and the address is noisy).
This route transliterates with the pipeline normaliser (learned Indic dictionary + rule-based IAST), retrieves the top S1
of the country by name (char 3-gram TF-IDF, weight 0.6) + address (word TF-IDF, weight 0.4), and scores the top-5 pairs
with a classifier trained on train labels (2-fold by record). final_build.py adds a record's top pair when q >= --thr,
the record is unclaimed and the pair is NOT in the pipeline's candidate list (pairs the pipeline scored and rejected are
only ~15% right on validation; out-of-candidate pairs are 97.7% right at q >= 0.9).

  python scripts/native_route.py --data DS --work WORK --claimed PRED.parquet --cand TEST_BLEND.parquet --out ADDS.parquet
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, polars as pl, scipy.sparse as sp, xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from ber import normalize as N
from ber.io import find_dataset_dir, read_source
from ber.normalize import normalize_frame

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--claimed", required=True, help="parquet (s1, m) of the current matches")
ap.add_argument("--cand", required=True, help="parquet (s1, m) of the pipeline's candidate pairs")
ap.add_argument("--thr", type=float, default=0.9); ap.add_argument("--jobs", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
N.set_indic_dictionary(json.load(open(os.path.join(os.path.dirname(__file__), "..", "artifacts", "indic_dict.json"))))
dd = find_dataset_dir(a.data)
NAT = pl.col("business_name").str.contains(r"[ऀ-෿]")
F = ['cos', 'cos_n', 'cos_a', 'r_c', 'ts_c', 'pr_c', 'ts_a', 'core_eq', 'leg_eq', 'l2e', 'l1e', 'nd', 'na', 'lt1', 'lt2', 'num_int', 'nn1', 'nn2',
     'first_num_eq', 'w_int', 'nw1', 'nw2', 'a2e', 'pin_eq', 'num_j', 'w_j', 'rk', 'ncand', 'cos1', 'cos2', 'tsc1', 'n_core_eq', 'tsa1', 'marg',
     'gap_tsc', 'gap_tsa', 's1_top1_cnt', 's1_ncand']


def load(split):
    s1 = read_source(os.path.join(dd, split, f"{split}_source1.tsv")).filter(pl.col("country") == "India")
    s23 = pl.concat([read_source(os.path.join(dd, split, f"{split}_source{k}.tsv")) for k in (2, 3)]).filter((pl.col("country") == "India") & NAT)
    fill = [pl.col("business_address").fill_null(""), pl.col("business_name").fill_null("")]
    return normalize_frame(s1.with_columns(fill), n_jobs=a.jobs), normalize_frame(s23.with_columns(fill), n_jobs=a.jobs)


def knn(A, B, k):
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32, sublinear_tf=True)
    va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
    An = vn.fit_transform(A["name_core"].to_list()); Bn = vn.transform(B["name_core"].to_list())
    Aa = va.fit_transform(A["addr_clean"].to_list()); Ba = va.transform(B["addr_clean"].to_list())
    w1, w2 = np.sqrt(0.6), np.sqrt(0.4)
    XA = sp.hstack([An * w1, Aa * w2]).tocsr(); XB = sp.hstack([Bn * w1, Ba * w2]).tocsr()
    R = sp_matmul_topn(XB, XA.T.tocsr(), top_n=k, threshold=0.25, n_threads=a.jobs).tocoo()
    ia, ib = R.col, R.row
    cn = np.asarray(An[ia].multiply(Bn[ib]).sum(axis=1)).ravel(); ca = np.asarray(Aa[ia].multiply(Ba[ib]).sum(axis=1)).ravel()
    return pl.DataFrame({"m": B["entity_id"].to_numpy()[ib], "s1": A["entity_id"].to_numpy()[ia], "cos": R.data.astype(np.float32),
                         "cos_n": cn.astype(np.float32), "cos_a": ca.astype(np.float32)})


def feats(d, A, B, topk=5):
    d = d.with_columns(pl.col("cos").rank("ordinal", descending=True).over("m").alias("_rk0")).filter(pl.col("_rk0") <= topk)
    d = d.with_columns(((pl.col("_rk0") == 1).cast(pl.Int32).sum().over("s1")).alias("s1_top1_cnt"), pl.len().over("s1").alias("s1_ncand")).drop("_rk0")
    d = d.join(A.select(pl.col("entity_id").alias("s1"), pl.col("name_core").alias("c1"), pl.col("name_legal").alias("l1"), pl.col("addr_nums").alias("u1"),
                        pl.col("addr_words").alias("w1"), pl.col("addr_clean").alias("ac1")), on="s1")
    d = d.join(B.select(pl.col("entity_id").alias("m"), pl.col("name_core").alias("c2"), pl.col("name_legal").alias("l2"), pl.col("addr_nums").alias("u2"),
                        pl.col("addr_words").alias("w2"), pl.col("addr_clean").alias("ac2")), on="m")
    c1, c2, a1, a2 = d["c1"].to_list(), d["c2"].to_list(), d["ac1"].to_list(), d["ac2"].to_list()
    d = d.with_columns(pl.Series("r_c", process.cpdist(c1, c2, scorer=fuzz.ratio, workers=a.jobs)),
                       pl.Series("ts_c", process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=a.jobs)),
                       pl.Series("pr_c", process.cpdist(c1, c2, scorer=fuzz.partial_ratio, workers=a.jobs)),
                       pl.Series("ts_a", process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=a.jobs)))
    spl = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = d.with_columns(spl("c1").alias("t1"), spl("c2").alias("t2"), spl("u1").alias("n1"), spl("u2").alias("n2"), spl("w1").alias("x1"), spl("w2").alias("x2"))
    d = d.with_columns(
        (pl.col("c1") == pl.col("c2")).cast(pl.Int8).alias("core_eq"), (pl.col("l1") == pl.col("l2")).cast(pl.Int8).alias("leg_eq"),
        (pl.col("l2") == "").cast(pl.Int8).alias("l2e"), (pl.col("l1") == "").cast(pl.Int8).alias("l1e"),
        pl.col("t1").list.set_difference(pl.col("t2")).list.len().alias("nd"), pl.col("t2").list.set_difference(pl.col("t1")).list.len().alias("na"),
        pl.col("t1").list.len().alias("lt1"), pl.col("t2").list.len().alias("lt2"),
        pl.col("n1").list.set_intersection(pl.col("n2")).list.len().alias("num_int"), pl.col("n1").list.len().alias("nn1"), pl.col("n2").list.len().alias("nn2"),
        (pl.col("n1").list.first() == pl.col("n2").list.first()).fill_null(False).cast(pl.Int8).alias("first_num_eq"),
        pl.col("n1").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{6}$"))).list.first().alias("pin1"),
        pl.col("n2").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{6}$"))).list.first().alias("pin2"),
        pl.col("x1").list.set_intersection(pl.col("x2")).list.len().alias("w_int"), pl.col("x1").list.len().alias("nw1"), pl.col("x2").list.len().alias("nw2"),
        (pl.col("ac2") == "").cast(pl.Int8).alias("a2e"))
    d = d.with_columns(pl.when(pl.col("pin1").is_null() | pl.col("pin2").is_null()).then(-1).when(pl.col("pin1") == pl.col("pin2")).then(1).otherwise(0).alias("pin_eq"),
                       (pl.col("num_int") / pl.max_horizontal(pl.col("nn1"), pl.col("nn2"), 1)).alias("num_j"),
                       (pl.col("w_int") / pl.max_horizontal(pl.col("nw1"), pl.col("nw2"), 1)).alias("w_j"))
    d = d.with_columns(pl.col("cos").rank("ordinal", descending=True).over("m").alias("rk"), pl.len().over("m").alias("ncand"),
                       pl.col("cos").max().over("m").alias("cos1"), pl.col("cos").sort(descending=True).slice(1, 1).first().over("m").fill_null(0).alias("cos2"),
                       pl.col("ts_c").max().over("m").alias("tsc1"), pl.col("core_eq").sum().over("m").alias("n_core_eq"), pl.col("ts_a").max().over("m").alias("tsa1"))
    d = d.with_columns(pl.when(pl.col("rk") == 1).then(pl.col("cos") - pl.col("cos2")).otherwise(pl.col("cos") - pl.col("cos1")).alias("marg"),
                       (pl.col("tsc1") - pl.col("ts_c")).alias("gap_tsc"), (pl.col("tsa1") - pl.col("ts_a")).alias("gap_tsa"))
    return d.select(["m", "s1"] + F)


# train: all India native-script records vs all India S1 (top-20 kNN, top-5 scored), labels from the ground truth
A, B = load("train")
tr = feats(knn(A, B, 20), A, B)
gt = pl.read_csv(os.path.join(dd, "train", "train_ground_truth.tsv"), separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
gt = gt.rename({gt.columns[0]: "s1", gt.columns[1]: "m"}).with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
tr = tr.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0), (pl.col("m").hash(7) % 2).alias("fold"))
models = []
for f in [0, 1]:
    x = tr.filter(pl.col("fold") != f)
    m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, n_jobs=a.jobs, tree_method="hist", min_child_weight=5)
    m.fit(x.select(F).to_numpy(), x["y"].to_numpy()); m.save_model(f"{a.work}/nat_xgb_f{f}.json"); models.append(m)
# test: mean of the two fold models, top pair per record
A, B = load("test")
te = feats(knn(A, B, 10), A, B)
q = np.mean([m.predict_proba(te.select(F).to_numpy())[:, 1] for m in models], axis=0)
top = te.select("m", "s1").with_columns(pl.Series("q", q)).sort("q", descending=True).group_by("m", maintain_order=True).first()
claimed = pl.read_parquet(a.claimed, columns=["m"]).unique()
cand = pl.read_parquet(a.cand, columns=["s1", "m"])
add = top.filter(pl.col("q") >= a.thr).join(claimed, on="m", how="anti").join(cand, on=["s1", "m"], how="anti")
add.write_parquet(a.out)
print("native-route adds:", add.height)
