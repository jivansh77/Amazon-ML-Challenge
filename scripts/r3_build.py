"""R3 route build: r3_v1 = the base build + R3 route adds (prob >= 0.70, unclaimed records, exclusivity);
rows of the test country without training labels (France) byte-identical; validator --check-ids; diff report.

  python scripts/r3_build.py --data <D> --scored work/r3_score --base_matching BASE/matching_results.tsv \
      --base_candidates BASE/candidate_pairs.tsv --out work/r3_build

Writes output/matching_results_r3_v1.tsv and output/candidate_pairs_r3_v1.tsv to --out (the base of scripts/r6_score.py).

Based on scripts/r3_score.py (US/India recall routes, STEPS 4-6).

4  Score the kept-route candidates (from scripts/r3_candidates.py) for clean val and test: string features + the
   e5-base cross-encoder of round 2 (--ce_dir, fp16). Fit a small XGBoost on clean val (2 folds by S1, out-of-fold) predicting a
   match for the route-only candidates.
5  GATE on clean val (reproduce the proxy blend first: dec p + base2 CE, logit w 0.6, thr 0.75 = 0.98942):
   add route candidates with prob >= t (unclaimed records only, exclusivity among the adds), t tuned on fold A,
   checked on fold B; pass = >= +0.0002 on both folds, singletons not worse.
6  If it passes: route_adds_test.parquet (s1, m, prob) and a delta-edited copy of the submitted file
   (--base_matching + route adds; exclusivity; rows of test countries without training labels
   byte-identical; validator --check-ids). Nothing is submitted.
Country is used only to leave the unlabelled test country (France) untouched; never as a model feature.
"""
import argparse, hashlib, json, os, subprocess, sys, time, traceback

import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.io import find_dataset_dir

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dataset dir (train/ and test/)")
ap.add_argument("--scored", required=True, help="scripts/r3_score.py output dir (route_adds_test.parquet)")
ap.add_argument("--base_matching", required=True, help="matching_results.tsv of the decoded build the route extends")
ap.add_argument("--base_candidates", required=True, help="candidate_pairs.tsv of that build")
ap.add_argument("--validator", default="", help="the challenge's utils/validate_submission.py (optional)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DD = find_dataset_dir(args.data)
W = args.out
os.makedirs(W, exist_ok=True)
T0 = time.time()
M = {}
LOGF = open(f"{W}/progress.log", "a")
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


def build_files(ft, t, tec, lab):
    """Submitted file + route adds (prob >= t, unclaimed, labelled-country S1 only), rows of other S1 untouched."""
    pm, pc = args.base_matching, args.base_candidates
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
    if args.validator:
        r = subprocess.run([sys.executable, args.validator, "--matching", om, "--candidate", oc, "--test-dir", os.path.join(DD, "test"),
                            "--check-ids"], capture_output=True, text=True)
        M["validator"] = (r.stdout + r.stderr)[-1500:]
        log("validator:", M["validator"])


T_ADD = 0.70
NAME = "r3_v1"


def diff_report(ft, t, tec):
    """Counts and examples of the added pairs (by country, probability, claim / exclusivity losses)."""
    add = ft.filter((pl.col("prob") >= t) & (pl.col("claimed") == 0))
    add = add.filter(pl.col("prob") == pl.col("prob").max().over("m")).unique("m", keep="first").join(tec, on="s1")
    M["diff"] = {"adds_by_country": dict(add.group_by("country").len().iter_rows()),
                 "adds_s1_by_country": dict(add.group_by("country").agg(pl.col("s1").n_unique()).iter_rows()),
                 "prob_bands": {str(b): int((add["prob"] >= b).sum()) for b in (0.7, 0.8, 0.9, 0.95, 0.99)},
                 "above_t_but_claimed": int(ft.filter((pl.col("prob") >= t) & (pl.col("claimed") == 1)).height),
                 "above_t_lost_to_exclusivity": int(ft.filter((pl.col("prob") >= t) & (pl.col("claimed") == 0)).height - add.height)}
    dd = os.path.join(DD, "test")
    txt = lambda d, c: d.select(pl.col("entity_id").alias(c), (pl.col("business_name") + " | " + pl.col("business_address")).alias(c + "_t"))
    s1 = txt(read_source(f"{dd}/test_source1.tsv"), "s1")
    s23 = txt(pl.concat([read_source(f"{dd}/test_source{k}.tsv") for k in (2, 3)]), "m")
    ex = add.sample(n=min(40, add.height), seed=0).join(s1, on="s1").join(s23, on="m").sort("prob")
    with open(f"{W}/diff_report_{NAME}.md", "w") as f:
        f.write(f"# {NAME}: base build + R3 route adds (t={t})\n\n")
        f.write("```\n" + json.dumps(M["diff"], indent=1) + "\n```\n\n| country | prob | S1 | added record |\n|---|---|---|---|\n")
        for r in ex.iter_rows(named=True):
            f.write(f"| {r['country']} | {r['prob']:.3f} | {r['s1_t']} | {r['m_t']} |\n")
    log("diff", json.dumps(M["diff"]))


def main():
    ft = pl.read_parquet(f"{args.scored}/route_adds_test.parquet")
    tec = read_source(os.path.join(DD, "test", "test_source1.tsv")).select(pl.col("entity_id").alias("s1"), "country")
    trc = read_source(os.path.join(DD, "train", "train_source1.tsv")).select("country").unique()
    lab = tec.join(trc, on="country")
    ft = ft.join(lab.select("s1"), on="s1")
    log("route_adds_test rows", ft.height)
    diff_report(ft, T_ADD, tec)
    save_metrics()
    build_files(ft, T_ADD, tec, lab)
    for a, b in (("matching_results_rt.tsv", f"matching_results_{NAME}.tsv"), ("candidate_pairs_rt.tsv", f"candidate_pairs_{NAME}.tsv")):
        os.rename(f"{W}/output/{a}", f"{W}/output/{b}")
    M["done"] = True
    save_metrics()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        M["error"] = traceback.format_exc(); log("ERROR", M["error"]); save_metrics()
