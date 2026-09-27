"""sh-hn: house-number (hn) change test and probes (NOT submitted).
(1) Bias check: hn-change rate among CONFIDENT claimed test pairs (stage-2 p >= 0.998) vs among TRUE train matches
    (labels), per country (US / India). If they agree, the France confident rate (0.9%) reflects the generator.
(2) Claimed pairs of the best file with both hn present and different, per country, by blend band
    ([.93,.97), [.97,.998), >= .998 for France; US/India bands approximate: base2 CE instead of the unavailable base CE)
    and by change type (upward shift 1..10 = decoy signature, downward 1..10, other).
    France blend = sigmoid(w logit(p) + (1-w) logit(ce_fr)), p from ber-dec-test3, ce_fr from ber-ce-fr-train;
    w chosen by agreement with the best file's claimed set.
(3) Probes on top of P-FILS-A2: P-HN-A drops France claimed pairs with changed hn and blend < 0.998; P-HN-B drops
    all France claimed pairs with changed hn. Expected LB effect by exact per-S1 F0.5 arithmetic, each dropped pair a
    true match with probability q = expected matches among changed pairs / changed pairs, where expected matches =
    r_m x claimed true matches in that band (r_m = France confident hn-change rate x US/India bias ratio train-true/confident).
Outputs: hn.json, hn_log.txt, metrics.json, diff_report.json, P_HN_A.tsv, P_HN_B.tsv."""
import glob, json, os, subprocess, time

import numpy as np
import polars as pl

W = "/kaggle/working"
LOG = open(f"{W}/hn_log.txt", "w")
OUT = {}
PROBE = "matching_results_dec_cebase_frv2_frce_fr93.tsv"
CANDS = "candidate_pairs_dec_frv2.tsv"
CHUNK = 5_000_000
STOP = ["inc", "llc", "ltd", "corp", "corporation", "co", "company", "pvt", "private", "limited", "llp", "lp", "pc",
        "pa", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "the", "and", "of", "et", "de", "des",
        "du", "la", "le", "les", "d", "l", "s", "a"]
FR_SUFFIX = ["fils", "groupe", "developpement", "associes", "cie", "compagnie", "freres", "frs"]
US_NOISE = ["center", "services", "service", "partners", "incorporated", "com", "lnc", "5ervices", "5ervice"]
FOCUS = FR_SUFFIX + US_NOISE + ["france", "club", "ecole", "centre", "federation"]


def log(*a):
    """Print to stdout and the log file, flushed."""
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def dump():
    """Write results so far."""
    json.dump(OUT, open(f"{W}/hn.json", "w"), indent=1, default=str)


def find(name, sub=""):
    """First path under /kaggle/input with basename `name` (optionally containing `sub`)."""
    hits = [h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if sub in h]
    return hits[0]


def read_src(path):
    """Read a source TSV as strings, quoting off."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=False)


def read_lists(path, col):
    """Explode an id-list file into (s1, m) pairs."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    return (df.select(pl.col("source1_entity_id").alias("s1"), pl.col(col).str.split(",").alias("m"))
            .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))


def norm(col):
    """NFKD, drop marks, lowercase, non-alphanumerics -> space."""
    return (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())


def prep(df):
    """Normalised text, core name tokens (legal forms / function words removed), first house number."""
    df = df.with_columns(norm("business_name").alias("n"), norm("business_address").alias("a"))
    return df.with_columns(
        pl.col("n").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).list.unique().alias("core"),
        pl.col("a").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"))


def pair_feats(p):
    """Address token_sort_ratio, dropped / added core words, house-number relation for aligned (S1 side = *2) pairs."""
    from rapidfuzz import fuzz, process
    ra, rn = [], []
    for i in range(0, p.height, CHUNK):
        q = p.slice(i, CHUNK)
        ra.append(process.cpdist(q["a"].to_list(), q["a2"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1))
        rn.append(process.cpdist(q["n"].to_list(), q["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1))
    cat = lambda x: np.concatenate(x) if x else np.array([])
    p = p.with_columns(pl.Series("ra", cat(ra), pl.Float64), pl.Series("rn", cat(rn), pl.Float64))
    return p.with_columns(
        pl.col("core2").list.set_difference(pl.col("core")).alias("dropped"),
        pl.col("core").list.set_difference(pl.col("core2")).alias("added"),
        pl.when(pl.col("hn").is_null() | pl.col("hn2").is_null()).then(pl.lit(None, pl.Boolean))
        .otherwise(pl.col("hn") == pl.col("hn2")).alias("same_hn")
    ).with_columns(pl.col("dropped").list.len().alias("nd"), pl.col("added").list.len().alias("na"),
                   pl.col("added").list.first().alias("aw"))


def write_file(path, s1_order, pairs):
    """Write a matching_results.tsv in the S1 order of the best file."""
    agg = pairs.group_by("s1").agg(pl.col("m").unique(maintain_order=True).str.join(",").alias("ids"))
    out = s1_order.join(agg, on="s1", how="left").with_columns(pl.col("ids").fill_null(""))
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out.select("s1", "ids").iter_rows():
            f.write(f"{a}\t{b}\n")


def diff(base, new, cty):
    """Per country: S1 rows changed, pairs added, pairs removed, S1 empty before/after."""
    res = {}
    kb = base.with_columns(pl.lit(1).alias("b")); kn = new.with_columns(pl.lit(1).alias("n"))
    j = kb.join(kn, on=["s1", "m"], how="full", coalesce=True).join(cty, on="s1", how="left")
    for c in sorted(cty["country"].unique().to_list()):
        g = j.filter(pl.col("country") == c)
        add = g.filter(pl.col("b").is_null()); rem = g.filter(pl.col("n").is_null())
        n1 = cty.filter(pl.col("country") == c).height
        nb = base.join(cty.filter(pl.col("country") == c), on="s1")["s1"].n_unique()
        nn = new.join(cty.filter(pl.col("country") == c), on="s1")["s1"].n_unique()
        res[c] = {"S1": n1, "S1_changed": pl.concat([add["s1"], rem["s1"]]).n_unique(), "pairs_added": add.height,
                  "pairs_removed": rem.height, "S1_nonempty_before": nb, "S1_nonempty_after": nn,
                  "empty_before": round(1 - nb / n1, 5), "empty_after": round(1 - nn / n1, 5)}
    return res


def fm(x):
    """Null-safe float of a mean."""
    return None if x is None else float(x)


def band(col, c):
    """Blend band label."""
    return (pl.when(pl.col(col) >= 0.998).then(pl.lit(">=.998")).when(pl.col(col) >= 0.97).then(pl.lit("[.97,.998)"))
            .when(pl.col(col) >= 0.93).then(pl.lit("[.93,.97)")).when(pl.col(col) >= 0.80).then(pl.lit("[.80,.93)"))
            .otherwise(pl.lit("<.80")))


def blend(df, ce, w):
    """Logit blend of stage-2 p with CE ce (pairs without CE keep p), as scripts/blend.py."""
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    d = df.join(ce, on=["s1", "m"], how="left")
    b = 1 / (1 + (-(w * lg("p") + (1 - w) * lg("ce"))).exp())
    return d.with_columns(pl.when(pl.col("ce").is_null()).then(pl.col("p")).otherwise(b).alias("bp"))


def f05(tp, npred, ntrue):
    """Per-entity F0.5 from counts (empty/empty = 1)."""
    if npred == 0 and ntrue == 0: return 1.0
    if npred == 0 or ntrue == 0 or tp == 0: return 0.0
    p, r = tp / npred, tp / ntrue
    return 1.25 * p * r / (0.25 * p + r)


def exp_gain(k, j, q):
    """Expected F0.5 change for one S1 with k predicted pairs when j are dropped, each dropped pair a true match
    with prob q (kept pairs assumed correct, truth = kept + matched dropped)."""
    from math import comb
    g = 0.0
    for t in range(j + 1):
        pr = comb(j, t) * q ** t * (1 - q) ** (j - t)
        g += pr * (f05(k - j, k - j, k - j + t) - f05(k - j + t, k, k - j + t))
    return g


def hn_of(df):
    """Normalised first house number of each record."""
    return df.with_columns(norm("business_address").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("hn"))


def main():
    """Bias check, change counts, probes."""
    t0 = time.time()
    subprocess.run("pip install -q -U 'rapidfuzz>=3.6' 2>&1 | tail -1", shell=True)
    # ---------------- (1) train true matches
    tr1 = hn_of(read_src(find("train_source1.tsv"))).select(pl.col("entity_id").alias("s1"), pl.col("hn").alias("hn2"), "country")
    trp = hn_of(pl.concat([read_src(find(f"train_source{k}.tsv")) for k in (2, 3)])).select(pl.col("entity_id").alias("m"), "hn")
    gt = read_lists(find("train_ground_truth.tsv"), "matched_entity_ids").join(tr1, on="s1").join(trp, on="m")
    gt = gt.filter(pl.col("hn").is_not_null() & pl.col("hn2").is_not_null())
    r1 = {c: {"train_true_pairs_with_hn": g.height, "train_true_hn_changed": (round(fm((g["hn"] != g["hn2"]).mean()), 4) if g.height else None)}
          for c, g in ((c, gt.filter(pl.col("country") == c)) for c in ("US", "India"))}
    del tr1, trp, gt
    te1 = hn_of(read_src(find("test_source1.tsv"))).select(pl.col("entity_id").alias("s1"), pl.col("hn").alias("hn2"), "country")
    tep = hn_of(pl.concat([read_src(find(f"test_source{k}.tsv")) for k in (2, 3)])).select(pl.col("entity_id").alias("m"), "hn")
    pred = read_lists(find(PROBE), "matched_entity_ids")
    probe_raw = pl.read_csv(find(PROBE), separator="\t", quote_char=None, infer_schema_length=0)
    s1_order = probe_raw.select(pl.col("source1_entity_id").alias("s1"))
    cty = te1.select("s1", "country"); n1 = dict(cty.group_by("country").len().rows())
    sc = pl.read_parquet(find("test_scores.parquet", "ber-dec-test3")).select("s1", "m", "p")
    cefr = pl.read_parquet(find("ce_test.parquet", "ber-ce-fr-train")).select("s1", "m", "ce").unique(["s1", "m"], keep="last")
    ceb2 = pl.read_parquet(find("ce_test.parquet", "ber-ce-base2")).select("s1", "m", "ce").unique(["s1", "m"], keep="last")
    log(f"loaded {time.time()-t0:.0f}s scores {sc.height} cefr {cefr.height} ceb2 {ceb2.height}")
    sc = sc.join(cty, on="s1")
    claimed = pred.with_columns(pl.lit(True).alias("claimed"))
    # choose France w by agreement: claimed <=> blend >= .93 (before exclusivity)
    frs = sc.filter(pl.col("country") == "France").join(claimed, on=["s1", "m"], how="left").with_columns(pl.col("claimed").fill_null(False))
    agree = {}
    for w in (0.5, 0.6, 0.7, 0.8, 1.0):
        b = blend(frs, cefr, w)
        agree[w] = {"claimed_with_bp_ge_.93": round(fm(b.filter(pl.col("claimed"))["bp"].ge(0.93).mean()) or 0.0, 4),
                    "unclaimed_with_bp_ge_.93_per_S1": round(b.filter(~pl.col("claimed") & (pl.col("bp") >= 0.93)).height / n1["France"], 4)}
    wfr = max(agree, key=lambda w: agree[w]["claimed_with_bp_ge_.93"] - agree[w]["unclaimed_with_bp_ge_.93_per_S1"])
    OUT["france_w_agreement"] = {str(k): v for k, v in agree.items()}; OUT["france_w"] = wfr
    log("France w agreement:", agree, "-> w", wfr)
    allb = pl.concat([blend(sc.filter(pl.col("country") == "France"), cefr, wfr),
                      blend(sc.filter(pl.col("country") != "France"), ceb2, 0.6)])
    cl = (pred.join(allb.select("s1", "m", "p", "bp"), on=["s1", "m"], how="left").join(cty, on="s1")
          .join(te1.select("s1", "hn2"), on="s1").join(tep, on="m"))
    cl = cl.with_columns(band("bp", None).alias("band"),
                         (pl.col("hn").cast(pl.Int64, strict=False) - pl.col("hn2").cast(pl.Int64, strict=False)).alias("dhn"))
    both = cl.filter(pl.col("hn").is_not_null() & pl.col("hn2").is_not_null())
    both = both.with_columns((pl.col("hn") != pl.col("hn2")).alias("chg"),
                             pl.when(pl.col("dhn").is_between(1, 10)).then(pl.lit("up1-10"))
                             .when(pl.col("dhn").is_between(-10, -1)).then(pl.lit("down1-10")).otherwise(pl.lit("other")).alias("ctype"))
    for c in sorted(n1):
        g = both.filter((pl.col("country") == c) & (pl.col("p") >= 0.998))
        r1.setdefault(c, {})["test_confident_claimed_pairs_with_hn"] = g.height
        r1[c]["test_confident_claimed_hn_changed"] = (round(fm(g["chg"].mean()), 4) if g.height else None)
    OUT["bias_check"] = r1
    log("(1) bias check:", json.dumps(r1, indent=1)); dump()
    # ---------------- (2) counts
    r2 = {}
    for c in sorted(n1):
        g = both.filter(pl.col("country") == c)
        tab = (g.group_by("band").agg(pl.len().alias("claimed_with_hn"), pl.col("chg").sum().alias("changed"),
                                      (pl.col("chg") & (pl.col("ctype") == "up1-10")).sum().alias("up1_10"),
                                      (pl.col("chg") & (pl.col("ctype") == "down1-10")).sum().alias("down1_10"),
                                      (pl.col("chg") & (pl.col("ctype") == "other")).sum().alias("other"))
               .with_columns((pl.col("changed") / pl.col("claimed_with_hn")).round(4).alias("rate")).sort("band"))
        r2[c] = {"claimed_pairs": cl.filter(pl.col("country") == c).height, "with_both_hn": g.height,
                 "changed": int(g["chg"].sum()), "by_band": tab.rows(named=True)}
    OUT["changed_counts"] = r2
    log("(2) changed counts:", json.dumps(r2, indent=1, default=str)); dump()
    # ---------------- (3) probes on top of P-FILS-A2
    t1p = prep(read_src(find("test_source1.tsv")).rename({"entity_id": "id"}))
    tpp = prep(pl.concat([read_src(find(f"test_source{k}.tsv")) for k in (2, 3)]).rename({"entity_id": "id"}))
    cand = read_lists(find(CANDS), "candidate_entity_ids")
    s1c = t1p.select(pl.col("id").alias("s1"), pl.col("n").alias("n2"), pl.col("a").alias("a2"), pl.col("core").alias("core2"),
                     pl.col("hn").alias("hn2"), "country").filter(pl.col("country") == "France")
    cp = pair_feats(cand.join(s1c, on="s1").join(tpp.select(pl.col("id").alias("m"), "n", "a", "core", "hn"), on="m"))
    best = cp.with_columns((pl.col("rn") + pl.col("ra")).alias("sc")).sort("sc", descending=True).group_by("m").first()
    un = best.join(pred.select("m").unique(), on="m", how="anti").filter(pl.col("same_hn") == True)
    add_a2 = un.filter((pl.col("nd") <= 1) & (pl.col("na") == 1) & pl.col("aw").is_in(["fils", "groupe", "developpement", "associes"])).select("s1", "m")
    a2 = pl.concat([pred, add_a2]).unique(maintain_order=True)
    OUT["A2_pairs_added"] = add_a2.height
    fr_chg = both.filter((pl.col("country") == "France") & pl.col("chg"))
    # match rate among changed pairs per band: r_m x (claimed pairs in band x band precision ~ 1) / changed in band
    # selection-bias correction: confident claimed pairs understate the true-match change rate by the US/India ratio
    ratio = float(np.mean([r1[c]["train_true_hn_changed"] / r1[c]["test_confident_claimed_hn_changed"] for c in ("US", "India")]))
    rm = (r1["France"]["test_confident_claimed_hn_changed"] or 0.0) * ratio
    OUT["bias_ratio_true_over_confident"] = ratio; OUT["france_true_hn_change_rate_est"] = rm
    frb = {b["band"]: b for b in r2["France"]["by_band"]}
    q_band = {bn: min(1.0, rm * v["claimed_with_hn"] / max(v["changed"], 1)) for bn, v in frb.items()}
    OUT["q_match_rate_among_changed_by_band"] = q_band
    log("q (match rate among changed) by band:", q_band)
    drops = {"P_HN_A": fr_chg.filter(pl.col("bp") < 0.998), "P_HN_B": fr_chg}
    val = find("validate_submission.py"); tdir = os.path.dirname(find("test_source1.tsv")); cfile = find(CANDS)
    rep = {}
    ka = a2.group_by("s1").len().rename({"len": "k"})
    for name, dr in drops.items():
        d = dr.select("s1", "m", "band")
        newp = a2.join(d.select("s1", "m"), on=["s1", "m"], how="anti")
        path = f"{W}/{name}.tsv"; write_file(path, s1_order, newp)
        rc = subprocess.run(f"python3 {val} --matching {path} --candidate {cfile} --test-dir {tdir}", shell=True, capture_output=True, text=True)
        per = d.group_by("s1").agg(pl.len().alias("j"), pl.col("band").alias("bands")).join(ka, on="s1")
        gain = 0.0
        for k, j, bands in per.select("k", "j", "bands").rows():
            q = float(np.mean([q_band.get(b, rm) for b in bands]))
            gain += exp_gain(k, j, q)
        ntot = sum(n1.values())
        rep[name] = {"validator": "PASS" if rc.returncode == 0 else "FAIL", "validator_tail": rc.stdout[-300:],
                     "pairs_dropped": d.height, "dropped_by_band": d.group_by("band").len().rows(),
                     "France_S1_affected": per.height, "S1_turned_empty": int((per["k"] == per["j"]).sum()),
                     "expected_LB_delta_vs_A2": gain / ntot, "expected_France_F05_delta": gain / n1["France"],
                     "diff_vs_best": diff(pred, newp, cty)}
        log(f"{name}: {json.dumps({k: v for k, v in rep[name].items() if k != 'validator_tail'}, default=str)}")
    json.dump(rep, open(f"{W}/diff_report.json", "w"), indent=1, default=str)
    OUT["probes"] = rep; dump()
    json.dump({"runtime_s": round(time.time() - t0), "probe_base": PROBE + " + A2"}, open(f"{W}/metrics.json", "w"))
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
