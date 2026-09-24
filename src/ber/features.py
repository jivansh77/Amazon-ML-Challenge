"""Pair features for candidate (S1, S2/S3) pairs, fully vectorised.

All features are pure similarities / agreement flags between the two records, plus
blocking-context features (ranks and margins). No country, token or id features are
used, so the model can transfer to countries unseen in training (France).

String similarities use rapidfuzz.process.cpdist (element-wise, multithreaded C++);
IDF-weighted overlaps use row-wise products of sparse token matrices.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer


def _split(s):
    return s.split()


class TokenSpace:
    """Binary token matrices for S1 and S2/S3 records over a shared vocabulary + idf."""

    def __init__(self, a_texts, b_texts):
        cv = CountVectorizer(analyzer=_split, binary=True, dtype=np.float32)
        cv.fit(list(a_texts) + list(b_texts))
        self.A = cv.transform(a_texts).tocsr()
        self.B = cv.transform(b_texts).tocsr()
        df = np.asarray(self.A.sum(0)).ravel() + np.asarray(self.B.sum(0)).ravel()
        n = self.A.shape[0] + self.B.shape[0]
        self.idf = np.log(n / np.maximum(df, 1)).astype(np.float32)

    def pair_stats(self, qi, ci):
        """Return (idf-weighted jaccard, max idf of shared tokens, n shared) per pair."""
        Aq = self.A[qi]
        Bc = self.B[ci]
        W = sp.diags(self.idf)
        inter = Aq.multiply(Bc)                       # binary shared tokens
        inter_w = inter @ W
        wi = np.asarray(inter_w.sum(1)).ravel()
        wa = np.asarray((Aq @ W).sum(1)).ravel()
        wb = np.asarray((Bc @ W).sum(1)).ravel()
        union = wa + wb - wi
        jac = np.where(union > 0, wi / np.maximum(union, 1e-9), -1.0)
        mx = inter_w.max(1).toarray().ravel()
        ns = np.asarray(inter.sum(1)).ravel()
        return jac.astype(np.float32), mx.astype(np.float32), ns.astype(np.float32)


def _cp(a, b, scorer, jobs):
    return cpdist(a, b, scorer=scorer, workers=jobs, dtype=np.float32)


def pair_features(s1, s23, cand, jobs=4, chunk=4_000_000):
    """cand: frame with qi (row in s1) and ci (row in s23). Returns float32 feature frame."""
    ts_n = TokenSpace(s1["name_core"].to_list(), s23["name_core"].to_list())
    ts_a = TokenSpace(s1["addr_clean"].to_list(), s23["addr_clean"].to_list())

    def col(df, c):
        return df[c].to_numpy()

    A = {c: col(s1, c) for c in ["name_clean", "name_core", "name_legal", "addr_clean", "addr_nums", "addr_words"]}
    B = {c: col(s23, c) for c in ["name_clean", "name_core", "name_alt", "name_legal", "addr_clean", "addr_nums",
                                  "addr_words"]}
    B["name_is_web"] = col(s23, "name_is_web").astype(np.float32)
    # derived per-record strings (computed once per record, not per pair)
    A["compact"] = np.array([s.replace(" ", "") for s in A["name_core"]], dtype=object)
    B["compact"] = np.array([s.replace(" ", "") for s in B["name_core"]], dtype=object)
    A["acr"] = np.array(["".join(w[0] for w in s.split()) if len(s.split()) >= 2 else "\x00" for s in A["name_core"]], dtype=object)
    B["acr"] = np.array(["".join(w[0] for w in s.split()) if len(s.split()) >= 2 else "\x01" for s in B["name_core"]], dtype=object)
    A["num0"] = np.array([s.split()[0] if s else "" for s in A["addr_nums"]], dtype=object)
    B["num0"] = np.array([s.split()[0] if s else "" for s in B["addr_nums"]], dtype=object)

    qi_all = cand["qi"].to_numpy()
    ci_all = cand["ci"].to_numpy()
    parts = []
    for s in range(0, len(qi_all), chunk):
        qi, ci = qi_all[s:s + chunk], ci_all[s:s + chunk]
        a = {k: v[qi] for k, v in A.items()}
        b = {k: v[ci] for k, v in B.items()}
        f = {}
        f["n_ratio"] = _cp(a["name_core"], b["name_core"], fuzz.ratio, jobs)
        f["n_tset"] = _cp(a["name_core"], b["name_core"], fuzz.token_set_ratio, jobs)
        f["n_tsort"] = _cp(a["name_core"], b["name_core"], fuzz.token_sort_ratio, jobs)
        f["n_partial"] = _cp(a["name_core"], b["name_core"], fuzz.partial_ratio, jobs)
        f["n_jw"] = _cp(a["name_core"], b["name_core"], JaroWinkler.normalized_similarity, jobs)
        f["n_clean_tset"] = _cp(a["name_clean"], b["name_clean"], fuzz.token_set_ratio, jobs)
        f["n_clean_ratio"] = _cp(a["name_clean"], b["name_clean"], fuzz.ratio, jobs)
        has_alt = b["name_alt"] != ""
        f["n_alt_tset"] = np.where(has_alt, _cp(a["name_core"], b["name_alt"], fuzz.token_set_ratio, jobs), -1)
        f["n_compact_ratio"] = _cp(a["compact"], b["compact"], fuzz.ratio, jobs)
        f["n_compact_partial"] = _cp(a["compact"], b["compact"], fuzz.partial_ratio, jobs)
        f["n_acr_a"] = (a["acr"] == b["compact"]).astype(np.float32)
        f["n_acr_b"] = (b["acr"] == a["compact"]).astype(np.float32)
        f["n_widf"], f["n_widf_max"], f["n_shared"] = ts_n.pair_stats(qi, ci)
        f["n_len_a"] = np.fromiter((len(x) for x in a["name_core"]), np.float32, len(qi))
        f["n_len_b"] = np.fromiter((len(x) for x in b["name_core"]), np.float32, len(qi))
        f["legal_eq"] = (a["name_legal"] == b["name_legal"]).astype(np.float32)
        f["legal_missing"] = ((a["name_legal"] == "") != (b["name_legal"] == "")).astype(np.float32)
        f["b_is_web"] = b["name_is_web"]
        empty = b["addr_clean"] == ""
        for nm, sc in [("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio),
                       ("a_partial", fuzz.partial_ratio), ("a_tsort", fuzz.token_sort_ratio)]:
            f[nm] = np.where(empty, -1, _cp(a["addr_clean"], b["addr_clean"], sc, jobs))
        f["a_words_tset"] = np.where(empty, -1, _cp(a["addr_words"], b["addr_words"], fuzz.token_set_ratio, jobs))
        f["a_widf"], f["a_widf_max"], f["a_shared"] = ts_a.pair_stats(qi, ci)
        no_num = (a["addr_nums"] == "") | (b["addr_nums"] == "")
        f["num_tset"] = np.where(no_num, -1, _cp(a["addr_nums"], b["addr_nums"], fuzz.token_set_ratio, jobs))
        f["num_first_eq"] = np.where(no_num, -1, (a["num0"] == b["num0"]).astype(np.float32))
        f["num_first_ratio"] = np.where(no_num, -1, _cp(a["num0"], b["num0"], fuzz.ratio, jobs))
        f["a_len_a"] = np.fromiter((len(x) for x in a["addr_clean"]), np.float32, len(qi))
        f["a_len_b"] = np.fromiter((len(x) for x in b["addr_clean"]), np.float32, len(qi))
        f["b_addr_empty"] = empty.astype(np.float32)
        f["x_nameb_in_addra"] = _cp(b["name_core"], a["addr_clean"], fuzz.partial_ratio, jobs)
        parts.append(pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()}))
    return pl.concat(parts)


def context_features(df, score_cols):
    """Per-S1 and per-candidate context for each score column (ranks, gaps, counts)."""
    exprs = []
    for c in score_cols:
        exprs += [
            (pl.col(c) - pl.col(c).max().over("qi")).alias(f"{c}_gap_q"),
            pl.col(c).rank("ordinal", descending=True).over("qi").cast(pl.Float32).alias(f"{c}_rank_q"),
            (pl.col(c) - pl.col(c).max().over("ci")).alias(f"{c}_gap_c"),
            pl.col(c).rank("ordinal", descending=True).over("ci").cast(pl.Float32).alias(f"{c}_rank_c"),
        ]
    exprs += [pl.len().over("qi").cast(pl.Float32).alias("n_cand_q"),
              pl.len().over("ci").cast(pl.Float32).alias("n_cand_c")]
    return df.with_columns(exprs)
