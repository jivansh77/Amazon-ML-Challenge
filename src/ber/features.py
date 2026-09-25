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
from rapidfuzz.distance import JaroWinkler, Levenshtein
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


A_COLS = ["name_clean", "name_core", "name_legal", "addr_clean", "addr_nums", "addr_words"]
B_COLS = ["name_clean", "name_core", "name_alt", "name_legal", "name_is_web", "addr_clean", "addr_nums", "addr_words"]


def _derived(df, side):
    """Per-record derived strings, computed once in Arrow memory (no Python objects)."""
    nc = pl.col("name_core")
    ntok = nc.str.split(" ")
    acr = pl.when(ntok.list.len() >= 2).then(ntok.list.eval(pl.element().str.slice(0, 1)).list.join("")) \
            .otherwise(pl.lit("\x00" if side == "a" else "\x01"))
    return df.with_columns(nc.str.replace_all(" ", "").alias("compact"), acr.alias("acr"),
                           pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("num0"))


def pair_features(s1, s23, cand, jobs=4, chunk=1_500_000, spaces=None):
    """cand: frame with qi (row in s1) and ci (row in s23). Returns float32 feature frame.
    s1/s23 need the columns in A_COLS/B_COLS; strings are gathered per chunk."""
    ts_n, ts_a = spaces or (TokenSpace(s1["name_core"].to_list(), s23["name_core"].to_list()),
                            TokenSpace(s1["addr_clean"].to_list(), s23["addr_clean"].to_list()))
    A = _derived(s1.select(A_COLS), "a")
    B = _derived(s23.select(B_COLS), "b")
    qi_all = cand["qi"].to_numpy()
    ci_all = cand["cr" if "cr" in cand.columns else "ci"].to_numpy()     # real S2/S3 row (duplicated decoys)
    parts = []
    for s in range(0, len(qi_all), chunk):
        qi, ci = qi_all[s:s + chunk], ci_all[s:s + chunk]
        ga, gb = A[qi], B[ci]                 # row gathers stay in Arrow memory

        def L(df, c):
            return df[c].to_list()
        a = {c: L(ga, c) for c in ["name_clean", "name_core", "addr_clean", "addr_nums", "addr_words", "compact", "num0"]}
        b = {c: L(gb, c) for c in ["name_clean", "name_core", "name_alt", "addr_clean", "addr_nums", "addr_words", "compact", "num0"]}
        f = {}
        f["n_ratio"] = _cp(a["name_core"], b["name_core"], fuzz.ratio, jobs)
        f["n_tset"] = _cp(a["name_core"], b["name_core"], fuzz.token_set_ratio, jobs)
        f["n_tsort"] = _cp(a["name_core"], b["name_core"], fuzz.token_sort_ratio, jobs)
        f["n_partial"] = _cp(a["name_core"], b["name_core"], fuzz.partial_ratio, jobs)
        f["n_jw"] = _cp(a["name_core"], b["name_core"], JaroWinkler.normalized_similarity, jobs)
        f["n_clean_tset"] = _cp(a["name_clean"], b["name_clean"], fuzz.token_set_ratio, jobs)
        f["n_clean_ratio"] = _cp(a["name_clean"], b["name_clean"], fuzz.ratio, jobs)
        has_alt = (gb["name_alt"] != "").to_numpy()
        f["n_alt_tset"] = np.where(has_alt, _cp(a["name_core"], b["name_alt"], fuzz.token_set_ratio, jobs), -1)
        f["n_compact_ratio"] = _cp(a["compact"], b["compact"], fuzz.ratio, jobs)
        f["n_compact_partial"] = _cp(a["compact"], b["compact"], fuzz.partial_ratio, jobs)
        f["n_acr_a"] = (ga["acr"] == gb["compact"]).to_numpy().astype(np.float32)
        f["n_acr_b"] = (gb["acr"] == ga["compact"]).to_numpy().astype(np.float32)
        f["n_widf"], f["n_widf_max"], f["n_shared"] = ts_n.pair_stats(qi, ci)
        f["n_len_a"] = ga["name_core"].str.len_chars().to_numpy()
        f["n_len_b"] = gb["name_core"].str.len_chars().to_numpy()
        la, lb = ga["name_legal"], gb["name_legal"]
        f["legal_eq"] = (la == lb).to_numpy().astype(np.float32)
        f["legal_missing"] = ((la == "") != (lb == "")).to_numpy().astype(np.float32)
        f["b_is_web"] = gb["name_is_web"].cast(pl.Float32).to_numpy()
        empty = (gb["addr_clean"] == "").to_numpy()
        for nm, sc in [("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio),
                       ("a_partial", fuzz.partial_ratio), ("a_tsort", fuzz.token_sort_ratio)]:
            f[nm] = np.where(empty, -1, _cp(a["addr_clean"], b["addr_clean"], sc, jobs))
        f["a_words_tset"] = np.where(empty, -1, _cp(a["addr_words"], b["addr_words"], fuzz.token_set_ratio, jobs))
        f["a_widf"], f["a_widf_max"], f["a_shared"] = ts_a.pair_stats(qi, ci)
        no_num = ((ga["addr_nums"] == "") | (gb["addr_nums"] == "")).to_numpy()
        f["num_tset"] = np.where(no_num, -1, _cp(a["addr_nums"], b["addr_nums"], fuzz.token_set_ratio, jobs))
        f["num_first_eq"] = np.where(no_num, -1, (ga["num0"] == gb["num0"]).to_numpy().astype(np.float32))
        f["num_first_ratio"] = np.where(no_num, -1, _cp(a["num0"], b["num0"], fuzz.ratio, jobs))
        # house-number perturbations are digit edits (705 vs 703); different businesses differ numerically
        f["num_first_lev"] = np.where(no_num, -1, cpdist(a["num0"], b["num0"], scorer=Levenshtein.distance,
                                                         workers=jobs, dtype=np.int32))
        na_ = ga["num0"].str.extract(r"^(\d+)").cast(pl.Float64, strict=False).to_numpy()
        nb_ = gb["num0"].str.extract(r"^(\d+)").cast(pl.Float64, strict=False).to_numpy()
        f["num_first_absdiff"] = np.where(np.isnan(na_) | np.isnan(nb_), -1, np.log1p(np.abs(na_ - nb_)))
        f["num_first_lendiff"] = np.where(no_num, -1, np.abs(ga["num0"].str.len_chars().to_numpy().astype(np.int32)
                                                              - gb["num0"].str.len_chars().to_numpy().astype(np.int32)))
        f["num_count_a"] = ga["addr_nums"].str.count_matches(r"\S+").to_numpy()
        f["num_count_b"] = gb["addr_nums"].str.count_matches(r"\S+").to_numpy()
        f["a_len_a"] = ga["addr_clean"].str.len_chars().to_numpy()
        f["a_len_b"] = gb["addr_clean"].str.len_chars().to_numpy()
        f["b_addr_empty"] = empty.astype(np.float32)
        f["x_nameb_in_addra"] = _cp(b["name_core"], a["addr_clean"], fuzz.partial_ratio, jobs)
        if TOK_ODDS is not None:
            f.update(decoy_features(ga["name_core"], gb["name_core"], na_, nb_))
        del a, b, ga, gb
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


TOK_ODDS = None     # {"extra": frame(token, score), "missing": frame(token, score)} or None


def decoy_features(name_a, name_b, na_, nb_):
    """Signature of generated decoys ("branches" of the S1 business): the S2/S3 name ADDS a qualifier
    word ("Central", "Downtown", "Enterprises") and the house number is shifted by a small step.
    Word scores are log-odds (match vs non-match among near-duplicate names) learned on a held-out
    slice of training S1 (see fit_token_odds); unseen words score 0."""
    n = len(name_a)
    d = pl.DataFrame({"i": np.arange(n, dtype=np.int32), "a": name_a, "b": name_b}).with_columns(
        pl.col("a").str.split(" ").alias("ta"), pl.col("b").str.split(" ").alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference("ta").alias("ex"),
                       pl.col("ta").list.set_difference("tb").alias("mi"))
    out = {"dec_n_extra": d["ex"].list.len().to_numpy().astype(np.float32),
           "dec_n_missing": d["mi"].list.len().to_numpy().astype(np.float32)}
    for kind, col in (("extra", "ex"), ("missing", "mi")):
        g = (d.select("i", col).explode(col).join(TOK_ODDS[kind], left_on=col, right_on="token", how="inner")
             .group_by("i").agg(pl.col("score").min().alias("mn"), pl.col("score").sum().alias("sm")))
        g = pl.DataFrame({"i": np.arange(n, dtype=np.int32)}).join(g, on="i", how="left").sort("i")
        out[f"dec_{kind}_min"] = g["mn"].fill_null(0).to_numpy().astype(np.float32)
        out[f"dec_{kind}_sum"] = g["sm"].fill_null(0).to_numpy().astype(np.float32)
    diff = nb_ - na_
    out["num_first_signed"] = np.where(np.isnan(diff), np.nan, np.sign(diff) * np.log1p(np.abs(diff))).astype(np.float32)
    return out


# Hand-written English -> French equivalents for name qualifier words (France has no training labels;
# its generated names use French qualifiers such as "Groupe", "Participations", "Developpement").
FR_EQUIV = {
    "holdings": ["holding", "participations"], "holding": ["participations"], "associates": ["associes"],
    "development": ["developpement"], "group": ["groupe"], "central": ["centre", "centrale"], "center": ["centre"],
    "north": ["nord"], "south": ["sud"], "east": ["est"], "west": ["ouest"], "northern": ["nord"], "southern": ["sud"],
    "eastern": ["est"], "western": ["ouest"], "health": ["sante"], "clinic": ["clinique"], "care": ["soins"],
    "valley": ["vallee"], "exports": ["export", "exportation"], "overseas": ["outremer"], "summit": ["sommet"],
    "coastal": ["littoral", "cotier"], "harbor": ["port"], "global": ["globale"], "engineering": ["ingenierie"],
    "foods": ["alimentation"], "royal": ["royale"], "digital": ["numerique"], "traders": ["negoce", "commerce"],
    "consultants": ["conseil"], "consulting": ["conseil"], "national": ["nationale"], "partners": ["partenaires"],
    "enterprises": ["entreprises"], "international": ["internationale"], "management": ["gestion"],
    "investments": ["investissements"], "medical": ["medicale"], "pharmacy": ["pharmacie"], "school": ["ecole"],
    "institute": ["institut"], "logistics": ["logistique"], "city": ["ville"], "new": ["nouveau", "nouvelle"],
    "grand": ["grande"], "sons": ["fils"], "brothers": ["freres"], "family": ["famille"], "industries": ["industrie"],
    "solutions": ["solution"], "technologies": ["technologie"], "distribution": ["distributions"],
    "services": ["service"], "trading": ["negoce"], "mountain": ["montagne"], "lake": ["lac"], "river": ["riviere"],
}


def add_french_equivalents(t):
    """Give French equivalents the score of their English word (when not already learned)."""
    known = set(t["token"].to_list())
    sc = dict(zip(t["token"].to_list(), t["score"].to_list()))
    add = [(fr, sc[en]) for en, frs in FR_EQUIV.items() if en in sc for fr in frs if fr not in known]
    if not add:
        return t
    return pl.concat([t, pl.DataFrame({"token": [x[0] for x in add], "score": [x[1] for x in add]},
                                      schema={"token": pl.Utf8, "score": pl.Float32})]).unique("token", keep="first")


def fit_token_odds(name_a, name_b, y, min_count=30, min_sim=80, jobs=4):
    """Learn word log-odds from labelled near-duplicate pairs (name token-set similarity >= min_sim)."""
    sim = cpdist(name_a, name_b, scorer=fuzz.token_set_ratio, workers=jobs)
    d = pl.DataFrame({"a": name_a, "b": name_b, "y": y}).filter(pl.Series(sim >= min_sim))
    d = d.with_columns(pl.col("a").str.split(" ").alias("ta"), pl.col("b").str.split(" ").alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference("ta").alias("ex"), pl.col("ta").list.set_difference("tb").alias("mi"))
    P, N = int(d["y"].sum()), d.height - int(d["y"].sum())
    base = np.log((P + 1) / (N + 1))
    res = {}
    for kind, col in (("extra", "ex"), ("missing", "mi")):
        t = (d.select("y", col).explode(col).filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
             .group_by(col).agg(pl.col("y").sum().alias("pos"), pl.len().alias("n"))
             .filter(pl.col("n") >= min_count)
             .with_columns((((pl.col("pos") + 1) / (pl.col("n") - pl.col("pos") + 1)).log() - base).cast(pl.Float32).alias("score"))
             .select(pl.col(col).alias("token"), "score"))
        res[kind] = t
    return res


def global_sims(s1, s23, cand, jobs=4, chunk=5_000_000):
    """Name / address token-set similarity for ALL candidate pairs (cheap), so candidate-side
    competition ("how many S1s have this exact name?") can be computed over every competitor."""
    n1, n2 = s1["name_core"], s23["name_core"]
    a1, a2 = s1["addr_clean"], s23["addr_clean"]
    qi_all, ci_all = cand["qi"].to_numpy(), cand["cr" if "cr" in cand.columns else "ci"].to_numpy()
    ns, as_ = [], []
    for s in range(0, len(qi_all), chunk):
        qi, ci = qi_all[s:s + chunk], ci_all[s:s + chunk]
        ns.append(_cp(n1.gather(qi).to_list(), n2.gather(ci).to_list(), fuzz.token_set_ratio, jobs))
        bb = a2.gather(ci)
        av = _cp(a1.gather(qi).to_list(), bb.to_list(), fuzz.token_set_ratio, jobs)
        as_.append(np.where((bb == "").to_numpy(), -1, av).astype(np.float32))
    return np.concatenate(ns), np.concatenate(as_)
