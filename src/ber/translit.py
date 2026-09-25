"""Learned Indic-script -> Latin word dictionary.

Built only from the training ground truth: for every matched (S1, S2/S3) pair where the
S2/S3 side contains Indic-script words, each Indic word is counted against the Latin
tokens of the S1 side (same field). The Latin word chosen for an Indic word maximises
    co-occurrence count * (0.5 + similarity(rule_based_transliteration(word), latin))
so that ties between e.g. "uttar"/"pradesh" for "उत्तर" are broken by pronunciation.
Words never seen in training fall back to rule-based transliteration.
"""
import re
from collections import Counter, defaultdict

from rapidfuzz.distance import JaroWinkler

_INDIC_WORD = re.compile(r"[ऀ-ൿ‌‍]+")


def indic_words(s):
    return _INDIC_WORD.findall(s or "")


def fit_indic_dictionary(pairs, min_count=3, min_share=0.3):
    """pairs: iterable of (latin_text, text_with_indic). Returns {indic_word: latin_word}."""
    from .normalize import base_clean, _romanize_indic
    co = defaultdict(Counter)
    tot = Counter()
    for lat, ind in pairs:
        ws = set(indic_words(ind))
        if not ws:
            continue
        lt = set(base_clean(lat).split())
        # Latin tokens already present on the Indic side are not candidates for translation
        lt -= set(base_clean(_INDIC_WORD.sub(" ", ind)).split())
        if not lt:
            continue
        for w in ws:
            tot[w] += 1
            co[w].update(lt)
    out = {}
    for w, c in co.items():
        if tot[w] < min_count:
            continue
        rule = _rule(w)
        best, best_s = None, 0.0
        for l, n in c.most_common(15):
            s = n * (0.5 + JaroWinkler.similarity(rule, l))
            if s > best_s:
                best, best_s = l, s
        if best is not None and c[best] >= min_share * tot[w]:
            out[w] = best
    return out


def _rule(w):
    from .normalize import _INDIC_RE, _romanize_indic
    return _INDIC_RE.sub(_romanize_indic, w).strip()
