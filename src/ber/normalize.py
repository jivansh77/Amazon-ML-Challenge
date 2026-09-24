"""Country-agnostic text normalisation for business names and addresses.

Every record gets several normalised variants so that downstream features can compare
names/addresses at different levels of aggressiveness:

name_clean   lowercase, accent-free, OCR-fixed, junk tags removed (legal forms kept)
name_core    name_clean minus titles and legal-form words (the distinctive part)
name_alt     core of the DBA ("t/a") part of the name, if present, otherwise ""
name_legal   the set of legal-form words found (e.g. "llc", "pvt ltd")
addr_clean   lowercase, accent-free address with EN/FR abbreviations expanded
addr_nums    the numeric tokens of the address (house / plot numbers, PIN / ZIP)
addr_words   the non-numeric address tokens
"""
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --- script handling -------------------------------------------------------------------
_INDIC_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI), (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA), (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU), (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
_INDIC_RE = re.compile(r"[ऀ-ൿ]+(?:[\s‌‍]+[ऀ-ൿ]+)*")


def _script_of(ch):
    o = ord(ch)
    for lo, hi, sc in _INDIC_BLOCKS:
        if lo <= o <= hi:
            return sc
    return None


def _romanize_indic(m):
    """Transliterate one run of Indic text to plain Latin (with crude schwa deletion)."""
    s = m.group(0)
    sc = _script_of(s[0])
    try:
        out = transliterate(s, sc, sanscript.IAST)
    except Exception:
        return " "
    out = strip_accents(out).lower()
    # schwa deletion: word-final inherent 'a' after a consonant ("limiteda" -> "limited")
    out = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b", "", out)
    return " " + out + " "


INDIC_DICT = {}   # learned word map (see translit.py); filled by set_indic_dictionary()
_INDIC_WORD = re.compile(r"[\u0900-\u0D7F\u200c\u200d]+")


def set_indic_dictionary(d):
    INDIC_DICT.clear()
    INDIC_DICT.update(d)


def has_indic(s):
    return bool(_INDIC_RE.search(s))


def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# --- OCR-style confusables inside words ("5ecure", "D0ts", "lnformation") -------------
_OCR_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th|er|e|eme|ere)$")


_LEAD_NUM = re.compile(r"^(\d{2,})([a-z]{3,})$")
_SINGLE_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")


def _fix_ocr_token(t):
    if t.isdigit() or t.isalpha() or _ORDINAL.match(t):
        return t
    m = _LEAD_NUM.match(t)             # "50wellesly" -> "50 wellesly" (glued house number)
    if m:
        return m.group(1) + " " + m.group(2)
    n_dig = sum(c.isdigit() for c in t)
    n_alpha = sum(c.isalpha() for c in t)
    if n_alpha >= 2 and n_dig <= 2 and n_alpha > n_dig:
        return t.translate(_OCR_MAP)
    return t


# --- dictionaries (hand-written, no external data) -------------------------------------
TITLES = {"mr", "mrs", "ms", "dr", "shri", "sri", "shree", "smt", "m/s", "ms.", "messrs", "the", "prof",
          "sir", "m s", "mme", "mlle", "m."}
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "l.l.c", "inc", "incorporated", "corp", "corporation",
    "co", "company", "cos", "lp", "llp", "plc", "pllc", "pc", "pa", "ltda", "gmbh", "ag", "bv", "nv",
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "scop", "scp", "selarl", "cie", "compagnie",
    "opc", "trust", "and", "&", "et", "of", "de", "du", "des", "la", "le", "les", "d", "l",
}
LEGAL_FORMS = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
               "co", "company", "lp", "llp", "plc", "pllc", "sarl", "sas", "sasu", "eurl", "sci", "sa",
               "snc", "gmbh", "opc"}
LEGAL_CANON = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp",
               "company": "co"}

ADDR_ABBR = {
    # English (US / India)
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "apt": "apartment", "ste": "suite", "fl": "floor",
    "bldg": "building", "no": "number", "nr": "near", "opp": "opposite", "ngr": "nagar", "mg": "marg",
    "po": "post", "dist": "district", "tq": "taluk", "tal": "taluk", "vill": "village", "hno": "house",
    "twp": "township", "twnship": "township", "cty": "county",
    # French
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "all": "allee", "imp": "impasse",
    "rte": "route", "ch": "chemin", "chem": "chemin", "fg": "faubourg", "fbg": "faubourg", "pte": "porte",
    "qu": "quai", "crs": "cours", "sq.": "square", "res": "residence", "lot": "lotissement", "zi": "zone",
    "za": "zone", "zac": "zone", "bis": "bis", "ter.": "ter", "ste.": "sainte",
}
# "st" is both Street and Saint; canonicalise street/saint/st to the same token.
ADDR_ABBR.update({"st": "st", "str": "st", "street": "st", "saint": "st"})

_JUNK_PATTERNS = [
    re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.I),     # (ID: 93609)
    re.compile(r"#\s*\d{3,}"),                          # #58665
    re.compile(r"\+?\d[\d\s\-]{8,}\d"),                  # phone numbers
    re.compile(r"\bid\s*[:#]\s*\d+", re.I),
]
_DBA = re.compile(r"\s(?:t/a|d/b/a|dba|trading as|a/k/a|aka|o/a)\s", re.I)
_WEB = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|co\.in|biz|info|io|fr|us)\b")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def base_clean(s):
    """Indic -> Latin, accent strip, lowercase, punctuation -> space, OCR token fix."""
    if not s:
        return ""
    if has_indic(s):
        if INDIC_DICT:
            s = _INDIC_WORD.sub(lambda m: " " + INDIC_DICT.get(m.group(0), m.group(0)) + " ", s)
        s = _INDIC_RE.sub(_romanize_indic, s)
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    s = _NON_ALNUM.sub(" ", s)
    s = s.replace("_", " ")
    s = " ".join(_fix_ocr_token(t) for t in s.split())
    # "s c i" / "l l c" (from "S.C.I." / "L.L.C.") -> "sci" / "llc"
    return _SINGLE_RUN.sub(lambda m: m.group(0).replace(" ", ""), s)


def _core(tokens):
    return [t for t in tokens if t not in LEGAL and t not in TITLES]


def normalize_name(raw):
    """Return dict of name variants for one raw business name."""
    s = raw or ""
    for p in _JUNK_PATTERNS:
        s = p.sub(" ", s)
    low = strip_accents(s).lower().strip()
    web = _WEB.match(low)
    alt = ""
    m = _DBA.search(" " + s + " ")
    if m:
        left = (" " + s + " ")[: m.start()]
        right = (" " + s + " ")[m.end():]
        s, alt = right, left          # "X t/a Y": Y is the trading name S1 usually carries
    clean = base_clean(s)
    if web:
        clean = web.group(1).replace("-", " ")
    toks = clean.split()
    # merge "m s" (from "M/s") and "l l c" style fragments
    joined = " ".join(toks).replace("m s ", "ms ").replace("l l c", "llc").replace("p ltd", "pvt ltd")
    toks = joined.split()
    core = _core(toks) or toks
    legal = sorted({LEGAL_CANON.get(t, t) for t in toks if t in LEGAL_FORMS})
    alt_core = " ".join(_core(base_clean(alt).split())) if alt else ""
    return {
        "name_clean": " ".join(toks),
        "name_core": " ".join(core),
        "name_alt": alt_core,
        "name_legal": " ".join(legal),
        "name_is_web": bool(web),
    }


def normalize_address(raw):
    """Return dict of address variants for one raw address string."""
    s = base_clean(raw or "")
    toks = []
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        # "0327" -> "327"; "527-" handled by punctuation stripping
        if t.isdigit():
            t = t.lstrip("0") or "0"
        toks.append(t)
    nums = [t for t in toks if any(c.isdigit() for c in t)]
    words = [t for t in toks if not any(c.isdigit() for c in t)]
    return {"addr_clean": " ".join(toks), "addr_nums": " ".join(nums), "addr_words": " ".join(words)}


def normalize_frame(df, n_jobs=4):
    """Add all normalised columns to a polars frame with business_name / business_address."""
    import polars as pl
    from multiprocessing import Pool
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    if n_jobs > 1 and len(names) > 50_000:
        with Pool(n_jobs) as p:
            nn = p.map(normalize_name, names, chunksize=20_000)
            aa = p.map(normalize_address, addrs, chunksize=20_000)
    else:
        nn = [normalize_name(x) for x in names]
        aa = [normalize_address(x) for x in addrs]
    return pl.concat([df, pl.DataFrame(nn), pl.DataFrame(aa)], how="horizontal")
