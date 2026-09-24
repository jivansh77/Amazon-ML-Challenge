"""Dense retrieval blocking route (GPU).

Encodes "name | address" with a multilingual sentence encoder (intfloat/multilingual-e5-small,
MIT licence) and runs exact inner-product top-k search per country with torch on GPU.
Multilingual encoders handle Indic scripts, typos and spacing variants natively.
"""
import numpy as np
import polars as pl

MODEL = "intfloat/multilingual-e5-small"


def record_text(df):
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


_MODELS = []


def _models():
    """One fp16 model copy per visible GPU (loaded once, reused across calls)."""
    import torch
    from sentence_transformers import SentenceTransformer
    if not _MODELS:
        for i in range(max(torch.cuda.device_count(), 1)):
            m = SentenceTransformer(MODEL, device=f"cuda:{i}")
            m.max_seq_length = 64
            m.half()
            _MODELS.append(m)
    return _MODELS


def encode(texts, batch=512):
    """Data-parallel encoding with one thread per GPU (threads, not processes: no re-import)."""
    from concurrent.futures import ThreadPoolExecutor
    models = _models()
    n = len(models)
    parts = [texts[i * len(texts) // n:(i + 1) * len(texts) // n] for i in range(n)]
    with ThreadPoolExecutor(n) as ex:
        embs = list(ex.map(lambda mp: mp[0].encode(mp[1], batch_size=batch, normalize_embeddings=True,
                                                    convert_to_numpy=True, show_progress_bar=False),
                           zip(models, parts)))
    return np.concatenate(embs).astype(np.float16)


def topk_by_country(q_emb, q_country, c_emb, c_country, k, chunk=8192):
    import torch
    out = []
    q_country, c_country = np.asarray(q_country), np.asarray(c_country)
    for ctry in np.unique(q_country):
        qi = np.where(q_country == ctry)[0]
        ci = np.where(c_country == ctry)[0]
        if len(ci) == 0:
            continue
        C = torch.from_numpy(c_emb[ci]).cuda()
        for s in range(0, len(qi), chunk):
            sub = qi[s:s + chunk]
            Q = torch.from_numpy(q_emb[sub]).cuda()
            sc, ix = torch.topk(Q @ C.T, min(k, len(ci)), dim=1)
            sc, ix = sc.float().cpu().numpy(), ix.cpu().numpy()
            out.append(pl.DataFrame({
                "qi": np.repeat(sub, sc.shape[1]).astype(np.int32),
                "ci": ci[ix.ravel()].astype(np.int32),
                "dense_score": sc.ravel().astype(np.float32),
                "dense_rank": np.tile(np.arange(1, sc.shape[1] + 1), len(sub)).astype(np.float32),
            }))
        del C
        torch.cuda.empty_cache()
    return pl.concat(out)
