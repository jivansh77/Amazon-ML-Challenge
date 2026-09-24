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


def topk_blocked(Q, C, k, q_chunk=2048, c_chunk=500_000):
    """Exact top-k inner product of rows of Q against rows of C, blocked on both sides so the
    similarity tile stays small (q_chunk x c_chunk). Q, C: torch tensors on the same device."""
    import torch
    k = min(k, C.shape[0])
    out_s, out_i = [], []
    for qs in range(0, Q.shape[0], q_chunk):
        q = Q[qs:qs + q_chunk]
        best_s = best_i = None
        for cs in range(0, C.shape[0], c_chunk):
            sim = q @ C[cs:cs + c_chunk].T
            s, i = torch.topk(sim, min(k, sim.shape[1]), dim=1)
            i = i + cs
            if best_s is None:
                best_s, best_i = s, i
            else:
                s = torch.cat([best_s, s], 1)
                i = torch.cat([best_i, i], 1)
                best_s, j = torch.topk(s, k, dim=1)
                best_i = torch.gather(i, 1, j)
        out_s.append(best_s.float().cpu())
        out_i.append(best_i.cpu())
    return torch.cat(out_s).numpy(), torch.cat(out_i).numpy()


def topk_by_country(q_emb, q_country, c_emb, c_country, k, device="cuda"):
    import torch
    out = []
    q_country, c_country = np.asarray(q_country), np.asarray(c_country)
    for ctry in np.unique(q_country):
        qi = np.where(q_country == ctry)[0]
        ci = np.where(c_country == ctry)[0]
        if len(ci) == 0:
            continue
        C = torch.from_numpy(c_emb[ci]).to(device)
        Q = torch.from_numpy(q_emb[qi]).to(device)
        if device == "cpu":
            C, Q = C.float(), Q.float()
        sc, ix = topk_blocked(Q, C, k)
        kk = sc.shape[1]
        out.append(pl.DataFrame({
            "qi": np.repeat(qi, kk).astype(np.int32),
            "ci": ci[ix.ravel()].astype(np.int32),
            "dense_score": sc.ravel().astype(np.float32),
            "dense_rank": np.tile(np.arange(1, kk + 1), len(qi)).astype(np.float32),
        }))
        del C, Q
        if device != "cpu":
            torch.cuda.empty_cache()
    return pl.concat(out)
