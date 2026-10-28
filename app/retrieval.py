"""Hybrid retrieval: BM25 keyword search (always on) + optional dense
embeddings, merged with Reciprocal Rank Fusion."""
import math
import re
from collections import Counter

import httpx
import numpy as np

from . import config

STOP = set("""a an the is are was were be been being am do does did doing have has had of to in on at for
from by with about as into than then so and or but if not no yes it its this that these those there here
i me my we our you your he she they them their what which who whom when where why how can could would
should will shall may might must please tell know want get give any some all also just only very much
many more most other such own same too s t don explain detail details information info like need
work works let us say describe sir madam kindly regarding""".split())

TOKEN_RE = re.compile(r"[a-z0-9ఀ-౿]+")


def stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5)

    def coverage(self, query: list[str], i: int) -> float:
        """Share of the question's (idf-weighted) words found in doc i."""
        q = set(query)
        total = sum(self.idf.get(t, self.unseen_idf) for t in q)
        hit = sum(self.idf[t] for t in q if t in self.docs[i])
        return hit / total if total else 0.0

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in set(query):
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


def embed(texts: list[str]) -> list[np.ndarray] | None:
    """Dense embeddings via an OpenAI-compatible API, or None if not configured / failed."""
    if not config.EMBED_API_KEY or not texts:
        return None
    try:
        r = httpx.post(f"{config.EMBED_BASE_URL.rstrip('/')}/embeddings",
                       headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                       json={"model": config.EMBED_MODEL, "input": texts}, timeout=30)
        r.raise_for_status()
        vecs = [np.array(d["embedding"], dtype=np.float32) for d in r.json()["data"]]
        return [v / (np.linalg.norm(v) or 1) for v in vecs]
    except Exception:
        return None


class Retriever:
    def __init__(self, store):
        self.store = store
        self.rebuild()

    def rebuild(self):
        self.chunks = self.store.all_chunks()
        self.bm25 = BM25([tokenize(f"{c['heading'] or ''} {c['doc_name']} {c['text']}") for c in self.chunks])
        vecs = [np.frombuffer(c["embedding"], dtype=np.float32) if c["embedding"] else None for c in self.chunks]
        self.dense = np.vstack(vecs) if vecs and all(v is not None for v in vecs) else None

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        """Return top-k chunks, each with `score` (BM25) and `rank_score` (fused)."""
        if not self.chunks:
            return []
        qt = tokenize(query)
        bm = self.bm25.scores(qt)
        order_bm = sorted(range(len(bm)), key=lambda i: -bm[i])
        fused = Counter()
        for rank, i in enumerate(order_bm[:20]):
            if bm[i] > 0:
                fused[i] += 1 / (60 + rank)
        if self.dense is not None:
            q = embed([query])
            if q:
                sims = self.dense @ q[0]
                for rank, i in enumerate(np.argsort(-sims)[:20]):
                    fused[int(i)] += 1 / (60 + rank)
        results = []
        for i, rs in fused.most_common(k):
            c = {key: v for key, v in self.chunks[i].items() if key != "embedding"}
            c["score"] = round(bm[i], 3)
            c["coverage"] = round(self.bm25.coverage(qt, i), 3)
            c["rank_score"] = round(rs, 5)
            results.append(c)
        return results
