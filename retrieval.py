import os, glob
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

try:
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    FAISS = None
    OpenAIEmbeddings = None

def load_corpus(folder: str) -> Tuple[List[str], List[str]]:
    texts, paths = [], []
    for p in glob.glob(os.path.join(folder, "*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())
        paths.append(p)
    return texts, paths

def bm25_candidates(query: str, texts: List[str], k: int = 5):
    tokens = [t.split() for t in texts]
    bm25 = BM25Okapi(tokens)
    scores = bm25.get_scores(query.split())
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:k]
    return ranked

def build_faiss(texts: List[str]):
    if OpenAIEmbeddings is None or FAISS is None:
        return None, None
    emb = OpenAIEmbeddings()  # requires OPENAI_API_KEY
    vectors = emb.embed_documents(texts)
    # FAISS convenience: build from (text, vector) pairs
    index = FAISS.from_embeddings(list(zip(texts, vectors)))
    doc_ids = list(range(len(texts)))
    return index, doc_ids

def vector_candidates(query: str, index, k: int = 5):
    if index is None:
        return []
    docs = index.similarity_search(query, k=k)
    # map back by order; FAISS wrapper may not expose ids cleanly
    return [(i, float(doc.metadata.get("score", 1.0))) for i, doc in enumerate(docs)]

def hybrid_search(query: str, texts: List[str], index=None, topk: int = 5) -> List[Dict[str, Any]]:
    bm = bm25_candidates(query, texts, k=topk*2)
    vv = vector_candidates(query, index, k=topk*2)
    score_map = {}
    for i, s in bm:
        score_map[i] = score_map.get(i, 0.0) + float(s)
    for i, s in vv:
        score_map[i] = score_map.get(i, 0.0) + float(s)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = [{"doc_index": i, "score": sc, "text": texts[i]} for i, sc in ranked]
    return results

def simple_rerank_by_len(cands: List[Dict[str, Any]]):
    scored = []
    for c in cands:
        L = len(c["text"])
        score = -abs(L - 500)  # sweet spot around 500 chars
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]