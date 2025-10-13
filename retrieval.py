# retrieval.py (BM25-only, no FAISS required)
# Compatible with your current workflow.py imports/calls.

from __future__ import annotations
import os
import glob
from typing import List, Tuple, Dict, Any

from rank_bm25 import BM25Okapi


# ---------------------------
# 1) Load corpus (txt / md)
# ---------------------------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def load_corpus(root: str = ".") -> Tuple[List[str], List[str]]:
    """
    Load .md/.txt under `root` (non-recursive). If none found, return a small demo corpus.
    """
    patterns = [os.path.join(root, "*.md"), os.path.join(root, "*.txt")]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))

    texts, sources = [], []
    for p in paths:
        content = _read_text_file(p)
        if content.strip():
            texts.append(content)
            sources.append(os.path.basename(p))

    if not texts:
        # Fallback demo corpus so the app always runs
        demo = [
            "RAG(Retrieval-Augmented Generation)은 검색과 생성 모델을 결합하여 최신 정보 기반 답변을 생성한다.",
            "Hybrid search는 전통 BM25와 임베딩 기반 검색을 결합해 정밀도와 재현율을 향상시킨다.",
            "LangGraph는 상태 기반 그래프 워크플로우와 조건부 에지를 제공한다.",
            "ReAct 패턴은 추론과 행동을 반복하면서 도구 호출과 증거 축적을 수행한다.",
        ]
        texts = demo
        sources = [f"demo_{i+1}" for i in range(len(demo))]
    return texts, sources


# --------------------------------
# 2) Vector index stub (no FAISS)
# --------------------------------
def build_faiss(texts: List[str], model: str = "text-embedding-3-small"):
    """
    Kept for compatibility. No FAISS here -> return (None, None).
    """
    return None, None


# -------------------------
# 3) BM25 search (top-k)
# -------------------------
def _bm25_rank(query: str, texts: List[str], topk: int = 8):
    # Simple whitespace tokenization (works reasonably for demo in multi-lingual)
    corpus_tokens = [t.split() for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = query.split()
    scores = bm25.get_scores(q_tokens)  # numpy array-like
    # indices sorted by score desc
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [(i, float(scores[i])) for i in idxs]


def hybrid_search(
    query: str,
    texts: List[str],
    index=None,            # ignored (for compatibility)
    emb_model: str = "",   # ignored
    topk: int = 8,
) -> List[Dict[str, Any]]:
    """
    BM25-only ranking (no vector search). Returns list of dicts:
    {id, rank, text, score, score_bm25, score_vec, source}
    """
    bm25_pairs = _bm25_rank(query, texts, topk=topk)

    results: List[Dict[str, Any]] = []
    for rank, (doc_id, s_bm25) in enumerate(bm25_pairs, start=1):
        results.append({
            "id": int(doc_id),
            "rank": int(rank),
            "text": texts[doc_id],
            "score": float(s_bm25),       # fused score == bm25 score in this minimal version
            "score_bm25": float(s_bm25),
            "score_vec": 0.0,             # no vector component
            "source": f"doc_{doc_id}",
        })
    # If no texts (unlikely due to fallback), return empty list
    return results


# --------------------------
# 4) Simple re-ranking util
# --------------------------
def simple_rerank_by_len(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Light heuristic: penalize extremely short/long chunks slightly.
    """
    if not candidates:
        return []

    def penalize(text: str) -> float:
        n = len(text)
        if n < 120:
            return -0.08
        if n > 1800:
            return -0.05
        return 0.0

    rescored = []
    for c in candidates:
        adj = c.get("score", 0.0) + penalize(c.get("text", ""))
        d = dict(c)
        d["score_adj"] = adj
        rescored.append(d)

    rescored.sort(key=lambda x: x.get("score_adj", 0.0), reverse=True)
    return rescored
