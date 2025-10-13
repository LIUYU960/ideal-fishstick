# retrieval.py — Hybrid search with safe fallback
# 有 OPENAI_API_KEY → 使用 OpenAIEmbeddings + FAISS
# 无 OPENAI_API_KEY → 自动退化为 BM25-only

import os
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi


# ---------- 语料加载 ----------
def load_corpus(path: str = ".") -> Tuple[List[str], List[str]]:
    # 你可以改成实际读取文件；这里放演示文本，保证开箱可用
    texts = [
        "RAG는 Retrieval-Augmented Generation의 약자이며 검색과 생성 모델을 결합한다.",
        "LangGraph는 상태 기계로 LLM 워크플로우를 구성할 수 있게 한다.",
        "Hybrid search는 BM25와 임베딩 기반 검색을 결합하여 재현율과 정밀도를 높인다.",
        "ReAct 패턴은 추론과 행위를 결합하여 도구 사용과 계획을 통합한다.",
        "reranking은 후보 문서를 재정렬하여 더 관련성 높은 문서를 상위에 올린다.",
    ]
    sources = [f"mem_{i}" for i in range(len(texts))]
    return texts, sources


# ---------- 构建向量索引（有 key 用向量；无 key 返回 None） ----------
def build_faiss(texts: List[str]):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # 无 Key：不建向量索引，返回 None
        return None, texts

    # 有 Key：使用 OpenAIEmbeddings + FAISS.from_texts（正确 API）
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)
    vs = FAISS.from_texts(texts, emb)   # ✅ 正确用法
    return vs, texts


# ---------- Hybrid Search（BM25 + 向量；向量缺失时自动 BM25-only） ----------
def hybrid_search(query: str, texts: List[str], index=None, topk: int = 5) -> List[Dict]:
    # BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm_scores = bm25.get_scores(query.split())
    bm_ranked = sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)[:topk]
    results = [{"text": texts[i], "score": float(s), "source": "bm25"} for i, s in bm_ranked]

    # 向量检索（当 index 可用时）
    if index is not None:
        try:
            docs = index.similarity_search(query, k=topk)
            for d in docs:
                results.append({"text": d.page_content, "score": 1.0, "source": "vector"})
        except Exception:
            pass  # 出错就忽略，保底 BM25

    # 去重 & 截断
    seen = set()
    dedup = []
    for r in results:
        t = r["text"]
        if t not in seen:
            seen.add(t)
            dedup.append(r)
    return dedup[:topk]


# ---------- 简单 rerank（按长度） ----------
def simple_rerank_by_len(items: List[Dict]) -> List[Dict]:
    return sorted(items, key=lambda x: len(x["text"]), reverse=True)
