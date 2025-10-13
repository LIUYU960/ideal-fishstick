# workflow.py — LangGraph RAG with robust fallbacks (safe on Streamlit Cloud)

import os
import re
from typing import TypedDict, List, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# ---------- Optional imports with fallbacks ----------
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

try:
    from langchain_community.vectorstores import FAISS
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAVE_ST = True
except Exception:
    HAVE_ST = False

# DuckDuckGo (optional)
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    HAVE_DDG_IMPORTS = True
except Exception:
    HAVE_DDG_IMPORTS = False

# Docs & splitters (fallback if missing)
try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=800, chunk_overlap=120):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            chunks = []
            n = len(text)
            k = self.chunk_size
            o = self.chunk_overlap
            i = 0
            while i < n:
                chunks.append(text[i:i+k])
                i += (k - o) if k > o else k
            return chunks

# ------------------ App State ------------------
class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: List[str]
    web_results: List[str]
    reranked_docs: List[str]
    draft_answer: str
    final_answer: str
    validation: Dict[str, Any]

# ------------------ LLM helpers ------------------
def make_llm(model: Optional[str] = None, temperature: float = 0.0):
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        use_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=use_model, temperature=temperature)
    class DummyLLM:
        def invoke(self, prompt):
            txt = str(prompt)
            return type("Msg", (), {"content": "[(no-LLM)] " + txt[:500]})
    return DummyLLM()

def make_embeddings():
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if HAVE_ST:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    class HashEmb:
        def embed_documents(self, texts):
            return [self._e(t) for t in texts]
        def embed_query(self, text):
            return self._e(text)
        def _e(self, t):
            import math
            v = [0.0]*256
            for i, ch in enumerate(t.encode("utf-8")):
                v[i % 256] += ((ch % 23) - 11) / 11.0
            n = math.sqrt(sum(x*x for x in v)) or 1.0
            return [x/n for x in v]
    return HashEmb()

# ------------------ Vector store / Retriever ------------------
def build_text_corpus(docs_path="docs"):
    texts = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    if os.path.isdir(docs_path):
        for root, _, files in os.walk(docs_path):
            for f in files:
                if f.lower().endswith((".md", ".txt")):
                    with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                        content = fp.read()
                    for chunk in splitter.split_text(content):
                        texts.append(Document(page_content=chunk, metadata={"source": f}))
    if not texts:
        texts = [Document("No docs found. Add files under /docs.", {"source": "empty.md"})]
    return texts

class DummyVS:
    def __init__(self, docs):
        self.docs = docs
    def similarity_search(self, q, k=8):
        qset = set(re.findall(r"\w+", q.lower()))
        def score(d):
            words = set(re.findall(r"\w+", d.page_content.lower()))
            return len(qset & words)
        return sorted(self.docs, key=score, reverse=True)[:k]

_VS = None
def get_vs():
    global _VS
    if _VS is not None:
        return _VS
    docs = build_text_corpus("docs")
    if HAVE_FAISS:
        embeddings = make_embeddings()
        _VS = FAISS.from_documents(docs, embeddings)
    else:
        _VS = DummyVS(docs)
    return _VS

# --------- helper: safe ddg tool creator ----------
def _safe_ddg(max_results=4):
    if not HAVE_DDG_IMPORTS:
        raise ImportError("langchain_community DuckDuckGo modules not importable")
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        return DuckDuckGoSearchRun(api_wrapper=wrapper)
    except Exception as e:
        raise ImportError("duckduckgo-search unavailable: %s" % e)

# ------------------ Nodes ------------------
def retrieve_node(state: RAGState) -> RAGState:
    q = state["question"]
    vs = get_vs()
    docs = vs.similarity_search(q, k=8)
    state["retrieved_docs"] = ["[%d] %s :: %s" % (i+1, d.metadata.get("source",""), d.page_content)
                               for i, d in enumerate(docs)]
    coverage = sum(len(d.page_content) for d in docs[:3])
    state["validation"] = {"coverage": coverage}
    return state

def rerank_node(state: RAGState) -> RAGState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        state["reranked_docs"] = []
        state.setdefault("validation", {})["confidence"] = 0.0
        return state
    llm = make_llm()
    prompt = (
        "You are a reranker. Given a question and a list of documents in the form [index] text, "
        "return the top 3 indices (comma-separated) that best answer the question.\n\n"
        "Question: %s\nDocuments:\n%s\n\nReturn: " % (state["question"], "\n".join(docs[:8]))
    )
    try:
        out = llm.invoke(prompt).content
        nums = re.findall(r"\d+", out)
        top_idx = [int(n) for n in nums[:3]] if nums else [1,2,3][:len(docs)]
    except Exception:
        top_idx = [1,2,3][:len(docs)]
    ordered = []
    for i in top_idx:
        if 1 <= i <= len(docs):
            ordered.append(docs[i-1])
    state["reranked_docs"] = ordered
    state["validation"]["confidence"] = 0.2*len(ordered) + (1.0 if len(" ".join(ordered)) > 500 else 0.0)
    return state

def websearch_node(state: RAGState) -> RAGState:
    try:
        tool = _safe_ddg(max_results=4)
    except ImportError:
        state["web_results"] = ["(web search unavailable: install `duckduckgo-search` to enable)"]
        return state
    except Exception as e:
        state["web_results"] = ["(web search init failed: %s)" % e]
        return state
    q = state["question"]
    try:
        raw = tool.invoke(q)
        results = [ln.strip() for ln in str(raw).split("\n") if ln.strip()]
        state["web_results"] = results[:4]
    except Exception as e:
        state["web_results"] = ["(web search failed: %s)" % e]
    return state

def react_node(state: RAGState) -> RAGState:
    q = state["question"]
    vs = get_vs()
    docs = vs.similarity_search(q, k=4)
    internal = "\n".join(getattr(d, "page_content", str(d))[:400] for d in docs)
    need_web = len(internal) < 200
    web_text = ""
    if need_web:
        try:
            tool = _safe_ddg(max_results=3)
            web_raw = str(tool.invoke(q))
            web_text = "\n".join(web_raw.split("\n")[:5])
        except Exception:
            web_text = "(web search unavailable)"
    llm = make_llm()
    prompt = (
        "You are a ReAct-style assistant. Think step by step using the observations.\n"
        "Question: %s\n\nObservation[internal]:\n%s\n\n%s"
        "Draft a concise answer in the user's language. Cite internal chunks as [1] and web info as [web] when used."
        % (q, internal, ("Observation[web]:\n%s\n\n" % web_text) if web_text else "")
    )
    try:
        out = llm.invoke(prompt).content
    except Exception as e:
        out = "(react synth failed: %s)\n\n%s" % (e, prompt[:600])
    state["draft_answer"] = out
    return state

# ---- No LLM fallback helpers ----
def _detect_lang(s):
    for ch in s:
        if u"\u4e00" <= ch <= u"\u9fff":
            return "zh"
        if u"\uac00" <= ch <= u"\ud7a3":
            return "ko"
    return "en"

def _simple_cn2ko_rule(q):
    m = re.se




