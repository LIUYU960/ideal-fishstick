# workflow.py —— LangGraph RAG with robust fallbacks (no hard deps; safe on Streamlit Cloud)

import os
import re
from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END

# --- Optional imports with fallbacks ---
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

# DuckDuckGo（可选，未安装也不会崩）
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    HAVE_DDG_IMPORTS = True
except Exception:
    HAVE_DDG_IMPORTS = False

# 文档与分块（若不可用则提供极简替代）
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
            self.chunk_size = chunk_size; self.chunk_overlap = chunk_overlap
        def split_text(self, text: str):
            chunks, n, k, o = [], len(text), self.chunk_size, self.chunk_overlap
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
def make_llm(model: str | None = None, temperature: float = 0.0):
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                          temperature=temperature)
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
        def embed_documents(self, texts: List[str]): return [self._e(t) for t in texts]
        def embed_query(self, text: str): return self._e(text)
        def _e(self, t: str):
            import math
            v = [0.0]*256
            for i, ch in enumerate(t.encode("utf-8")):
                v[i % 256] += ((ch % 23) - 11) / 11.0
            n = math.sqrt(sum(x*x for x in v)) or 1.0
            return [x/n for x in v]
    return HashEmb()

# ------------------ Vector store / Retriever ------------------
def build_text_corpus(docs_path: str = "docs"):
    texts: List[Document] = []
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
    def __init__(self, docs: List[Document]): self.docs = docs
    def similarity_search(self, q: str, k: int = 8):
        qset = set(re.findall(r"\w+", q.lower()))
        def score(d: Document):
            words = set(re.findall(r"\w+", d.page_content.lower()))
            return len(qset & words)
        return sorted(self.docs, key=score, reverse=True)[:k]

_VS = None
def get_vs():
    global _VS
    if _VS is not None: return _VS
    docs = build_text_corpus("docs")
    if HAVE_FAISS:
        embeddings = make_embeddings()
        _VS = FAISS.from_documents(docs, embeddings)
    else:
        _VS = DummyVS(docs)
    return _VS

# --------- helper: safe ddg tool creator (handles missing dependency) ----------
def _safe_ddg(max_results=4):
    if not HAVE_DDG_IMPORTS:
        raise ImportError("langchain_community DuckDuckGo modules not importable")
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        return DuckDuckGoSearchRun(api_wrapper=wrapper)
    except Exception as e:
        # 未安装 duckduckgo-search 时会抛错，这里向上抛给调用方做降级
        raise ImportError(f"duckduckgo-search unavailable: {e}")

# ------------------ Nodes ------------------
def retrieve_node(state: RAGState) -> RAGState:
    q = state["question"]
    vs = get_vs()
    docs = vs.similarity_search(q, k=8)
    state["retrieved_docs"] = [f"[{i+1}] {d.metadata.get('source','')} :: {d.page_content}"
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
        f"Question: {state['question']}\n"
        "Documents:\n" + "\n".join(docs[:8]) + "\n\nReturn: "
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
        state["web_results"] = [f"(web search init failed: {e})"]
        return state
    q = state["question"]
    try:
        raw = tool.invoke(q)
        results = [ln.strip() for ln in str(raw).split("\n") if ln.strip()]
        state["web_results"] = results[:4]
    except Exception as e:
        state["web_results"] = [f"(web search failed: {e})"]
    return state

def react_node(state: RAGState) -> RAGState:
    """
    轻量 ReAct：不用 LangChain Agents（避免 pydantic 依赖链），
    思考→内部检索→必要时网页搜索→综合出草稿答案。
    """
    q = state["question"]
    vs = get_vs()
    # Thought 1: 内部检索
    docs = vs.similarity_search(q, k=4)
    internal = "\n".join(getattr(d, "page_content", str(d))[:400] for d in docs)
    # 是否需要网页搜索
    need_web = len(internal) < 200
    web_text = ""
    if need_web:
        try:
            tool = _safe_ddg(max_results=3)
            web_raw = str(tool.invoke(q))
            web_text = "\n".join(web_raw.split("\n")[:5])
        except Exception:
            web_text = "(web search unavailable)"
    # 组织提示词并让 LLM/兜底生成草稿
    llm = make_llm()
    prompt = (
        "You are a ReAct-style assistant. Think step by step using the observations.\n"
        f"Question: {q}\n\n"
        "Observation[internal]:\n" + internal + "\n\n" +
        ("Observation[web]:\n" + web_text + "\n\n" if web_text else "") +
        "Draft a concise answer in the user's language. Cite internal chunks as [1] and web info as [web] when used."
    )
    try:
        out = llm.invoke(prompt).content
    except Exception as e:
        out = f"(react synth failed: {e})\n\n{prompt[:600]}"
    state["draft_answer"] = out
    return state

def synthesize_node(state: RAGState) -> RAGState:
    """
    有 OpenAI 时用 LLM 生成；否则采用规则化兜底，避免出现 '[(no-LLM)] ...' 的回显。
    """
    use_llm = (HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"))
    context_docs = state.get("reranked_docs", [])
    web = state.get("web_results", [])
    q = state.get("question", "").strip()

    if use_llm:
        llm = make_llm()
        context = "\n\n".join(context_docs + web)
        sys_prompt = (
            "You are a helpful, multilingual RAG assistant. "
            "Answer in the same language as the user's question. "
            "Cite sources inline using [n] for retrieved docs and [web] for web results."
        )
        user = f"Question: {q}\n\nContext:\n{context}\n\nAnswer:"
        try:
            out = llm.invoke(sys_prompt + "\n\n" + user).content
        except Exception as e:
            out = f"(synthesize failed: {e})"
        state["final_answer"] = out
        return state

    # ---- 无 LLM 兜底：用检索内容拼接简洁回答 ----
    def detect_lang(s: str):
        if any('\u4e00' <= ch <= '\u9fff' for ch in s): return "zh"
        if any('가' <= ch <= '힣' for ch in s): return "ko"
        return "en"

    lang = detect_lang(q)
    bullets = []
    for i, d in enumerate(context_docs[:2], 1):
        txt = d.split("::", 1)[-1].strip()
        head = txt.splitlines()[0][:120]
        bullets.append(f"[{i}] {head}")

    if lang == "zh":
        greeting = "你好！基于知识库为你简要回答："
        tail = "如需更多细节可以继续提问。"
    elif lang == "ko":
        greeting = "안녕하세요! 지식베이스를 바탕으로 간단히 답변드립니다:"
        tail = "추가 정보가 필요하면 질문해 주세요."
    else:
        greeting = "Hi! Here is a brief answer based on the knowledge base:"
        tail = "Ask more for details."

    lines = [greeting]
    lines += [f"- {b}" for b in bullets] or ["- (no internal docs matched)"]
    if web and "(unavailable" not in " ".join(web):
        lines.append("- [web] " + web[0][:140])
    lines.append(tail)
    state["final_answer"] = "\n".join(lines)
    return state

def validate_node(state: RAGState) -> RAGState:
    llm = make_llm(model=os.environ.get("VALIDATOR_MODEL", None))
    prompt = (
        "Validator: PASS if the answer addresses the question and includes [1] or [web]. "
        "Else FAIL. Respond with exactly PASS or FAIL.\n\n"
        f"Q: {state['question']}\nA: {state.get('final_answer','')}\nResult:"
    )
    try:
        res = llm.invoke(prompt).content.strip().upper()
    except Exception:
        res = "PASS" if any(tag in state.get("final_answer","") for tag in ("[1]", "[web]")) else "FAIL"
    state.setdefault("validation", {})["final"] = "PASS" if "PASS" in res else "FAIL"
    return state

# ------------------ Routing rules ------------------
def route_after_rerank(state: RAGState) -> Literal["web", "synth"]:
    conf = float(state.get("validation", {}).get("confidence", 0.0))
    return "web" if conf < 1.2 else "synth"

def route_after_validate(state: RAGState) -> Literal["done", "react"]:
    return "done" if state.get("validation", {}).get("final", "FAIL") == "PASS" else "react"

# ------------------ Build graph ------------------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("web", websearch_node)
    graph.add_node("react", react_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", route_after_rerank, {"web": "web", "synth": "synthesize"})
    graph.add_edge("web", "react")
    graph.add_edge("react", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"done": END, "react": "react"})
    return graph.compile()

GRAPH = build_graph()



