# workflow.py  —— LangGraph RAG with strong fallbacks (safe without duckduckgo-search)

import os
import re
from typing import TypedDict, List, Literal, Dict, Any

from langgraph.graph import StateGraph, END

# --- Optional imports with fallbacks ---
# LLM & Embeddings
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# Vector store (FAISS). If not available, we will use a dummy lexical retriever.
try:
    from langchain_community.vectorstores import FAISS
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# Sentence-Transformers as an optional local embedding fallback
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAVE_ST = True
except Exception:
    HAVE_ST = False

# Web search via DuckDuckGo (optional)
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    HAVE_DDG_IMPORTS = True
except Exception:
    HAVE_DDG_IMPORTS = False

# Common LangChain bits（若不可用则提供极简替代）
try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.agents import initialize_agent, AgentType, Tool
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
            chunks, n, k, o = [], len(text), self.chunk_size, self.chunk_overlap
            i = 0
            while i < n:
                chunks.append(text[i:i+k])
                i += (k - o) if k > o else k
            return chunks
    def initialize_agent(*args, **kwargs):
        class _A:
            def invoke(self, inputs):
                q = inputs.get("input") if isinstance(inputs, dict) else str(inputs)
                return {"output": f"(ReAct fallback: no agent libs) {q[:200]}"}
        return _A()
    class AgentType:
        ZERO_SHOT_REACT_DESCRIPTION = "react"
    class Tool:
        def __init__(self, name, func, description=None):
            self.name, self.func, self.description = name, func, description

# ------------------ App State ------------------
class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: List[str]
    web_results: List[str]
    reranked_docs: List[str]
    draft_answer: str
    final_answer: str
    reasoning: str
    route: str
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
        def embed_documents(self, texts: List[str]):
            return [self._embed(t) for t in texts]
        def embed_query(self, text: str):
            return self._embed(text)
        def _embed(self, t: str):
            import math
            vec = [0.0]*256
            for i, ch in enumerate(t.encode("utf-8")):
                vec[i % 256] += ((ch % 23) - 11) / 11.0
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            return [x / norm for x in vec]
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
    def __init__(self, docs: List[Document]):
        self.docs = docs
    def similarity_search(self, q: str, k: int = 8):
        qset = set(re.findall(r"\w+", q.lower()))
        def score(d: Document):
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

# --------- helper: safe ddg tool creator (handles missing dependency) ----------
def _make_ddg_tool(max_results=4):
    if not HAVE_DDG_IMPORTS:
        raise ImportError("langchain_community DuckDuckGo modules not importable")
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        return DuckDuckGoSearchRun(api_wrapper=wrapper)
    except Exception as e:
        # This catches the case where `duckduckgo_search` package itself is not installed
        raise ImportError(f"duckduckgo-search unavailable: {e}")

# ------------------ Nodes ------------------
def retrieve_node(state: RAGState) -> RAGState:
    q = state["question"]
    vs = get_vs()
    docs = vs.similarity_search(q, k=8)
    state["retrieved_docs"] = [
        f"[{i+1}] {d.metadata.get('source','')} :: {d.page_content}"
        for i, d in enumerate(docs)
    ]
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
        tool = _make_ddg_tool(max_results=4)
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
    vs = get_vs()
    def retrieve_tool_run(q: str) -> str:
        docs = vs.similarity_search(q, k=4)
        return "\n".join(getattr(d, "page_content", str(d))[:400] for d in docs)

    def web_fn(q: str) -> str:
        try:
            tool = _make_ddg_tool(max_results=3)
            return str(tool.invoke(q))
        except Exception:
            return "(web_search tool unavailable: install `duckduckgo-search`)"

    tools = [
        Tool(name="vector_retrieve", func=retrieve_tool_run,
             description="Look up internal knowledge chunks relevant to the question."),
        Tool(name="web_search", func=web_fn,
             description="Search the public web when internal docs are insufficient."),
    ]
    llm = make_llm()
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True, verbose=False,
    )
    try:
        result = agent.invoke({"input": state["question"]})
        output = result["output"] if isinstance(result, dict) else str(result)
    except Exception as e:
        output = f"(ReAct agent failed: {e})"
    state["draft_answer"] = output
    return state

def synthesize_node(state: RAGState) -> RAGState:
    llm = make_llm()
    context = "\n\n".join(state.get("reranked_docs", []) + state.get("web_results", []))
    sys_prompt = (
        "You are a helpful, multilingual RAG assistant. "
        "Answer in the same language as the user's question. "
        "Cite sources inline using [n] for retrieved docs and [web] for web results."
    )
    user = f"Question: {state['question']}\n\nContext:\n{context}\n\nAnswer:"
    try:
        out = llm.invoke(sys_prompt + "\n\n" + user).content
    except Exception as e:
        out = f"(synthesize failed: {e})\n\nContext used:\n{context[:600]}"
    state["final_answer"] = out
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
    graph.add_conditional_edges("rerank", route_after_rerank,
                                {"web": "web", "synth": "synthesize"})
    graph.add_edge("web", "react")
    graph.add_edge("react", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                                {"done": END, "react": "react"})
    return graph.compile()

GRAPH = build_graph()


