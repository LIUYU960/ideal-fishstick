# workflow.py — Chat-first RAG with safe splitter, no external Document deps,
# and graceful downgrade on LLM quota errors.

import os, re
from typing import TypedDict, List, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END

CHAT_MODE = True  # 最终答案不附链接/引用

# --------- 可选依赖（都缺也能跑） ---------
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAVE_ST = True
except Exception:
    HAVE_ST = False
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    HAVE_DDG_IMPORTS = True
except Exception:
    HAVE_DDG_IMPORTS = False

# --------- 自定义极简文档类型（避免外部 Document 兼容问题）---------
class Doc:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# ---------------- App State ----------------
class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: List[str]
    web_results: List[str]
    reranked_docs: List[str]
    draft_answer: str
    final_answer: str
    validation: Dict[str, Any]

# ---------------- LLM helpers ----------------
def make_llm(model: Optional[str] = None, temperature: float = 0.2):
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                          temperature=temperature)
    class DummyLLM:
        def invoke(self, prompt):
            # 返回自然语言，不回显系统前缀
            return type("Msg", (), {"content": str(prompt).split("\nAnswer:",1)[-1].strip()[:800]})
    return DummyLLM()

def make_embeddings():
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if HAVE_ST:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    class HashEmb:
        def embed_documents(self, texts): return [self._e(t) for t in texts]
        def embed_query(self, text): return self._e(text)
        def _e(self, t):
            import math
            v=[0.0]*256
            for i,ch in enumerate(t.encode("utf-8")): v[i%256]+=((ch%23)-11)/11.0
            n=math.sqrt(sum(x*x for x in v)) or 1.0
            return [x/n for x in v]
    return HashEmb()

# ---------------- Safe splitter ----------------
def _safe_split(text, chunk_size=800, overlap=120):
    """安全分段：不依赖 langchain_text_splitters。"""
    parts=[]
    for para in text.split("\n\n"):
        if len(para)<=chunk_size:
            parts.append(para)
        else:
            lines=para.split("\n")
            buf=""
            for ln in lines:
                if len(buf)+len(ln)+1<=chunk_size:
                    buf=(buf+"\n"+ln) if buf else ln
                else:
                    if buf: parts.append(buf); buf=""
                    step=max(1, chunk_size-overlap)
                    for i in range(0, len(ln), step):
                        parts.append(ln[i:i+chunk_size])
            if buf: parts.append(buf)
    final=[]
    step=max(1, chunk_size-overlap)
    for p in parts:
        if len(p)<=chunk_size: final.append(p)
        else:
            for i in range(0, len(p), step):
                final.append(p[i:i+chunk_size])
    return [c for c in final if c.strip()]

# ---------------- Vector store / Retriever ----------------
def build_text_corpus(docs_path="docs"):
    texts=[]
    if os.path.isdir(docs_path):
        for root,_,files in os.walk(docs_path):
            for f in files:
                if f.lower().endswith((".md",".txt")):
                    with open(os.path.join(root,f),"r",encoding="utf-8") as fp:
                        for ck in _safe_split(fp.read(), 800, 120):
                            texts.append(Doc(ck, {"source": f}))
    if not texts:
        texts=[Doc("No docs found. Add files under /docs.", {"source":"empty.md"})]
    return texts

class DummyVS:
    def __init__(self, docs: List[Doc]): self.docs=docs
    def similarity_search(self, q: str, k: int = 8):
        qset=set(re.findall(r"\w+", q.lower()))
        def score(d: Doc): return len(qset & set(re.findall(r"\w+", d.page_content.lower())))
        return sorted(self.docs, key=score, reverse=True)[:k]

_VS=None
def get_vs():
    global _VS
    if _VS: return _VS
    _VS = DummyVS(build_text_corpus("docs"))  # 使用内置检索器，避免外部依赖
    return _VS

def _safe_ddg(max_results=4):
    if not HAVE_DDG_IMPORTS: raise ImportError("ddg imports missing")
    try:
        wrap=DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        return DuckDuckGoSearchRun(api_wrapper=wrap)
    except Exception as e:
        raise ImportError("duckduckgo-search unavailable: %s" % e)

# ---------------- Smalltalk utilities ----------------
def _smalltalk_reply(q: str) -> Optional[str]:
    s=q.strip().lower()
    if re.search(r"(你.?好|您.?好|你好吗|在吗|早上好|下午好|晚上好)", q):
        return "我很好，谢谢！你呢？有什么我可以帮你的吗？"
    if re.search(r"(안녕|안녕하세요|안녕하십니까|잘 지내|좋은\s?아침|만나서 반갑)", q):
        return "저는 잘 지내요, 감사합니다! 무엇을 도와드릴까요?"
    if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening)|how are you)\b", s):
        return "I'm great, thanks! How can I help you today?"
    return None

def _detect_lang(s):
    for ch in s:
        if u"\u4e00"<=ch<=u"\u9fff": return "zh"
        if u"\uac00"<=ch<=u"\ud7a3": return "ko"
    return "en"

def _simple_cn2ko_rule(q):
    m=re.search(r"(.*?)(用)?韩语怎么说", q)
    if not m: return None
    term=m.group(1).strip()
    mapping={"中文":"중국어","韩语":"한국어","英语":"영어","日语":"일본어","谢谢":"감사합니다","你好":"안녕하세요","对不起":"죄송합니다"}
    return f"“{term}”的韩语是 **{mapping.get(term,'（建议使用 LLM 翻译）')}**。"

# ---------------- Nodes ----------------
def retrieve_node(state:RAGState)->RAGState:
    docs=get_vs().similarity_search(state["question"], k=8)
    state["retrieved_docs"]=[f"[{i+1}] {d.metadata.get('source','')} :: {d.page_content}" for i,d in enumerate(docs)]
    state["validation"]={"coverage": sum(len(d.page_content) for d in docs[:3])}
    return state

def rerank_node(state:RAGState)->RAGState:
    docs=state.get("retrieved_docs",[])
    if not docs:
        state["reranked_docs"]=[]; state.setdefault("validation",{})["confidence"]=0.0; return state
    llm=make_llm()
    prompt=("Pick the best 3 doc indices for the question.\n\n"
            f"Q: {state['question']}\nDocs:\n"+"\n".join(docs[:8])+"\n\nReturn: 1,2,3")
    try:
        nums=re.findall(r"\d+", llm.invoke(prompt).content); idx=[int(n) for n in nums[:3]] or [1,2,3]
    except Exception:
        idx=[1,2,3]
    state["reranked_docs"]=[docs[i-1] for i in idx if 1<=i<=len(docs)]
    state["validation"]["confidence"]=0.2*len(state["reranked_docs"])+(1.0 if len(" ".join(state["reranked_docs"]))>500 else 0.0)
    return state

def websearch_node(state:RAGState)->RAGState:
    try: tool=_safe_ddg(4)
    except Exception: state["web_results"]=[]; return state
    try:
        raw=str(tool.invoke(state["question"]))
        state["web_results"]=[ln.strip() for ln in raw.split("\n") if ln.strip()][:4]
    except Exception: state["web_results"]=[]
    return state

def react_node(state:RAGState)->RAGState:
    q=state["question"]; vs=get_vs()
    internal="\n".join(getattr(d,"page_content","")[:400] for d in vs.similarity_search(q,k=4))
    need_web=len(internal)<200; web_text=""
    if need_web:
        try: web_text="\n".join(str(_safe_ddg(3).invoke(q)).split("\n")[:5])
        except Exception: pass
    llm=make_llm()
    prompt=("You are a ReAct-style assistant. Think briefly and answer naturally.\n"
            f"Question: {q}\n\nInternal notes:\n{internal}\n\n"
            + (f"Web notes:\n{web_text}\n\n" if web_text else "")
            + "Reply in the user's language. Be conversational. Do NOT include citations or URLs.")
    try: out=llm.invoke(prompt).content
    except Exception as e: out=f"(react failed: {e})"
    state["draft_answer"]=out; return state

def synthesize_node(state: RAGState) -> RAGState:
    # 小聊直出
    q = state.get("question", "").strip()
    st_reply = _smalltalk_reply(q)
    if st_reply:
        state["final_answer"] = st_reply
        return state

    use_llm = bool(HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"))
    ctx = "\n\n".join(state.get("reranked_docs", []))  # CHAT_MODE：不拼web

    # 优先尝试 LLM —— 失败则静默降级
    if use_llm:
        llm = make_llm()
        sys = ("You are a friendly assistant. Answer directly and naturally in the user's language. "
               "Do not include citations or URLs unless the user asks.")
        user = f"Question: {q}\n\nContext:\n{ctx}\n\nAnswer:"
        try:
            out = llm.invoke(sys + "\n\n" + user).content
            state["final_answer"] = out
            return state
        except Exception as e:
            # 不把错误暴露给最终用户；仅记录到 validation 里，侧边栏可见
            state.setdefault("validation", {})["llm_error"] = f"{type(e).__name__}: {str(e)[:180]}"
            # 继续兜底

    # —— 无 LLM 兜底（离线）——
    lang = _detect_lang(q)
    rule = _simple_cn2ko_rule(q) if lang == "zh" else None
    if rule:
        state["final_answer"] = rule
        return state

    if lang == "zh":
        greet, tail = "好的，我来直接回答：", ""
    elif lang == "ko":
        greet, tail = "좋아요. 바로 답변드릴게요:", ""
    else:
        greet, tail = "Sure — here’s a direct answer:", ""

    points = []
    for d in state.get("reranked_docs", [])[:2]:
        txt = d.split("::", 1)[-1].strip().splitlines()[0]
        points.append("• " + txt[:140])
    state["final_answer"] = greet + ("\n" + "\n".join(points) if points else "")
    return state

def validate_node(state:RAGState)->RAGState:
    # 聊天模式：答案非空即 PASS，避免循环
    state.setdefault("validation",{})["final"]="PASS" if state.get("final_answer","").strip() else "FAIL"
    return state

def route_after_rerank(state:RAGState)->Literal["web","synth"]:
    conf=float(state.get("validation",{}).get("confidence",0.0))
    return "web" if (not CHAT_MODE and conf<1.2) else "synth"

def route_after_validate(state:RAGState)->Literal["done","react"]:
    return "done" if state.get("validation",{}).get("final","FAIL")=="PASS" else "react"

def build_graph():
    g=StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("web", websearch_node)
    g.add_node("react", react_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("validate", validate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve","rerank")
    g.add_conditional_edges("rerank", route_after_rerank, {"web":"web","synth":"synthesize"})
    g.add_edge("web","react")
    g.add_edge("react","synthesize")
    g.add_edge("synthesize","validate")
    g.add_conditional_edges("validate", route_after_validate, {"done":END,"react":"react"})
    return g.compile()

GRAPH = build_graph()





