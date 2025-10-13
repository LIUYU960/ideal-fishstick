# workflow.py — Chat-first mode (no links/citations in the final answer)

import os, re
from typing import TypedDict, List, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# —— 开关：聊天优先（不在答案里放引用/链接）——
CHAT_MODE = True

# 可选依赖（缺了也能跑）
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
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    HAVE_DDG_IMPORTS = True
except Exception:
    HAVE_DDG_IMPORTS = False
try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content, self.metadata = page_content, (metadata or {})
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=800, chunk_overlap=120):
            self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        def split_text(self, text):
            out, i = [], 0
            step = max(1, self.chunk_size - self.chunk_overlap)
            while i < len(text):
                out.append(text[i:i+self.chunk_size]); i += step
            return out

class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: List[str]
    web_results: List[str]
    reranked_docs: List[str]
    draft_answer: str
    final_answer: str
    validation: Dict[str, Any]

def make_llm(model: Optional[str] = None, temperature: float = 0.2):
    if HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    class DummyLLM:
        def invoke(self, prompt):
            return type("Msg", (), {"content": str(prompt)[:500]})
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

def build_text_corpus(docs_path="docs"):
    texts=[]; splitter=RecursiveCharacterTextSplitter(800,120)
    if os.path.isdir(docs_path):
        for root,_,files in os.walk(docs_path):
            for f in files:
                if f.lower().endswith((".md",".txt")):
                    with open(os.path.join(root,f),"r",encoding="utf-8") as fp:
                        for ck in splitter.split_text(fp.read()):
                            texts.append(Document(ck, {"source": f}))
    if not texts: texts=[Document("No docs found. Add files under /docs.", {"source":"empty.md"})]
    return texts

class DummyVS:
    def __init__(self, docs): self.docs=docs
    def similarity_search(self, q, k=8):
        qset=set(re.findall(r"\w+", q.lower()))
        def score(d): return len(qset & set(re.findall(r"\w+", d.page_content.lower())))
        return sorted(self.docs, key=score, reverse=True)[:k]

_VS=None
def get_vs():
    global _VS
    if _VS: return _VS
    docs=build_text_corpus("docs")
    if HAVE_FAISS:
        _VS=FAISS.from_documents(docs, make_embeddings())
    else:
        _VS=DummyVS(docs)
    return _VS

def _safe_ddg(max_results=4):
    if not HAVE_DDG_IMPORTS: raise ImportError("ddg imports missing")
    try:
        wrap=DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        return DuckDuckGoSearchRun(api_wrapper=wrap)
    except Exception as e:
        raise ImportError("duckduckgo-search unavailable: %s" % e)

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

def synthesize_node(state:RAGState)->RAGState:
    use_llm=(HAVE_OPENAI and os.environ.get("OPENAI_API_KEY"))
    q=state.get("question","").strip()
    ctx="\n\n".join(state.get("reranked_docs",[]) + ([] if CHAT_MODE else state.get("web_results",[])))

    if use_llm:
        llm=make_llm()
        sys=("You are a friendly assistant. Answer directly and naturally in the user's language. "
             "Do not include citations, brackets, or URLs unless the user asks.")
        user=f"Question: {q}\n\nContext:\n{ctx}\n\nAnswer:"
        try: out=llm.invoke(sys+"\n\n"+user).content
        except Exception as e: out=f"(synthesize failed: {e})"
        state["final_answer"]=out; return state

    # 无 LLM 的聊天兜底
    lang=_detect_lang(q)
    if lang=="zh":
        rule=_simple_cn2ko_rule(q)
        if rule: state["final_answer"]=rule; return state
        greet="好的，这是根据资料给你的简要回答："
    elif lang=="ko":
        greet="좋아요. 자료를 바탕으로 간단히 답변드릴게요:"
    else:
        greet="Alright—here's a brief answer based on the docs:"

    points=[]
    for d in state.get("reranked_docs", [])[:2]:
        txt=d.split("::",1)[-1].strip().splitlines()[0]
        points.append("• "+txt[:120])
    state["final_answer"]=greet+"\n"+"\n".join(points or ["• (no internal docs matched)"])
    return state

def validate_node(state:RAGState)->RAGState:
    # 聊天模式下：只要答案非空就 PASS，避免循环
    if CHAT_MODE:
        state.setdefault("validation",{})["final"]="PASS" if state.get("final_answer","").strip() else "FAIL"
        return state
    # 非聊天模式（保留旧逻辑，若你以后想恢复带引用的答案）
    llm=make_llm()
    prompt=("Validator: PASS if the answer addresses the question and includes [1] or [web]. "
            "Else FAIL. Respond with exactly PASS or FAIL.\n\n"
            f"Q: {state.get('question','')}\nA: {state.get('final_answer','')}\nResult:")
    try: res=llm.invoke(prompt).content.strip().upper()
    except Exception: res="PASS" if any(tag in state.get("final_answer","") for tag in ("[1]","[web]")) else "FAIL"
    state.setdefault("validation",{})["final"]="PASS" if "PASS" in res else "FAIL"
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





