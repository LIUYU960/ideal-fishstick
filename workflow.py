import os, re
from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType, Tool

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

def make_llm(model: str = None, temperature: float = 0.0):
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return ChatOpenAI(model=model, temperature=temperature)
    class DummyLLM:
        def invoke(self, prompt):
            return type("Msg", (), {"content": "[(no-LLM fallback)] " + str(prompt)[:300]})
    return DummyLLM()

def make_embeddings():
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return OpenAIEmbeddings(model="text-embedding-3-small")

def build_vectorstore(docs_path: str = "docs"):
    texts = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    for root, _, files in os.walk(docs_path):
        for f in files:
            if f.lower().endswith((".md", ".txt")):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                    content = fp.read()
                for chunk in splitter.split_text(content):
                    texts.append(Document(page_content=chunk, metadata={"source": f}))
    if not texts:
        texts = [Document(page_content="No docs found.", metadata={"source": "empty"})]
    embeddings = make_embeddings()
    return FAISS.from_documents(texts, embeddings)

_VS = None
def get_vs():
    global _VS
    if _VS is None:
        _VS = build_vectorstore()
    return _VS

def retrieve_node(state: RAGState) -> RAGState:
    q = state["question"]
    vs = get_vs()
    docs = vs.similarity_search(q, k=8)
    state["retrieved_docs"] = [f"[{i+1}] {d.metadata.get('source','')} :: {d.page_content}" for i, d in enumerate(docs)]
    coverage = sum(len(d.page_content) for d in docs[:3])
    state["validation"] = {"coverage": coverage}
    return state

def rerank_node(state: RAGState) -> RAGState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        state["reranked_docs"] = []
        return state
    llm = make_llm()
    prompt = (
        "You are a reranker. Given a question and a list of documents in the form [index] text, "
        "return the top 3 indices (comma-separated) that best answer the question.\n\n"
        f"Question: {state['question']}\n"
        f"Documents:\n" + "\n".join(docs[:8]) + "\n\nReturn: "
    )
    try:
        out = llm.invoke(prompt).content
        import re as _re
        nums = _re.findall(r"\d+", out)
        top_idx = [int(n) for n in nums[:3]] if nums else [1, 2, 3][:len(docs)]
    except Exception:
        top_idx = [1, 2, 3][:len(docs)]
    ordered = []
    for i in top_idx:
        if 1 <= i <= len(docs):
            ordered.append(docs[i-1])
    state["reranked_docs"] = ordered
    state["validation"]["confidence"] = 0.2 * len(ordered) + (1.0 if len(" ".join(ordered)) > 500 else 0.0)
    return state

def websearch_node(state: RAGState) -> RAGState:
    q = state["question"]
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=4)
    tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    try:
        lines = tool.invoke(q)
        results = [ln.strip() for ln in str(lines).split("\n") if ln.strip()]
        state["web_results"] = results[:4]
    except Exception as e:
        state["web_results"] = [f"(web search failed: {e})"]
    return state

def react_node(state: RAGState) -> RAGState:
    vs = get_vs()
    def retrieve_tool_run(q: str) -> str:
        docs = vs.similarity_search(q, k=4)
        return "\n".join(d.page_content[:400] for d in docs)

    tools = [
        Tool(
            name="vector_retrieve",
            func=retrieve_tool_run,
            description="Look up internal knowledge chunks relevant to the question."
        ),
        Tool(
            name="web_search",
            func=lambda q: "\n".join(DuckDuckGoSearchAPIWrapper(max_results=3).results(q)),
            description="Search the public web when internal docs are insufficient."
        ),
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
    llm = make_llm(model=os.environ.get("VALIDATOR_MODEL", "gpt-4o-mini"))
    prompt = (
        "Validator: PASS if the answer addresses the question and includes [1] or [web]. "
        "Else FAIL. Respond with exactly PASS or FAIL.\n\n"
        f"Q: {state['question']}\nA: {state.get('final_answer','')}\nResult:"
    )
    try:
        res = llm.invoke(prompt).content.strip().upper()
    except Exception:
        res = "PASS" if any(tag in state.get("final_answer","") for tag in ("[1]", "[web]")) else "FAIL"
    state["validation"]["final"] = "PASS" if "PASS" in res else "FAIL"
    return state

def route_after_rerank(state: RAGState) -> Literal["web", "synth"]:
    conf = float(state.get("validation", {}).get("confidence", 0.0))
    return "web" if conf < 1.2 else "synth"

def route_after_validate(state: RAGState) -> Literal["done", "react"]:
    return "done" if state.get("validation", {}).get("final", "FAIL") == "PASS" else "react"

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
