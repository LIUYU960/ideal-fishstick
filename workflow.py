import re
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field

from search import web_search_tool
from retrieval import load_corpus, build_faiss, hybrid_search, simple_rerank_by_len
from validators import validate_node_output, final_answer_guardrail

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

class GraphState(BaseModel):
    question: str
    route: Literal["rag","web","math","react","compose"] = "rag"
    contexts: List[str] = Field(default_factory=list)
    draft: str = ""
    answer: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)

def RAGAgent(state: GraphState) -> GraphState:
    texts, _ = load_corpus("sample_data")
    index, _ = build_faiss(texts)
    cands = hybrid_search(state.question, texts, index=index, topk=5)
    cands = simple_rerank_by_len(cands)
    contexts = [c["text"] for c in cands[:3]]
    model = init_chat_model("gpt-4o-mini")
    prompt = "다음 컨텍스트를 기반으로 한국어로 간결하고 정확한 답변을 작성하세요.\n\n" + "\n\n---\n\n".join(contexts) + "\n\n질문: " + state.question
    resp = model.invoke(prompt)
    draft = resp.content if hasattr(resp, "content") else str(resp)
    state.contexts = contexts
    state.draft = draft
    return state

def WebSearchAgent(state: GraphState) -> GraphState:
    search = web_search_tool()
    results = search.run(state.question)
    model = init_chat_model("gpt-4o-mini")
    prompt = f"아래 웹검색 결과를 참고하여 최신성에 유의해 한국어로 요약 답변하세요.\n\n검색결과:\n{results}\n\n질문:{state.question}"
    resp = model.invoke(prompt)
    state.draft = resp.content if hasattr(resp, "content") else str(resp)
    return state

def ReActAgent(state: GraphState) -> GraphState:
    # Simplified ReAct: try RAG first, fallback to web
    try:
        s = RAGAgent(state)
    except Exception:
        s = WebSearchAgent(state)
    state.draft = s.draft
    return state

def AnswerComposer(state: GraphState) -> GraphState:
    model = init_chat_model("gpt-4o-mini")
    ctx = "\n\n".join(state.contexts) if state.contexts else "(컨텍스트 없음)"
    prompt = f"""당신은 답변 작성기입니다.
- 한국어로 간결하고 정중하게 답하세요.
- 출처가 있으면 요약해서 같이 제시하세요.
- 확신이 없으면 불확실성을 표시하세요.

[컨텍스트]
{ctx}

[임시 초안]
{state.draft}

[질문]
{state.question}
"""
    resp = model.invoke(prompt)
    state.answer = resp.content if hasattr(resp, "content") else str(resp)
    return state

def NodeValidator(state: GraphState) -> GraphState:
    ok = validate_node_output({"text": state.draft})
    state.meta["node_validation"] = ok
    return state

def FinalValidator(state: GraphState) -> GraphState:
    res = final_answer_guardrail(state.answer)
    state.meta["final_validation"] = res
    if not res.get("ok", True):
        state.answer = state.answer + f"\n\n[검증 노트] {res.get('reason')}"
    return state

def needs_recent(state: GraphState) -> Literal["web", "rag"]:
    if any(t in state.question for t in ["오늘","최신","최근","news","오늘의","price"]):
        return "web"
    return "rag"

def needs_math(state: GraphState) -> Literal["react","rag"]:
    if any(ch in state.question for ch in "+-*/=%0123456789"):
        return "react"
    return "rag"

def failed_validation(state: GraphState) -> Literal["compose", "rag"]:
    ok = state.meta.get("node_validation", {}).get("ok", True)
    return "compose" if ok else "rag"

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("RAGAgent", RAGAgent)
    g.add_node("WebSearchAgent", WebSearchAgent)
    g.add_node("ReActAgent", ReActAgent)
    g.add_node("NodeValidator", NodeValidator)
    g.add_node("AnswerComposer", AnswerComposer)
    g.add_node("FinalValidator", FinalValidator)

    g.add_conditional_edges(START, needs_recent, {"web":"WebSearchAgent", "rag":"RAGAgent"})
    g.add_conditional_edges("RAGAgent", needs_math, {"react":"ReActAgent", "rag":"NodeValidator"})
    g.add_edge("WebSearchAgent", "NodeValidator")
    g.add_edge("ReActAgent", "NodeValidator")
    g.add_conditional_edges("NodeValidator", failed_validation, {"rag":"RAGAgent", "compose":"AnswerComposer"})
    g.add_edge("AnswerComposer", "FinalValidator")
    g.add_edge("FinalValidator", END)
    return g.compile()

GRAPH = build_graph()
model = init_chat_model("gpt-4o")
