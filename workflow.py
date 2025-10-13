# workflow.py
# LangGraph workflow with explicit OpenAI API key usage.

from __future__ import annotations

import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# ---------- Global model (explicit api_key) ----------

def init_chat_model(name: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with an explicit API key.
    The key should be provided via env var OPENAI_API_KEY (set in Streamlit Secrets).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    # Do not proceed with empty key to avoid confusing 401s later.
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in Streamlit Secrets.")
    return ChatOpenAI(model=name, temperature=temperature, api_key=api_key)


model: ChatOpenAI = init_chat_model("gpt-4o", temperature=0.0)


# ---------- Graph state ----------

class GraphState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    route: str  # "web" or "rag"


# ---------- Nodes ----------

def router(state: GraphState) -> GraphState:
    """
    Route to 'web' or 'rag' based on the question.
    Simple keyword routing to demonstrate add_conditional_edges.
    """
    q = (state.get("question") or "").lower()
    web_keywords = ["web", "google", "bing", "搜索", "검색", "网址", "http", "https"]
    route = "web" if any(k in q for k in web_keywords) else "rag"
    return {"route": route}


def web_agent(state: GraphState) -> GraphState:
    """
    Web search placeholder agent.
    We do not actually hit the web here—just instruct the model to answer generally.
    """
    question = state.get("question", "")
    prompt = (
        "You are a helpful assistant. The user asked a question that seems to require web search, "
        "but this minimal build cannot browse the internet. Provide a general, helpful answer "
        "and clearly state that no live web search was performed.\n\n"
        f"Question: {question}\n\nAnswer thoughtfully:"
    )
    resp = model.invoke(prompt)
    return {"answer": getattr(resp, "content", str(resp))}


def rag_agent(state: GraphState) -> GraphState:
    """
    Minimal RAG agent. Uses (optional) `context` if present; otherwise answers directly.
    """
    question = state.get("question", "")
    context = state.get("context", "")
    prompt = (
        "You are a precise assistant. If the context below is relevant, use it; "
        "otherwise answer from general knowledge. Be concise and correct.\n\n"
        f"Context:\n{context or '[no additional context]'}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    resp = model.invoke(prompt)
    return {"answer": getattr(resp, "content", str(resp))}


def validator(state: GraphState) -> GraphState:
    """
    Very small 'validation' step. If the answer looks too short,
    ask the model to expand and add one concrete detail.
    """
    answer = (state.get("answer") or "").strip()
    if len(answer) >= 40:
        return {"answer": answer}

    fix_prompt = (
        "The following answer is too short. Rewrite it to be clear, complete, and actionable. "
        "Add one concrete detail or example, but keep it under 120 words.\n\n"
        f"Original answer:\n{answer}\n\nImproved answer:"
    )
    resp = model.invoke(fix_prompt)
    return {"answer": getattr(resp, "content", str(resp))}


# ---------- Build graph ----------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router)
    graph.add_node("web_agent", web_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("validator", validator)

    graph.set_entry_point("router")

    # Conditional edges from router -> either web_agent or rag_agent
    def route_selector(s: GraphState) -> str:
        return s.get("route", "rag")

    graph.add_conditional_edges(
        "router",
        route_selector,
        {
            "web": "web_agent",
            "rag": "rag_agent",
        },
    )

    # Both agents go to validator, then END
    graph.add_edge("web_agent", "validator")
    graph.add_edge("rag_agent", "validator")
    graph.add_edge("validator", END)

    return graph.compile()


# Compiled graph exported for app_streamlit.py
GRAPH = build_graph()
