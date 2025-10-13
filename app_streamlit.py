import os
import streamlit as st
from workflow import GRAPH

st.set_page_config(page_title="LIUYU RAG LangGraph Chatbot", page_icon="💬")

st.title("LIUYU — RAG Chatbot (LangGraph) 💬")
st.write("LangChain + LangGraph + Streamlit. 支持：检索/重排/网页搜索/ReAct/验证（无依赖也能兜底运行）。")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "- 若设置 `OPENAI_API_KEY` 将使用 OpenAI 模型（质量更好）\n"
        "- 若未设置，将使用本地兜底逻辑（可用但更简单）\n"
        "- 网页搜索使用 DuckDuckGo（requirements.txt 已包含）"
    )
    st.text_input("OPENAI_API_KEY (optional)", type="password", key="openai_key")
    if st.session_state.get("openai_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_key"]
    st.markdown("---")
    st.markdown("**已启用功能**：ReAct、Web search、Reranking、Conditional Edges、Validation")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("输入你的问题…（中文/한국어/English）")

for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(msg)

if user_input:
    st.session_state["history"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = GRAPH.invoke({"question": user_input})
        answer = result.get("final_answer") or result.get("draft_answer") or "(no output)"
        st.markdown(answer)

        with st.expander("🔎 Evidence / Debug"):
            st.markdown("**Reranked docs:**")
            for d in result.get("reranked_docs", []):
                st.code(d[:800])
            st.markdown("**Web results:**")
            for r in result.get("web_results", []):
                st.code(r)
            st.markdown("**Validation:**")
            st.json(result.get("validation", {}))

    st.session_state["history"].append(("assistant", answer))


