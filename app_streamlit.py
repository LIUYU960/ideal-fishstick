import os
import streamlit as st
from workflow import GRAPH

st.set_page_config(page_title="LIUYU RAG LangGraph Chatbot", page_icon="💬")

st.title("LIUYU — RAG Chatbot (LangGraph) 💬")
st.write("LangChain + LangGraph + Streamlit. 웹검색/리트리브/리랭크/ReAct/검증 포함.")

with st.sidebar:
    st.header("Settings")
    st.markdown("- Uses OpenAI if `OPENAI_API_KEY` is set.\n- Web search via DuckDuckGo (no key).")
    st.text_input("OPENAI_API_KEY (optional)", type="password", key="openai_key")
    if st.session_state.get("openai_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_key"]
    st.markdown("---")
    st.markdown("**Extras:** ReAct, Web search, Reranking")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("질문을 입력하세요 (한국어/中文/English)...")

for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(msg)

if user_input:
    st.session_state["history"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
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
