import os
import importlib
import streamlit as st

st.set_page_config(page_title="LIUYU RAG LangGraph Chatbot", page_icon="💬")

# 安全导入 GRAPH，避免 workflow.py 出错时整页崩溃
try:
    workflow = importlib.import_module("workflow")
    GRAPH = getattr(workflow, "GRAPH")
except Exception as e:
    st.error("加载 workflow.py 时出错：\n\n" + repr(e))
    st.info("请检查 workflow.py 是否粘贴完整（括号/引号成对、无奇怪字符）。")
    st.stop()

st.title("LIUYU — RAG Chatbot (LangGraph) 💬")
st.write("现在默认走【正常对话模式】：不在答案里附链接/引用。")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "- 设置 `OPENAI_API_KEY` → 用 OpenAI，回复更自然\n"
        "- 不设置也能运行（本地兜底）\n"
    )
    st.text_input("OPENAI_API_KEY (optional)", type="password", key="openai_key")
    if st.session_state.get("openai_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_key"]
    show_debug = st.checkbox("显示 Evidence / Debug（开发调试用）", value=False)

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("来聊天吧…（中文/한국어/English）")

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

        if show_debug:
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




