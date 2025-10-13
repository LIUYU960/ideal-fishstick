import os
import importlib
import streamlit as st

st.set_page_config(page_title="RAG Chatbot (Minimal Fix)", page_icon="💬")

# 安全导入 workflow.GRAPH，避免 workflow.py 错误导致整页崩溃
try:
    workflow = importlib.import_module("workflow")
    GRAPH = getattr(workflow, "GRAPH")
except Exception as e:
    st.error("加载 workflow.py 时出错：\n\n" + repr(e))
    st.info("请检查 workflow.py 是否完整（括号/引号成对、无奇怪字符）。")
    st.stop()

st.title("RAG Chatbot (Minimal Fix)")
st.write("纯对话体验：正常聊天，不在答案里放引用/链接。")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "- 如设置 `OPENAI_API_KEY` → 使用 OpenAI，回答更自然\n"
        "- 未设置也可用（自动走本地兜底）\n"
    )
    st.text_input("OPENAI_API_KEY (optional)", type="password", key="openai_key")
    if st.session_state.get("openai_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_key"]
    show_debug = st.checkbox("显示 Evidence / Debug（开发调试用）", value=False)

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("输入问题或打个招呼…（中文/한국어/English）")

# 回显历史
for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(msg)

# 新消息处理
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
                # 显示被隐藏的 LLM 错误（比如 429 insufficient_quota）
                err = result.get("validation", {}).get("llm_error")
                if err:
                    st.warning(f"LLM error (hidden from user): {err}")

    st.session_state["history"].append(("assistant", answer))





