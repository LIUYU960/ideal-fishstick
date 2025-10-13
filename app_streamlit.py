# app_streamlit.py
import os, re, streamlit as st
from dotenv import load_dotenv

# 不要用 .env 覆盖 Streamlit Secrets
load_dotenv(override=False)

# ✅ 正确导入：同目录下的 workflow.py 导出 GRAPH
from workflow import GRAPH

def _sanitize(name: str):
    v = os.getenv(name, "")
    if not v:
        return
    v = v.strip()
    v = re.sub(r"[^\x00-\x7F]+", "", v)  # 删除非 ASCII
    os.environ[name] = v

# 清理并设定 OPENAI_API_KEY（来自 Secrets）
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"].strip()
_sanitize("OPENAI_API_KEY")

st.set_page_config(page_title="LangChain/LangGraph RAG", page_icon="🔎", layout="wide")
st.markdown("### 🔎 LangChain/LangGraph 기반 RAG 챗봇")

question = st.text_input("询问 / Ask（韩/中/英均可）")
if st.button("发送 / Ask") and question.strip():
    with st.spinner("Running graph..."):
        try:
            result = GRAPH.invoke({"question": question.strip()})
            answer = result.get("answer", "(no answer)")
            st.success("回答 / Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Graph 执行出错：{type(e).__name__}: {e}")
