from dotenv import load_dotenv
load_dotenv(override=False)  # 避免 .env 覆盖 Secrets

import os, streamlit as st
from openai import OpenAI

API_KEY = st.secrets["OPENAI_API_KEY"].strip()
os.environ["OPENAI_API_KEY"] = API_KEY  # 需要的话，仍设置到环境里

# 最小直连测试（成功则不抛异常）
OpenAI(api_key=API_KEY).models.list()


st.set_page_config(page_title="RAG Chatbot (LangGraph)", page_icon="🔎")
st.title("🔎 LangChain/LangGraph 기반 RAG 챗봇")

# —— 主区输入（不用侧边栏，避免没展开看不到）——
st.markdown("### 询问 / Ask")
question = st.text_input("请输入你的问题（韩/中/英均可）", value="하이브리드 검색과 reranking이 뭐예요?")
run = st.button("发送 / Ask", type="primary")

if run:
    try:
        # 关键：只传最小输入，避免 Pydantic 校验错误
        result = GRAPH.invoke({"question": question})

        st.success("完成！")
        st.markdown("#### 答复 / Answer")
        st.write(result.get("answer", ""))

        st.markdown("#### 验证与元信息 / Validation & Meta")
        st.json(result.get("meta", {}))

        if result.get("contexts"):
            st.markdown("#### 使用到的检索上下文 / Contexts")
            for i, c in enumerate(result["contexts"], 1):
                with st.expander(f"Context {i}"):
                    st.write(c)

    except Exception as e:
        import traceback
        st.error("Graph 执行出错")
        st.code("".join(traceback.format_exc()))
import os, streamlit as st
st.caption("KEY OK" if (os.getenv("OPENAI_API_KEY","").startswith("sk-") and os.getenv("OPENAI_API_KEY","").isascii()) else "NO/INVALID KEY")
from openai import OpenAI
client = OpenAI()
client.models.list()  # 如果这里 401，说明 key 在 OpenAI 侧就是无效/未开通
