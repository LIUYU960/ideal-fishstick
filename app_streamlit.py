import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from workflow import GRAPH, GraphState

load_dotenv()

st.set_page_config(page_title="RAG Chatbot (LangGraph)", page_icon="🔎")
st.title("🔎 LangChain/LangGraph 기반 RAG 챗봇")

with st.sidebar:
    st.markdown("### 설정")
    question = st.text_input("질문을 입력하세요", value="하이브리드 검색과 reranking이 뭐예요?")
    if st.button("질문 보내기", type="primary"):
        st.session_state["go"] = True

if st.session_state.get("go"):
    with st.spinner("그래프 실행 중..."):
        state = GraphState(question=question)
        result = GRAPH.invoke(state.dict())
        st.success("완료!")
        st.markdown("#### 답변")
        st.write(result.get("answer", ""))

        st.markdown("#### 메타데이터/검증 정보")
        st.json(result.get("meta", {}))

        if result.get("contexts"):
            st.markdown("#### 사용된 컨텍스트")
            for i, c in enumerate(result["contexts"], 1):
                with st.expander(f"컨텍스트 {i}"):
                    st.write(c)
