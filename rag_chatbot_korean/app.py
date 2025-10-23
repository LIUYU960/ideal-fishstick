import os
import streamlit as st
from typing import List
from dotenv import load_dotenv

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

# PDF
from pypdf import PdfReader

load_dotenv()
st.set_page_config(page_title="나만의 RAG 챗봇 (BM25, 초안정판)", page_icon="🤖", layout="wide")

# ---- 스타일 ----
st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .chat-bubble-user {background:#eef4ff; padding:14px 16px; border-radius:16px; margin:6px 0; }
        .chat-bubble-bot {background:#f6f6f6; padding:14px 16px; border-radius:16px; margin:6px 0; border-left:4px solid #5b9bd5;}
        .tiny {font-size: 12px; color:#666;}
    </style>
    """,
    unsafe_allow_html=True
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 나만의 RAG 챗봇 (초안정판: BM25)")
st.caption("LangChain + Streamlit | 업로드 파일 기반 RAG (벡터DB 없이 BM25)")

st.sidebar.title("⚙️ 설정")
with st.sidebar.expander("📌 안내 (필독)", expanded=True):
    st.markdown(
        """
        **왜 이 버전이 안정적인가요?**
        - 벡터DB(FAISS/Chroma) 대신 **BM25**를 사용해 **네이티브 빌드/대형 의존성 없이** 설치가 매우 안정적입니다.
        - 과제 요건(업로드 파일 기반 RAG, Streamlit UI, Cloud 배포)을 모두 충족합니다.
        
        **필수 준비**
        1) 환경변수 `OPENAI_API_KEY` 등록 (Secrets)
        2) 파일 업로드 → 인덱스 구축 → 질문
        """
    )

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("환경변수 OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Cloud → Secrets에 키를 등록해 주세요.")
else:
    st.sidebar.success("OPENAI_API_KEY 감지됨")

# ---- 업로드 & 파라미터 ----
uploaded_files = st.file_uploader("📄 참고할 파일 업로드 (PDF/TXT, 다중 선택 가능)", type=["pdf", "txt"], accept_multiple_files=True)
col1, col2 = st.columns([1,1])
with col1:
    chunk_size = st.number_input("청크 크기", min_value=200, max_value=3000, step=100, value=1000)
with col2:
    chunk_overlap = st.number_input("청크 중첩", min_value=0, max_value=800, step=50, value=150)

build_clicked = st.button("🔨 인덱스 구축 (BM25)")

# ---- 유틸 ----
def read_pdf(file) -> str:
    pdf = PdfReader(file)
    out = []
    for p in pdf.pages:
        out.append(p.extract_text() or "")
    return "\n".join(out)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def load_documents(files) -> List[Document]:
    docs = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            text = read_pdf(f)
        elif name.lower().endswith(".txt"):
            text = read_txt(f)
        else:
            continue
        docs.append(Document(page_content=text, metadata={"source": name}))
    return docs

def build_bm25_retriever(docs: List[Document], chunk_size:int, chunk_overlap:int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 4
    return retriever

if build_clicked:
    if not uploaded_files:
        st.warning("먼저 파일을 업로드하세요.")
    else:
        with st.spinner("인덱스 구축 중..."):
            try:
                docs = load_documents(uploaded_files)
                st.session_state.retriever = build_bm25_retriever(docs, chunk_size, chunk_overlap)
                st.success("완료! 이제 질문을 입력해 보세요.")
            except Exception as e:
                st.error(f"인덱스 구축 오류: {e}")

# ---- 질문 ----
st.subheader("💬 질문하기")
q = st.text_input("예: 이 문서의 핵심 요약은? / 중요한 수치와 근거는?")
top_k = st.slider("검색 문서 수 (k)", min_value=1, max_value=10, value=4)
ask = st.button("질문 보내기")

def build_chain(k:int):
    # BM25Retriever는 Retriever 인터페이스를 구현하므로 동일하게 사용 가능
    st.session_state.retriever.k = int(k)
    template = """
당신은 업로드된 자료를 바탕으로 정확하고 간결하게 한국어로 답변하는 조교입니다.
반드시 답변 끝에 '근거:'를 붙이고 출처 파일명을 나열하세요.
사용자 질문: {question}
한국어 답변:"""
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa

if ask:
    if st.session_state.retriever is None:
        st.warning("먼저 '인덱스 구축 (BM25)'을 실행해 주세요.")
    elif not q.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            try:
                qa = build_chain(top_k)
                result = qa({"query": q})
                answer = result["result"]

                # 근거 표시 (간단)
                srcs = []
                try:
                    docs = st.session_state.retriever.get_relevant_documents(q)
                    srcs = [d.metadata.get("source", "unknown") for d in docs[:top_k]]
                except Exception:
                    pass

                st.session_state.history.append(("user", q))
                st.session_state.history.append(("bot", answer, srcs))
            except Exception as e:
                st.error(f"질문 처리 오류: {e}")

# ---- 대화 표시 ----
st.subheader("🧾 대화 기록")
for item in st.session_state.history:
    role = item[0]
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user"><b>나</b><br>{item[1]}</div>', unsafe_allow_html=True)
    else:
        ans, srcs = item[1], item[2]
        st.markdown(f'<div class="chat-bubble-bot"><b>봇</b><br>{ans}</div>', unsafe_allow_html=True)
        if srcs:
            with st.expander("🔎 참고/근거 문서"):
                for s in srcs:
                    st.markdown(f"- `{s}`")

# ---- 푸터 ----
import time as _t
st.markdown(
    '<p class="tiny">© {year} 중간고사용 RAG 초안정판 | Streamlit Cloud 배포 시 Secrets에 OPENAI_API_KEY 등록 필수.</p>'.format(year=_t.strftime("%Y")),
    unsafe_allow_html=True
)